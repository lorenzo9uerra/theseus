import csv
import os

import numpy as np
from matplotlib import pyplot as plt

import wandb
from utils.utils import log, read_node_table


def _get_reapr_dir(root_dir: str) -> str:
    """Returns the absolute path to the REAPR ground truth directory if present."""
    candidate = os.path.join(
        root_dir, "ground_truth", "reapr-ground-truth", "darpa-tc-engagement3"
    )
    return candidate if os.path.isdir(candidate) else ""


def _get_reapr_csv_path(config) -> str | None:
    """Gets the REAPR CSV path for the specific dataset in config."""
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    reapr_dir = _get_reapr_dir(repo_root)
    if not reapr_dir:
        return None

    csv_name = f"{config.dataset_info.name.lower().split('_')[0]}_labels.csv"
    if not csv_name:
        return None

    csv_path = os.path.join(reapr_dir, csv_name)
    return csv_path if os.path.isfile(csv_path) else None


def _parse_reapr_labels_by_attack(csv_path: str) -> dict[str, dict[str, set[str]]]:
    """Parses REAPR CSVs into a mapping of attack chains to sets of node UUIDs."""
    attack_to_uuids = {}

    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        first = True
        for row in reader:
            if not row:
                continue
            row = [c.strip() for c in row]

            # Skip header
            if first:
                if row[0].lower() == "attack_chain" or row[1].lower() == "uuid":
                    first = False
                    continue
                first = False

            # Schema check: need at least attack_chain, uuid, ..., label
            if len(row) < 4:
                continue

            attack_chain, uuid, label = row[0], row[1], row[-1].lower()

            if label in {"attack", "contaminated"} and uuid and attack_chain:
                if attack_chain not in attack_to_uuids:
                    attack_to_uuids[attack_chain] = {
                        "attack": set(),
                        "contaminated": set(),
                    }
                attack_to_uuids[attack_chain][label].add(uuid)

    return attack_to_uuids


def log_image_to_wandb(out_file, log_key):
    """Logs an image file to WandB, converting SVG/PDF to PNG if necessary."""
    if wandb.run is None:
        return

    if out_file.endswith((".svg", ".pdf")):
        png_file = out_file.replace(".svg", ".png").replace(".pdf", ".png")
        if out_file.endswith(".svg"):
            plt.savefig(png_file, dpi=300, format="png")

        if os.path.exists(png_file):
            wandb.log({log_key: wandb.Image(png_file)})
        else:
            log(f"Skipping WandB log for {out_file} (PNG conversion failed/not found)")
    else:
        wandb.log({log_key: wandb.Image(out_file)})


def compute_adp_score(
    scores, nodes, node_to_attack_ids, labels, out_file=None, plot=True
):
    """
    Computes the Attack Detection Precision (ADP) curve and score.
    ADP measures the % of unique attacks detected vs. Precision across thresholds.
    """
    # Sort by anomaly score descending
    sorted_indices = np.argsort(scores)[::-1]
    sorted_nodes = [nodes[i] for i in sorted_indices]
    sorted_labels = [labels[i] for i in sorted_indices]

    # Identify total unique attacks in this evaluation set
    attacks_in_eval_set = set()
    for node in nodes:
        if node in node_to_attack_ids:
            attacks_in_eval_set.update(node_to_attack_ids[node])

    total_attacks = len(attacks_in_eval_set)
    if total_attacks == 0:
        log("Warning: No attacks found in this evaluation set")
        return 0.0

    detected_attacks = set()
    precisions = []
    detected_percentages = []
    tp, fp = 0, 0

    # Sweep threshold
    for node, label in zip(sorted_nodes, sorted_labels, strict=False):
        if label == 1:
            tp += 1
            if node in node_to_attack_ids:
                detected_attacks.update(node_to_attack_ids[node])
        else:
            fp += 1

        if tp + fp > 0:
            precision = tp / (tp + fp)
            detected_percentage = (len(detected_attacks) / total_attacks) * 100
            precisions.append(precision)
            detected_percentages.append(detected_percentage)

    if not precisions:
        log("Warning: No valid threshold points found")
        return 0.0

    # Integrate area under curve
    sorted_pairs = sorted(zip(precisions, detected_percentages, strict=False))
    precisions_sorted = [p for p, _ in sorted_pairs]
    detected_sorted = [d for _, d in sorted_pairs]
    area_under_curve = np.trapezoid(detected_sorted, precisions_sorted) / 100.0

    if plot and out_file is not None:
        try:
            plt.figure(figsize=(10, 6))
            plt.plot(
                precisions_sorted,
                detected_sorted,
                color="b",
                linewidth=2,
                label=f"ADP Score = {area_under_curve:.3f}",
            )
            plt.fill_between(
                precisions_sorted, detected_sorted, color="blue", alpha=0.2
            )
            plt.xlabel("Precision (High = Few False Positives)", fontsize=12)
            plt.ylabel("% of Detected Attacks", fontsize=12)
            plt.title("Attack Detection vs Precision", fontsize=14)
            plt.legend(loc="lower right", fontsize=11)
            plt.xlim(0, 1)
            plt.ylim(0, 100.5)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(out_file, dpi=300, bbox_inches="tight")
            log_image_to_wandb(out_file, "adp_img")
        except Exception as e:
            log(f"Error while generating ADP plot: {e}")
        finally:
            plt.close()

    return area_under_curve


def get_uuid_to_index_id_mapping(config):
    """Builds a map from node UUIDs to integer index IDs from dataset tables."""
    data_dir = os.path.join(config.data_dir, config.dataset)
    uuid_to_index_id = {}

    for table_name in ["file_node_table", "netflow_node_table", "process_node_table"]:
        df = read_node_table(data_dir, table_name, columns=["index_id", "node_uuid"])
        if df is not None:
            for row in df.iter_rows(named=True):
                uuid_to_index_id[row["node_uuid"]] = row["index_id"]
            del df

    return uuid_to_index_id


def get_excluded_node_ids(config) -> set[int]:
    excluded_attack_chains = set(
        getattr(getattr(config, "dataset_info", None), "excluded_attack_chains", [])
        or []
    )

    reapr_csv = _get_reapr_csv_path(config)
    if not reapr_csv:
        return set()

    uuid_to_index_id = get_uuid_to_index_id_mapping(config)
    attack_to_uuids = _parse_reapr_labels_by_attack(reapr_csv)

    excluded_uuids: set[str] = set()
    for chain_name in excluded_attack_chains:
        chain_data = attack_to_uuids.get(chain_name)
        if not chain_data:
            continue
        excluded_uuids.update(chain_data.get("attack", set()) or set())
        excluded_uuids.update(chain_data.get("contaminated", set()) or set())

    excluded_node_ids: set[int] = set()
    missing = 0
    for node_uuid in excluded_uuids:
        idx = uuid_to_index_id.get(node_uuid)
        if idx is None:
            missing += 1
            continue
        excluded_node_ids.add(int(idx))

    if excluded_node_ids:
        log(
            f"Excluding {len(excluded_node_ids)} node(s) from evaluation metrics: {sorted(excluded_attack_chains)}"
        )
    if missing:
        log(
            f"WARNING: {missing}/{len(excluded_uuids)} excluded UUIDs not found in node tables"
        )

    return excluded_node_ids


def get_ground_truth(config):
    """Loads REAPR ground truth labels and maps them to node IDs."""
    uuid_to_index_id = get_uuid_to_index_id_mapping(config)
    attack_metadata = {}

    reapr_csv = _get_reapr_csv_path(config)
    if not reapr_csv:
        raise ValueError(
            f"REAPR ground truth not available for '{config.dataset_info.name}'"
        )

    attack_to_uuids = _parse_reapr_labels_by_attack(reapr_csv)

    for chain_name, chain_data in attack_to_uuids.items():
        attack_node_ids = set()
        contaminated_node_ids = set()

        # Map UUIDs to indices
        for uuid_set, target_set, label_type in [
            (chain_data["attack"], attack_node_ids, "attack"),
            (chain_data["contaminated"], contaminated_node_ids, "contaminated"),
        ]:
            missing_count = 0
            for node_uuid in uuid_set:
                if node_uuid in uuid_to_index_id:
                    target_set.add(int(uuid_to_index_id[node_uuid]))
                else:
                    missing_count += 1

            if missing_count > 0:
                log(
                    f"WARNING: {missing_count}/{len(uuid_set)} {label_type} UUIDs from '{chain_name}' not found"
                )

        log(
            f"Attack ('{chain_name}'): {len(attack_node_ids)} attack nodes, {len(contaminated_node_ids)} contaminated nodes"
        )

        attack_metadata[chain_name] = {
            "nids": attack_node_ids,
            "contaminated_nids": contaminated_node_ids,
        }

    return attack_metadata


def plot_anomaly_score_distribution(
    scores, y_truth, out_file, threshold=None, nodes=None, node_to_attack_ids=None
):
    benign_scores = np.array(
        [score for score, label in zip(scores, y_truth, strict=False) if label == 0]
    )

    # Log-spaced bins
    scores_arr = np.array(scores)

    if np.any(scores_arr > 0):
        min_score = np.min(scores_arr[scores_arr > 0])
        real_max = np.max(scores_arr)
    else:
        min_score = 1e-6
        real_max = 1e-6

    max_score = real_max * 1.5

    if max_score <= min_score:
        max_score = min_score * 10

    bins = np.logspace(np.log10(min_score), np.log10(max_score), 50)

    plt.figure(figsize=(14, 5))

    if len(benign_scores) > 0:
        plt.hist(
            benign_scores,
            bins=bins,
            density=True,
            alpha=0.6,
            label="Benign",
            color="green",
        )

    if nodes is not None and node_to_attack_ids is not None:
        # Group malicious by attack type
        attack_scores = {}
        for score, label, node in zip(scores, y_truth, nodes, strict=False):
            if label == 1:
                # Handle int/str mismatch
                node_key = None
                if node in node_to_attack_ids:
                    node_key = node
                elif str(node) in node_to_attack_ids:
                    node_key = str(node)
                elif isinstance(node, str) and int(node) in node_to_attack_ids:
                    node_key = int(node)

                if node_key is not None:
                    for attack_id in node_to_attack_ids[node_key]:
                        attack_scores.setdefault(attack_id, []).append(score)

        tab10 = plt.color_sequences["tab10"]
        attack_colors = [
            tab10[0],
            tab10[1],
            tab10[3],
            tab10[4],
            tab10[5],
            tab10[6],
            tab10[7],
            tab10[8],
            tab10[9],
        ]

        for i, (attack_id, s_list) in enumerate(sorted(attack_scores.items())):
            plt.hist(
                s_list,
                bins=bins,
                density=True,
                alpha=0.6,
                label=f"{attack_id} (n={len(s_list)})",
                color=attack_colors[i % len(attack_colors)],
            )
    else:
        malicious_scores = np.array(
            [score for score, label in zip(scores, y_truth, strict=False) if label == 1]
        )
        if len(malicious_scores) > 0:
            plt.hist(
                malicious_scores,
                bins=bins,
                density=True,
                alpha=0.6,
                label=f"Malicious (n={len(malicious_scores)})",
                color="red",
            )

    if threshold is not None and not np.isinf(threshold):
        plt.axvline(x=threshold, color="black", linestyle="--", linewidth=2)

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Anomaly Score", fontsize=20)
    plt.ylabel("Density", fontsize=20)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=16)
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.tight_layout()

    # Save PDF and PNG
    pdf_file = (
        out_file.replace(".png", ".pdf")
        if out_file.endswith(".png")
        else out_file + ".pdf"
    )
    plt.savefig(pdf_file, format="pdf", dpi=300, bbox_inches="tight")

    png_file = (
        out_file.replace(".pdf", ".png") if out_file.endswith(".pdf") else out_file
    )
    plt.savefig(png_file, format="png", dpi=300, bbox_inches="tight")

    plt.close()

    if wandb.run is not None:
        try:
            wandb.log({"anomaly_score_distribution": wandb.Image(png_file)})
        except Exception as e:
            log(f"Warning: Could not log image to wandb: {e}")


def plot_scores_neat(
    scores, y_truth, nodes, node_to_attack_ids, out_file, threshold=None
):
    # Downsample benign
    scores_0 = [
        score
        for i, (score, label) in enumerate(zip(scores, y_truth, strict=False))
        if label == 0 and (score > 0.9 or (score <= 0.9 and i % 500 == 1))
    ]
    scores_1 = [
        (score, node)
        for score, label, node in zip(scores, y_truth, nodes, strict=False)
        if label == 1
    ]

    center_coef = 0.2
    y_zeros = [center_coef] * len(scores_0)
    y_ones = [1 - center_coef] * len(scores_1)

    plt.figure(figsize=(6, 1.5))
    plt.scatter(scores_0, y_zeros, color="green", label="Benign", rasterized=True)

    labels, colors, s_list = [], [], []
    for score, node in scores_1:
        s_list.append(score)
        attack_type = list(node_to_attack_ids.get(node))[0]
        labels.append(f"Attack {attack_type}")
        colors.append("red")

    plt.scatter(s_list, y_ones, color=colors, label=labels, rasterized=True)

    if threshold is not None:
        plt.axvline(
            x=threshold,
            color="black",
            linestyle="-",
            linewidth=2,
            label=f"Threshold: {threshold}",
        )

    plt.xlabel("Node anomaly scores")
    plt.yticks([center_coef, 1 - center_coef], ["Benign", "Malicious"])
    plt.ylim(-0.1, 1.1)
    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    log_image_to_wandb(out_file, "neat_scores_img_file")
    plt.close()
