import os
import pickle as pkl
import random

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

_MAGIC_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_EVAL_RESULT_DIR = os.path.join(_MAGIC_ROOT, "eval_result")

# Optional dependencies
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def _checked_zip(*sequences):
    lengths = [len(seq) for seq in sequences]
    if len(set(lengths)) != 1:
        raise ValueError(f"Mismatched sequence lengths: {lengths}")
    return zip(*sequences)  # noqa: B905


def _select_max_benign_threshold(scores, labels):
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels)

    finite_mask = np.isfinite(scores)
    if not finite_mask.any():
        return float("inf")

    scores = scores[finite_mask]
    labels = labels[finite_mask]
    benign_scores = scores[labels == 0]
    if benign_scores.size > 0:
        return float(benign_scores.max())

    print(
        "  Warning: no benign validation entities available for threshold calibration; "
        "using the maximum finite validation score."
    )
    return float(scores.max())


def _threshold_predictions(scores, threshold):
    if np.isinf(threshold):
        return np.zeros_like(scores, dtype=int)
    return (scores > threshold).astype(int)


def aggregate_to_entity_level(scores, labels, uuids, node_to_attack_chain=None):
    """Aggregate repeated UUID observations with max score and max label."""
    entity_map = {}  # uuid -> {'score': max, 'label': max, 'attack_chain': str}

    for i, (score, label, uuid) in enumerate(_checked_zip(scores, labels, uuids)):
        if uuid not in entity_map:
            entity_map[uuid] = {
                "score": -float("inf"),
                "label": 0,
                "attack_chain": None,
            }

        if score > entity_map[uuid]["score"]:
            entity_map[uuid]["score"] = score

        if label > entity_map[uuid]["label"]:
            entity_map[uuid]["label"] = label

        # Track attack chain if available
        if node_to_attack_chain and i in node_to_attack_chain:
            entity_map[uuid]["attack_chain"] = node_to_attack_chain[i]

    agg_scores = []
    agg_labels = []
    agg_uuids = []
    uuid_to_attack_chain = {}

    for uuid, data in entity_map.items():
        agg_scores.append(data["score"])
        agg_labels.append(data["label"])
        agg_uuids.append(uuid)
        if data["attack_chain"]:
            uuid_to_attack_chain[len(agg_uuids) - 1] = data["attack_chain"]

    return (
        np.array(agg_scores),
        np.array(agg_labels, dtype=int),
        agg_uuids,
        uuid_to_attack_chain,
    )


def find_best_threshold_mcc(scores, labels):
    """
    Finds the threshold that maximizes Matthews Correlation Coefficient (MCC).
    Uses a vectorized approach (sorting + cumulative sum) for O(n log n) efficiency.
    """
    n = len(labels)
    n_pos = int(labels.sum())
    n_neg = n - n_pos

    if n_pos == 0 or n_neg == 0:
        return np.median(scores), 0.0

    # Sort descending by score
    sorted_indices = np.argsort(scores)[::-1]
    sorted_labels = labels[sorted_indices]
    sorted_scores = scores[sorted_indices]

    # Compute cumulative confusion matrix stats
    cum_tp = np.cumsum(sorted_labels)
    cum_fp = np.arange(1, n + 1) - cum_tp
    fn = n_pos - cum_tp
    tn = n_neg - cum_fp

    # Vectorized MCC calculation
    numerator = cum_tp * tn - cum_fp * fn
    denominator = np.sqrt(
        (cum_tp + cum_fp).astype(np.float64)
        * (cum_tp + fn).astype(np.float64)
        * (tn + cum_fp).astype(np.float64)
        * (tn + fn).astype(np.float64)
        + 1e-10
    )
    mcc = numerator / denominator

    best_idx = np.argmax(mcc)
    return sorted_scores[best_idx], mcc[best_idx]


def compute_adp_score(scores, labels, node_to_attack_chain, out_file=None, plot=True):
    """
    Computes the Attack Detection Precision (ADP) score.
    ADP is the area under the curve of '% Unique Attacks Detected' vs 'Precision'.
    """
    attacks_in_eval_set = set(node_to_attack_chain.values())
    total_attacks = len(attacks_in_eval_set)

    if total_attacks == 0:
        print("  ADP Warning: No attacks found in this evaluation set")
        return 0.0

    print(
        f"  ADP: Found {total_attacks} unique attack chain(s): {sorted(attacks_in_eval_set)}"
    )

    # Sort nodes by score descending
    sorted_indices = np.argsort(scores)[::-1]
    sorted_labels = labels[sorted_indices]

    detected_attacks = set()
    precisions = []
    detected_percentages = []
    tp = 0
    fp = 0

    # Sweep threshold from high to low
    for orig_idx, label in _checked_zip(sorted_indices, sorted_labels):
        if label == 1:
            tp += 1
            if orig_idx in node_to_attack_chain:
                detected_attacks.add(node_to_attack_chain[orig_idx])
        else:
            fp += 1

        if tp + fp > 0:
            precision = tp / (tp + fp)
            detected_percentage = (len(detected_attacks) / total_attacks) * 100
            precisions.append(precision)
            detected_percentages.append(detected_percentage)

    if len(precisions) == 0:
        print("  ADP Warning: No valid threshold points found")
        return 0.0

    # Sort by precision for integration
    sorted_pairs = sorted(_checked_zip(precisions, detected_percentages))
    precisions_sorted = [p for p, _ in sorted_pairs]
    detected_sorted = [d for _, d in sorted_pairs]

    area_under_curve = np.trapz(detected_sorted, precisions_sorted) / 100.0

    if plot and out_file is not None and MATPLOTLIB_AVAILABLE:
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
            plt.xlabel("Precision")
            plt.ylabel("% of Detected Attacks")
            plt.title("Attack Detection vs Precision (ADP Curve)")
            plt.legend(loc="lower right")
            plt.xlim(0, 1)
            plt.ylim(0, 100.5)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(out_file, dpi=300, bbox_inches="tight")
            print(f"  ADP plot saved to {out_file}")
        except Exception as e:
            print(f"  ADP Warning: Error generating plot: {e}")
        finally:
            plt.close()

    return area_under_curve


def log_image_to_wandb(file_path, key):
    """Helper function to log images to wandb if available."""
    if WANDB_AVAILABLE and wandb.run is not None:
        try:
            wandb.log({key: wandb.Image(file_path)})
        except Exception as e:
            print(f"  Warning: Failed to log image to wandb: {e}")


def plot_anomaly_score_distribution(
    scores, y_truth, out_file, threshold=None, nodes=None, node_to_attack_chain=None
):
    """Plots log-scale histograms of anomaly scores for benign vs. malicious nodes."""
    if not MATPLOTLIB_AVAILABLE:
        print("  Warning: matplotlib not available, skipping score distribution plot")
        return

    benign_scores = np.array(
        [score for score, label in _checked_zip(scores, y_truth) if label == 0]
    )

    # Calculate global bins for consistent log scaling
    scores_arr = np.array(scores)
    min_score = np.min(scores_arr[scores_arr > 0]) if np.any(scores_arr > 0) else 1e-6
    max_score = np.max(scores_arr)
    if max_score <= min_score:
        max_score = min_score + 1.0
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

    # Plot malicious scores (grouped by attack type if metadata available)
    if nodes is not None and node_to_attack_chain is not None:
        attack_scores = {}
        unknown_malicious_scores = []

        for score, label, node in _checked_zip(scores, y_truth, nodes):
            if label == 1:
                if node in node_to_attack_chain:
                    attack_chain = node_to_attack_chain[node]
                    attack_scores.setdefault(attack_chain, []).append(score)
                else:
                    unknown_malicious_scores.append(score)

        # Plot specific attack chains
        colors = [
            "#1f77b4",
            "#ff7f0e",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
            "#17becf",
        ]
        for i, (attack_chain, score_list) in enumerate(sorted(attack_scores.items())):
            plt.hist(
                score_list,
                bins=bins,
                density=True,
                alpha=0.6,
                label=f"{attack_chain} (n={len(score_list)})",
                color=colors[i % len(colors)],
            )

        if unknown_malicious_scores:
            plt.hist(
                unknown_malicious_scores,
                bins=bins,
                density=True,
                alpha=0.6,
                label=f"Other Malicious (n={len(unknown_malicious_scores)})",
                color="black",
            )
    else:
        # Fallback: all malicious together
        malicious_scores = np.array(
            [score for score, label in _checked_zip(scores, y_truth) if label == 1]
        )
        if len(malicious_scores) > 0:
            plt.hist(
                malicious_scores,
                bins=bins,
                density=True,
                alpha=0.6,
                label="Malicious",
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

    plt.savefig(out_file, format="pdf", dpi=300, bbox_inches="tight")
    png_file = out_file.replace(".pdf", ".png")
    plt.savefig(png_file, format="png", dpi=300, bbox_inches="tight")
    plt.close()
    log_image_to_wandb(out_file, "anomaly_score_distribution")


def evaluate_entity_level_with_validation(
    dataset,
    x_train,
    x_test,
    y_test,
    val_ratio=0.2,
    random_state=42,
    x_val=None,
    y_val=None,
):
    """Evaluate entity-level detection with validation-based threshold selection."""
    print("\n" + "=" * 80)
    if x_val is not None:
        print("DEDICATED VALIDATION SET")
    else:
        print("VALIDATION-BASED THRESHOLD SELECTION")
    print("=" * 80)

    if x_val is not None:
        x_final_test = x_test
        y_final_test = y_test

        if y_val is None:
            raise ValueError(
                "y_val must be provided when using dedicated validation set (x_val)"
            )

        print("\nData split:")
        print(f"  Training: {len(x_train)} (benign only, for fitting KNN)")
        print(
            f"  Validation: {len(x_val)} ({np.sum(y_val == 0):.0f} benign, {np.sum(y_val == 1):.0f} attack)"
        )
        print(
            f"  Test: {len(y_final_test)} ({np.sum(y_final_test == 0):.0f} benign, {np.sum(y_final_test == 1):.0f} attack)"
        )
    else:
        # Split test data into validation and test (stratified)
        test_indices = np.arange(len(y_test))
        val_indices, final_test_indices = train_test_split(
            test_indices,
            test_size=(1 - val_ratio),
            stratify=y_test,
            random_state=random_state,
        )

        x_val, y_val = x_test[val_indices], y_test[val_indices]
        x_final_test, y_final_test = (
            x_test[final_test_indices],
            y_test[final_test_indices],
        )

        print("\nData split:")
        print(f"  Training: {len(x_train)} (benign only, for fitting KNN)")
        print(
            f"  Validation: {len(y_val)} ({np.sum(y_val == 0):.0f} benign, {np.sum(y_val == 1):.0f} attack)"
        )
        print(
            f"  Test: {len(y_final_test)} ({np.sum(y_final_test == 0):.0f} benign, {np.sum(y_final_test == 1):.0f} attack)"
        )

    # Normalize using training statistics
    x_train_mean, x_train_std = x_train.mean(axis=0), x_train.std(axis=0)
    x_train_norm = (x_train - x_train_mean) / (x_train_std + 1e-9)
    x_val_norm = (x_val - x_train_mean) / (x_train_std + 1e-9)
    x_test_norm = (x_final_test - x_train_mean) / (x_train_std + 1e-9)

    # For large datasets, subsample training data for KNN to reduce computation time
    max_knn_train_samples = 100000
    if len(x_train_norm) > max_knn_train_samples:
        print(
            f"\nSubsampling training data for KNN: {len(x_train_norm):,} -> {max_knn_train_samples:,} samples"
        )
        idx_subsample = np.random.choice(
            len(x_train_norm), max_knn_train_samples, replace=False
        )
        x_train_knn = x_train_norm[idx_subsample]
    else:
        x_train_knn = x_train_norm

    # Fit KNN
    n_neighbors = 200 if dataset in ["cadets", "theia"] else 10
    print(f"\nFitting KNN with k={n_neighbors} on {len(x_train_knn):,} samples...")
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=-1)
    nbrs.fit(x_train_knn)

    # Compute baseline from training
    idx = list(range(len(x_train_knn)))
    random.shuffle(idx)
    train_dist, _ = nbrs.kneighbors(
        x_train_knn[idx][: min(50000, len(x_train_knn))], n_neighbors=n_neighbors
    )
    mean_distance = train_dist.mean()
    print(f"  Mean training distance: {mean_distance:.4f}")

    # Helper function for batched KNN computation
    batch_size = 100000

    def compute_scores_batched(x_norm, desc=""):
        """Compute KNN anomaly scores in batches for memory efficiency."""
        n_samples = len(x_norm)
        if n_samples <= batch_size:
            dist, _ = nbrs.kneighbors(x_norm, n_neighbors=n_neighbors)
            return dist.mean(axis=1) / mean_distance

        print(f"  Computing {desc} scores in batches ({n_samples:,} samples)...")
        scores = np.zeros(n_samples)
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            dist, _ = nbrs.kneighbors(x_norm[start:end], n_neighbors=n_neighbors)
            scores[start:end] = dist.mean(axis=1) / mean_distance
            print(f"    Processed {end:,}/{n_samples:,} samples")
        return scores

    print("\nSelecting threshold on validation set...")
    val_scores = compute_scores_batched(x_val_norm, "validation")

    best_threshold = _select_max_benign_threshold(val_scores, y_val)

    val_auc = roc_auc_score(y_val, val_scores)
    val_ap = average_precision_score(y_val, val_scores) if y_val.sum() > 0 else 0.0
    y_val_pred = _threshold_predictions(val_scores, best_threshold)
    val_tp = np.sum((y_val == 1) & (y_val_pred == 1))
    val_fp = np.sum((y_val == 0) & (y_val_pred == 1))
    val_tn = np.sum((y_val == 0) & (y_val_pred == 0))
    val_fn = np.sum((y_val == 1) & (y_val_pred == 0))
    val_prec = val_tp / (val_tp + val_fp) if (val_tp + val_fp) > 0 else 0
    val_rec = val_tp / (val_tp + val_fn) if (val_tp + val_fn) > 0 else 0
    val_f1 = (
        2 * val_prec * val_rec / (val_prec + val_rec) if (val_prec + val_rec) > 0 else 0
    )

    val_source = "dedicated validation set" if x_val is not None else "split from test"
    print(f"\nValidation Results (from {val_source}):")
    print(f"  ROC-AUC: {val_auc:.6f}")
    print(f"  AP: {val_ap:.6f}")
    print(f"  F1: {val_f1:.6f} (Prec: {val_prec:.6f}, Rec: {val_rec:.6f})")
    print(f"  Threshold (max benign validation score): {best_threshold:.6f}")
    print(f"  TP: {val_tp}, FP: {val_fp}, TN: {val_tn}, FN: {val_fn}")

    print("\nEvaluating on held-out test set...")
    test_scores = compute_scores_batched(x_test_norm, "test")

    y_test_pred = _threshold_predictions(test_scores, best_threshold)
    test_auc = roc_auc_score(y_final_test, test_scores)

    test_ap = (
        average_precision_score(y_final_test, test_scores)
        if y_final_test.sum() > 0
        else 0.0
    )

    test_tp = np.sum((y_final_test == 1) & (y_test_pred == 1))
    test_fp = np.sum((y_final_test == 0) & (y_test_pred == 1))
    test_tn = np.sum((y_final_test == 0) & (y_test_pred == 0))
    test_fn = np.sum((y_final_test == 1) & (y_test_pred == 0))
    test_prec = test_tp / (test_tp + test_fp) if (test_tp + test_fp) > 0 else 0
    test_rec = test_tp / (test_tp + test_fn) if (test_tp + test_fn) > 0 else 0
    test_f1 = (
        2 * test_prec * test_rec / (test_prec + test_rec)
        if (test_prec + test_rec) > 0
        else 0
    )

    print("\n" + "=" * 80)
    print("FINAL TEST RESULTS")
    print("=" * 80)
    print(f"Threshold: {best_threshold:.6f} (max benign validation score)")
    print(f"ROC-AUC: {test_auc:.6f}")
    print(f"AP: {test_ap:.6f}")
    print(f"F1: {test_f1:.6f}")
    print(f"PRECISION: {test_prec:.6f}")
    print(f"RECALL: {test_rec:.6f}")
    print(f"TN: {test_tn}, FN: {test_fn}, TP: {test_tp}, FP: {test_fp}")
    print("=" * 80)

    os.makedirs(_EVAL_RESULT_DIR, exist_ok=True)
    save_dict = {
        "dataset": dataset,
        "val_ratio": val_ratio,
        "threshold": best_threshold,
        "validation": {
            "auc": val_auc,
            "ap": val_ap,
            "f1": val_f1,
            "precision": val_prec,
            "recall": val_rec,
        },
        "test": {
            "auc": test_auc,
            "ap": test_ap,
            "f1": test_f1,
            "precision": test_prec,
            "recall": test_rec,
        },
    }
    with open(
        os.path.join(_EVAL_RESULT_DIR, f"{dataset}_validation_results.pkl"), "wb"
    ) as f:
        pkl.dump(save_dict, f)

    return test_auc, 0.0, best_threshold, test_scores


def strict_evaluation(
    dataset,
    x_train,
    x_val,
    x_test,
    val_attack_idx,
    val_contaminated_idx,
    test_attack_idx,
    test_contaminated_idx,
    n_val_nodes,
    n_test_nodes,
    val_node_to_attack_chain_attack_only=None,
    val_node_to_attack_chain_all=None,
    test_node_to_attack_chain_attack_only=None,
    test_node_to_attack_chain_all=None,
    val_idx_to_uuid=None,
    test_idx_to_uuid=None,
    excluded_uuids=None,
    result_prefix=None,
):
    """
    Entity-level evaluation under strict attack-chain semantics only.
    Contaminated nodes are masked out of calibration and test metrics.
    """
    print("\n" + "=" * 80)
    print("ENTITY-LEVEL STRICT EVALUATION")
    print("=" * 80)
    result_stem = result_prefix or dataset
    os.makedirs(_EVAL_RESULT_DIR, exist_ok=True)

    # Normalize data
    x_train_mean, x_train_std = x_train.mean(axis=0), x_train.std(axis=0)
    x_train_norm = (x_train - x_train_mean) / (x_train_std + 1e-9)
    x_val_norm = (x_val - x_train_mean) / (x_train_std + 1e-9)
    x_test_norm = (x_test - x_train_mean) / (x_train_std + 1e-9)

    # KNN setup
    max_knn_samples = 100000
    if len(x_train_norm) > max_knn_samples:
        idx_sub = np.random.choice(len(x_train_norm), max_knn_samples, replace=False)
        x_train_knn = x_train_norm[idx_sub]
    else:
        x_train_knn = x_train_norm

    n_neighbors = 200 if dataset in ["cadets", "theia"] else 10
    print(f"Fitting KNN (k={n_neighbors})...")
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=-1).fit(x_train_knn)

    idx = list(range(len(x_train_knn)))
    random.shuffle(idx)
    train_dist, _ = nbrs.kneighbors(
        x_train_knn[idx][: min(50000, len(x_train_knn))], n_neighbors=n_neighbors
    )
    mean_distance = train_dist.mean()

    def compute_scores_batched(x_norm, desc=""):
        batch_size = 100000
        n_samples = len(x_norm)
        if n_samples <= batch_size:
            dist, _ = nbrs.kneighbors(x_norm, n_neighbors=n_neighbors)
            return dist.mean(axis=1) / mean_distance
        scores = np.zeros(n_samples)
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            dist, _ = nbrs.kneighbors(x_norm[start:end], n_neighbors=n_neighbors)
            scores[start:end] = dist.mean(axis=1) / mean_distance
        return scores

    # Compute instance-level scores
    val_scores_inst = compute_scores_batched(x_val_norm, "validation")
    test_scores_inst = compute_scores_batched(x_test_norm, "test")

    # Create instance-level strict labels and contamination masks.
    val_labels_inst = np.zeros(n_val_nodes)
    for idx in val_attack_idx:
        if idx < n_val_nodes:
            val_labels_inst[idx] = 1

    val_contaminated_mask_inst = np.zeros(n_val_nodes, dtype=bool)
    for idx in val_contaminated_idx:
        if idx < n_val_nodes:
            val_contaminated_mask_inst[idx] = True

    test_labels_inst = np.zeros(n_test_nodes)
    for idx in test_attack_idx:
        if idx < n_test_nodes:
            test_labels_inst[idx] = 1

    test_contaminated_mask_inst = np.zeros(n_test_nodes, dtype=bool)
    for idx in test_contaminated_idx:
        if idx < n_test_nodes:
            test_contaminated_mask_inst[idx] = True

    # Entity-level aggregation
    if val_idx_to_uuid and test_idx_to_uuid:
        print("\nAggregating to entity level (instance -> unique UUID)...")
        val_uuids = [val_idx_to_uuid.get(i, f"unknown_{i}") for i in range(n_val_nodes)]
        test_uuids = [
            test_idx_to_uuid.get(i, f"unknown_{i}") for i in range(n_test_nodes)
        ]

        val_scores, val_labels, val_uuid_list, val_uuid_to_chain = (
            aggregate_to_entity_level(
                val_scores_inst,
                val_labels_inst,
                val_uuids,
                val_node_to_attack_chain_attack_only,
            )
        )
        test_scores, test_labels, test_uuid_list, test_uuid_to_chain = (
            aggregate_to_entity_level(
                test_scores_inst,
                test_labels_inst,
                test_uuids,
                test_node_to_attack_chain_attack_only,
            )
        )
        print(f"  Validation: {n_val_nodes} instances -> {len(val_scores)} entities")
        print(f"  Test: {n_test_nodes} instances -> {len(test_scores)} entities")

        if excluded_uuids:
            excluded_uuids = set(excluded_uuids)

            def _exclude_entities(scores, labels, uuid_list, uuid_to_chain, split_name):
                if len(uuid_list) == 0:
                    return scores, labels, uuid_list, uuid_to_chain

                keep_mask = np.array([u not in excluded_uuids for u in uuid_list])
                if keep_mask.all():
                    return scores, labels, uuid_list, uuid_to_chain

                kept_old_indices = [i for i, keep in enumerate(keep_mask) if keep]
                new_uuid_list = [uuid_list[i] for i in kept_old_indices]

                old_to_new = {
                    old_i: new_i for new_i, old_i in enumerate(kept_old_indices)
                }
                new_uuid_to_chain = {
                    old_to_new[old_i]: chain
                    for old_i, chain in (uuid_to_chain or {}).items()
                    if old_i in old_to_new
                }

                dropped = len(uuid_list) - len(new_uuid_list)
                if dropped > 0:
                    print(
                        f"  Excluded {dropped} entities from {split_name} evaluation (excluded attack chains)"
                    )

                return (
                    scores[keep_mask],
                    labels[keep_mask],
                    new_uuid_list,
                    new_uuid_to_chain,
                )

            val_scores, val_labels, val_uuid_list, val_uuid_to_chain = (
                _exclude_entities(
                    val_scores,
                    val_labels,
                    val_uuid_list,
                    val_uuid_to_chain,
                    "validation",
                )
            )
            test_scores, test_labels, test_uuid_list, test_uuid_to_chain = (
                _exclude_entities(
                    test_scores, test_labels, test_uuid_list, test_uuid_to_chain, "test"
                )
            )
    else:
        print("WARNING: No UUID mapping provided, using instance-level evaluation")
        val_scores, val_labels = val_scores_inst, val_labels_inst
        test_scores, test_labels = test_scores_inst, test_labels_inst
        test_uuid_to_chain = test_node_to_attack_chain_attack_only or {}

    # Strict attack-chain evaluation: mask contaminated nodes.
    print("\n" + "-" * 80)
    print("STRICT ATTACK CHAIN (Contaminated masked)")
    print("-" * 80)

    # For entity-level: need to identify which entities are contaminated-only
    if val_idx_to_uuid and test_idx_to_uuid:
        # Build sets of UUIDs that are attack vs contaminated
        val_attack_uuids = {
            val_idx_to_uuid.get(i) for i in val_attack_idx if i in val_idx_to_uuid
        }
        val_contam_uuids = {
            val_idx_to_uuid.get(i) for i in val_contaminated_idx if i in val_idx_to_uuid
        }
        test_attack_uuids = {
            test_idx_to_uuid.get(i) for i in test_attack_idx if i in test_idx_to_uuid
        }
        test_contam_uuids = {
            test_idx_to_uuid.get(i)
            for i in test_contaminated_idx
            if i in test_idx_to_uuid
        }

        # Mask: keep attack and benign, exclude contaminated-only
        val_mask = np.array(
            [u not in val_contam_uuids or u in val_attack_uuids for u in val_uuid_list]
        )
        test_mask = np.array(
            [
                u not in test_contam_uuids or u in test_attack_uuids
                for u in test_uuid_list
            ]
        )

        val_scores_eval = val_scores[val_mask]
        val_labels_eval = np.array(
            [
                1 if val_uuid_list[i] in val_attack_uuids else 0
                for i in range(len(val_uuid_list))
            ]
        )[val_mask]

        test_scores_eval = test_scores[test_mask]
        test_labels_eval = np.array(
            [
                1 if test_uuid_list[i] in test_attack_uuids else 0
                for i in range(len(test_uuid_list))
            ]
        )[test_mask]
        masked_test_uuid_list = [
            test_uuid_list[i] for i in range(len(test_uuid_list)) if test_mask[i]
        ]
    else:
        # Fallback to instance-level masking
        val_mask = np.ones(len(val_scores), dtype=bool)
        test_mask = np.ones(len(test_scores), dtype=bool)
        val_mask = ~val_contaminated_mask_inst
        test_mask = ~test_contaminated_mask_inst
        val_scores_eval = val_scores[val_mask]
        val_labels_eval = val_labels[val_mask]
        test_scores_eval = test_scores[test_mask]
        test_labels_eval = test_labels[test_mask]
        masked_test_uuid_list = []

    print(f"  After masking: val={len(val_scores_eval)}, test={len(test_scores_eval)}")

    threshold = _select_max_benign_threshold(val_scores_eval, val_labels_eval)
    val_auc = (
        roc_auc_score(val_labels_eval, val_scores_eval)
        if val_labels_eval.sum() > 0
        else 0.0
    )

    y_test_pred = _threshold_predictions(test_scores_eval, threshold)
    test_auc = (
        roc_auc_score(test_labels_eval, test_scores_eval)
        if test_labels_eval.sum() > 0
        else 0.0
    )
    test_ap = (
        average_precision_score(test_labels_eval, test_scores_eval)
        if test_labels_eval.sum() > 0
        else 0.0
    )

    test_tp = np.sum((test_labels_eval == 1) & (y_test_pred == 1))
    test_fp = np.sum((test_labels_eval == 0) & (y_test_pred == 1))
    test_tn = np.sum((test_labels_eval == 0) & (y_test_pred == 0))
    test_fn = np.sum((test_labels_eval == 1) & (y_test_pred == 0))
    test_prec = test_tp / (test_tp + test_fp) if (test_tp + test_fp) > 0 else 0
    test_rec = test_tp / (test_tp + test_fn) if (test_tp + test_fn) > 0 else 0
    test_f1 = (
        2 * test_prec * test_rec / (test_prec + test_rec)
        if (test_prec + test_rec) > 0
        else 0
    )
    test_fpr = test_fp / (test_fp + test_tn) if (test_fp + test_tn) > 0 else 0
    mcc_denom = np.sqrt(
        (test_tp + test_fp)
        * (test_tp + test_fn)
        * (test_tn + test_fp)
        * (test_tn + test_fn)
    )
    test_mcc = (test_tp * test_tn - test_fp * test_fn) / mcc_denom if mcc_denom > 0 else 0

    print(f"  F1: {test_f1:.4f}, Prec: {test_prec:.4f}, Rec: {test_rec:.4f}")
    print(f"  TP: {test_tp}, FP: {test_fp}, TN: {test_tn}, FN: {test_fn}")

    test_adp = 0.0
    test_eval_idx_to_chain = {}
    if val_idx_to_uuid and test_idx_to_uuid and test_node_to_attack_chain_attack_only:
        test_uuid_to_chain_attack_only = {}
        for idx, chain in test_node_to_attack_chain_attack_only.items():
            if idx in test_idx_to_uuid:
                uuid = test_idx_to_uuid[idx]
                test_uuid_to_chain_attack_only[uuid] = chain

        for new_idx, uuid in enumerate(masked_test_uuid_list):
            if uuid in test_uuid_to_chain_attack_only:
                test_eval_idx_to_chain[new_idx] = test_uuid_to_chain_attack_only[uuid]

        if test_eval_idx_to_chain:
            test_adp = compute_adp_score(
                test_scores_eval,
                test_labels_eval,
                test_eval_idx_to_chain,
                out_file=os.path.join(_EVAL_RESULT_DIR, f"{result_stem}_adp_strict.png"),
            )

    plot_anomaly_score_distribution(
        test_scores_eval,
        test_labels_eval,
        os.path.join(_EVAL_RESULT_DIR, f"{result_stem}_anomaly_score_distribution_strict.pdf"),
        threshold=threshold,
        nodes=list(range(len(test_scores_eval))),
        node_to_attack_chain=test_eval_idx_to_chain or None,
    )

    results = {
        "threshold": threshold,
        "val_auc": val_auc,
        "test_auc": test_auc,
        "test_ap": test_ap,
        "test_f1": test_f1,
        "test_precision": test_prec,
        "test_recall": test_rec,
        "test_fpr": test_fpr,
        "test_mcc": test_mcc,
        "test_tp": int(test_tp),
        "test_fp": int(test_fp),
        "test_tn": int(test_tn),
        "test_fn": int(test_fn),
        "test_adp": test_adp,
    }

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Metric':<15} {'Strict Attack Chain':<20}")
    print(f"{'F1':<15} {test_f1:<20.4f}")
    print(f"{'AP':<15} {test_ap:<20.4f}")
    print(f"{'AUROC':<15} {test_auc:<20.4f}")
    print(f"{'Precision':<15} {test_prec:<20.4f}")
    print(f"{'FPR':<15} {test_fpr:<20.4f}")
    print(f"{'MCC':<15} {test_mcc:<20.4f}")
    print(f"{'ADP':<15} {test_adp:<20.4f}")

    os.makedirs(_EVAL_RESULT_DIR, exist_ok=True)
    with open(
        os.path.join(_EVAL_RESULT_DIR, f"{result_stem}_strict_results.pkl"), "wb"
    ) as f:
        pkl.dump(results, f)

    return results
