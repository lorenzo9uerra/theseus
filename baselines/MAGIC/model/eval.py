import os
import pickle as pkl
import random

import numpy as np
from sklearn.metrics import auc as compute_auc
from sklearn.metrics import precision_recall_curve, roc_auc_score
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


def aggregate_to_entity_level(scores, labels, uuids, node_to_attack_chain=None):
    """
    Max-pooling aggregation to entity level for fair comparison with baselines.
    Each unique UUID gets its maximum score; label=1 if positive in any instance.

    Args:
        scores: Instance-level anomaly scores
        labels: Instance-level labels (0/1)
        uuids: UUID for each instance
        node_to_attack_chain: Optional instance-level idx->attack_chain mapping

    Returns:
        agg_scores, agg_labels, agg_uuids, uuid_to_attack_chain (entity-level)
    """
    entity_map = {}  # uuid -> {'score': max, 'label': max, 'attack_chain': str}

    for i, (score, label, uuid) in enumerate(zip(scores, labels, uuids)):
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
    for orig_idx, label in zip(sorted_indices, sorted_labels):
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
    sorted_pairs = sorted(zip(precisions, detected_percentages))
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
        [score for score, label in zip(scores, y_truth) if label == 0]
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

        for score, label, node in zip(scores, y_truth, nodes):
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
            [score for score, label in zip(scores, y_truth) if label == 1]
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
    """
    Evaluate entity-level detection with validation-based threshold selection (NO DATA LEAKAGE).

    Can use either:
    1. Dedicated validation set (x_val + y_val provided) - PREFERRED
    2. Split test data into validation and final test (x_val=None, val_ratio used)

    Args:
        dataset: Dataset name
        x_train: Training embeddings (benign only)
        x_test: Test embeddings (benign + attack)
        y_test: Test labels (0=benign, 1=attack)
        val_ratio: Fraction of test data for validation (used only if x_val=None)
        random_state: Random seed for reproducibility
        x_val: Optional dedicated validation embeddings
        y_val: Optional validation labels (required if x_val provided)

    Returns:
        auc, std, threshold, test_scores
    """
    print("\n" + "=" * 80)
    if x_val is not None:
        print("DEDICATED VALIDATION SET (Day-based split, no data leakage)")
    else:
        print("VALIDATION-BASED THRESHOLD SELECTION (Split from test)")
    print("=" * 80)

    if x_val is not None:
        # Use dedicated validation set with labels
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
    MAX_KNN_TRAIN_SAMPLES = 100000
    if len(x_train_norm) > MAX_KNN_TRAIN_SAMPLES:
        print(
            f"\nSubsampling training data for KNN: {len(x_train_norm):,} -> {MAX_KNN_TRAIN_SAMPLES:,} samples"
        )
        idx_subsample = np.random.choice(
            len(x_train_norm), MAX_KNN_TRAIN_SAMPLES, replace=False
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

    # Validation: select threshold
    print("\nSelecting threshold on validation set...")
    val_scores = compute_scores_batched(x_val_norm, "validation")

    # Use F1-based threshold selection (works for both dedicated val and split test)
    prec_val, rec_val, thresholds = precision_recall_curve(y_val, val_scores)
    f1_val = 2 * prec_val * rec_val / (prec_val + rec_val + 1e-9)
    best_idx = np.argmax(f1_val)
    best_threshold = thresholds[best_idx]

    val_auc = roc_auc_score(y_val, val_scores)
    val_prauc = compute_auc(rec_val, prec_val)  # PR-AUC
    y_val_pred = (val_scores >= best_threshold).astype(int)
    val_tp = np.sum((y_val == 1) & (y_val_pred == 1))
    val_fp = np.sum((y_val == 0) & (y_val_pred == 1))
    val_tn = np.sum((y_val == 0) & (y_val_pred == 0))
    val_fn = np.sum((y_val == 1) & (y_val_pred == 0))
    val_prec = val_tp / (val_tp + val_fp) if (val_tp + val_fp) > 0 else 0
    val_rec = val_tp / (val_tp + val_fn) if (val_tp + val_fn) > 0 else 0
    val_f1 = f1_val[best_idx]

    val_source = "dedicated validation set" if x_val is not None else "split from test"
    print(f"\nValidation Results (from {val_source}):")
    print(f"  ROC-AUC: {val_auc:.6f}")
    print(f"  PR-AUC: {val_prauc:.6f}")
    print(f"  F1: {val_f1:.6f} (Prec: {val_prec:.6f}, Rec: {val_rec:.6f})")
    print(f"  Threshold: {best_threshold:.6f}")
    print(f"  TP: {val_tp}, FP: {val_fp}, TN: {val_tn}, FN: {val_fn}")

    # Test: apply threshold
    print("\nEvaluating on held-out test set...")
    test_scores = compute_scores_batched(x_test_norm, "test")

    y_test_pred = (test_scores >= best_threshold).astype(int)
    test_auc = roc_auc_score(y_final_test, test_scores)

    prec_test, rec_test, _ = precision_recall_curve(y_final_test, test_scores)
    test_prauc = compute_auc(rec_test, prec_test)

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
    print("FINAL TEST RESULTS (No Data Leakage)")
    print("=" * 80)
    print(f"Threshold: {best_threshold:.6f} (from validation)")
    print(f"ROC-AUC: {test_auc:.6f}")
    print(f"PR-AUC: {test_prauc:.6f}")
    print(f"F1: {test_f1:.6f}")
    print(f"PRECISION: {test_prec:.6f}")
    print(f"RECALL: {test_rec:.6f}")
    print(f"TN: {test_tn}, FN: {test_fn}, TP: {test_tp}, FP: {test_fp}")
    print("=" * 80)

    # Save results
    os.makedirs(_EVAL_RESULT_DIR, exist_ok=True)
    save_dict = {
        "dataset": dataset,
        "val_ratio": val_ratio,
        "threshold": best_threshold,
        "validation": {
            "auc": val_auc,
            "prauc": val_prauc,
            "f1": val_f1,
            "precision": val_prec,
            "recall": val_rec,
        },
        "test": {
            "auc": test_auc,
            "prauc": test_prauc,
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


def two_level_evaluation(
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
):
    """
    Entity-level evaluation with two scopes:
    1. Causal Scope: Attack + Contaminated nodes as positives
    2. Strict Attack Chain: Attack nodes only (Contaminated masked)

    Entity-level aggregation (max-pooling) ensures fair comparison with baselines.
    """
    print("\n" + "=" * 80)
    print("ENTITY-LEVEL TWO-SCOPE EVALUATION")
    print("=" * 80)

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

    # Create instance-level labels
    val_labels_inst = np.zeros(n_val_nodes)
    for idx in val_attack_idx + val_contaminated_idx:
        if idx < n_val_nodes:
            val_labels_inst[idx] = 1

    test_labels_inst = np.zeros(n_test_nodes)
    for idx in test_attack_idx + test_contaminated_idx:
        if idx < n_test_nodes:
            test_labels_inst[idx] = 1

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
                val_node_to_attack_chain_all,
            )
        )
        test_scores, test_labels, test_uuid_list, test_uuid_to_chain = (
            aggregate_to_entity_level(
                test_scores_inst,
                test_labels_inst,
                test_uuids,
                test_node_to_attack_chain_all,
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
        test_uuid_to_chain = test_node_to_attack_chain_all or {}

    results = {}

    # Level 1: Causal Scope
    print("\n" + "-" * 80)
    print("LEVEL 1: CAUSAL SCOPE (Attack + Contaminated as positives)")
    print("-" * 80)

    if val_labels.sum() > 0:
        threshold_l1, _ = find_best_threshold_mcc(val_scores, val_labels)
        val_auc_l1 = roc_auc_score(val_labels, val_scores)
    else:
        threshold_l1 = np.percentile(val_scores, 95)
        val_auc_l1 = 0.0

    y_test_pred_l1 = (test_scores >= threshold_l1).astype(int)
    test_auc_l1 = (
        roc_auc_score(test_labels, test_scores) if test_labels.sum() > 0 else 0.0
    )
    prec_test, rec_test, _ = precision_recall_curve(test_labels, test_scores)
    test_prauc_l1 = compute_auc(rec_test, prec_test) if test_labels.sum() > 0 else 0.0

    test_tp_l1 = np.sum((test_labels == 1) & (y_test_pred_l1 == 1))
    test_fp_l1 = np.sum((test_labels == 0) & (y_test_pred_l1 == 1))
    test_tn_l1 = np.sum((test_labels == 0) & (y_test_pred_l1 == 0))
    test_fn_l1 = np.sum((test_labels == 1) & (y_test_pred_l1 == 0))
    test_prec_l1 = (
        test_tp_l1 / (test_tp_l1 + test_fp_l1) if (test_tp_l1 + test_fp_l1) > 0 else 0
    )
    test_rec_l1 = (
        test_tp_l1 / (test_tp_l1 + test_fn_l1) if (test_tp_l1 + test_fn_l1) > 0 else 0
    )
    test_f1_l1 = (
        2 * test_prec_l1 * test_rec_l1 / (test_prec_l1 + test_rec_l1)
        if (test_prec_l1 + test_rec_l1) > 0
        else 0
    )
    test_fpr_l1 = (
        test_fp_l1 / (test_fp_l1 + test_tn_l1) if (test_fp_l1 + test_tn_l1) > 0 else 0
    )
    mcc_denom = np.sqrt(
        (test_tp_l1 + test_fp_l1)
        * (test_tp_l1 + test_fn_l1)
        * (test_tn_l1 + test_fp_l1)
        * (test_tn_l1 + test_fn_l1)
    )
    test_mcc_l1 = (
        (test_tp_l1 * test_tn_l1 - test_fp_l1 * test_fn_l1) / mcc_denom
        if mcc_denom > 0
        else 0
    )

    print(f"  F1: {test_f1_l1:.4f}, Prec: {test_prec_l1:.4f}, Rec: {test_rec_l1:.4f}")
    print(f"  TP: {test_tp_l1}, FP: {test_fp_l1}, TN: {test_tn_l1}, FN: {test_fn_l1}")

    # ADP for Level 1
    test_adp_l1 = 0.0
    if test_uuid_to_chain:
        os.makedirs(_EVAL_RESULT_DIR, exist_ok=True)
        test_adp_l1 = compute_adp_score(
            test_scores,
            test_labels,
            test_uuid_to_chain,
            out_file=os.path.join(_EVAL_RESULT_DIR, f"{dataset}_adp_level1.png"),
        )

    results["level1_causal"] = {
        "threshold": threshold_l1,
        "val_auc": val_auc_l1,
        "test_auc": test_auc_l1,
        "test_prauc": test_prauc_l1,
        "test_f1": test_f1_l1,
        "test_precision": test_prec_l1,
        "test_recall": test_rec_l1,
        "test_fpr": test_fpr_l1,
        "test_mcc": test_mcc_l1,
        "test_tp": int(test_tp_l1),
        "test_fp": int(test_fp_l1),
        "test_tn": int(test_tn_l1),
        "test_fn": int(test_fn_l1),
        "test_adp": test_adp_l1,
    }

    # Level 2: Strict Attack Chain (mask contaminated)
    print("\n" + "-" * 80)
    print("LEVEL 2: STRICT ATTACK CHAIN (Contaminated masked)")
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

        val_scores_l2 = val_scores[val_mask]
        val_labels_l2 = np.array(
            [
                1 if val_uuid_list[i] in val_attack_uuids else 0
                for i in range(len(val_uuid_list))
            ]
        )[val_mask]

        test_scores_l2 = test_scores[test_mask]
        test_labels_l2 = np.array(
            [
                1 if test_uuid_list[i] in test_attack_uuids else 0
                for i in range(len(test_uuid_list))
            ]
        )[test_mask]
    else:
        # Fallback to instance-level masking
        val_mask = np.ones(len(val_scores), dtype=bool)
        test_mask = np.ones(len(test_scores), dtype=bool)
        for idx in val_contaminated_idx:
            if idx < len(val_mask):
                val_mask[idx] = False
        for idx in test_contaminated_idx:
            if idx < len(test_mask):
                test_mask[idx] = False
        val_scores_l2 = val_scores[val_mask]
        val_labels_l2 = np.zeros(len(val_scores))
        for idx in val_attack_idx:
            if idx < len(val_labels_l2):
                val_labels_l2[idx] = 1
        val_labels_l2 = val_labels_l2[val_mask]
        test_scores_l2 = test_scores[test_mask]
        test_labels_l2 = np.zeros(len(test_scores))
        for idx in test_attack_idx:
            if idx < len(test_labels_l2):
                test_labels_l2[idx] = 1
        test_labels_l2 = test_labels_l2[test_mask]

    print(f"  After masking: val={len(val_scores_l2)}, test={len(test_scores_l2)}")

    # Use same threshold from Level 1 for consistency with Theseus/PIDSMaker
    threshold_l2 = threshold_l1
    val_auc_l2 = (
        roc_auc_score(val_labels_l2, val_scores_l2) if val_labels_l2.sum() > 0 else 0.0
    )

    y_test_pred_l2 = (test_scores_l2 >= threshold_l2).astype(int)
    test_auc_l2 = (
        roc_auc_score(test_labels_l2, test_scores_l2)
        if test_labels_l2.sum() > 0
        else 0.0
    )
    prec_test, rec_test, _ = precision_recall_curve(test_labels_l2, test_scores_l2)
    test_prauc_l2 = (
        compute_auc(rec_test, prec_test) if test_labels_l2.sum() > 0 else 0.0
    )

    test_tp_l2 = np.sum((test_labels_l2 == 1) & (y_test_pred_l2 == 1))
    test_fp_l2 = np.sum((test_labels_l2 == 0) & (y_test_pred_l2 == 1))
    test_tn_l2 = np.sum((test_labels_l2 == 0) & (y_test_pred_l2 == 0))
    test_fn_l2 = np.sum((test_labels_l2 == 1) & (y_test_pred_l2 == 0))
    test_prec_l2 = (
        test_tp_l2 / (test_tp_l2 + test_fp_l2) if (test_tp_l2 + test_fp_l2) > 0 else 0
    )
    test_rec_l2 = (
        test_tp_l2 / (test_tp_l2 + test_fn_l2) if (test_tp_l2 + test_fn_l2) > 0 else 0
    )
    test_f1_l2 = (
        2 * test_prec_l2 * test_rec_l2 / (test_prec_l2 + test_rec_l2)
        if (test_prec_l2 + test_rec_l2) > 0
        else 0
    )
    test_fpr_l2 = (
        test_fp_l2 / (test_fp_l2 + test_tn_l2) if (test_fp_l2 + test_tn_l2) > 0 else 0
    )
    mcc_denom = np.sqrt(
        (test_tp_l2 + test_fp_l2)
        * (test_tp_l2 + test_fn_l2)
        * (test_tn_l2 + test_fp_l2)
        * (test_tn_l2 + test_fn_l2)
    )
    test_mcc_l2 = (
        (test_tp_l2 * test_tn_l2 - test_fp_l2 * test_fn_l2) / mcc_denom
        if mcc_denom > 0
        else 0
    )

    print(f"  F1: {test_f1_l2:.4f}, Prec: {test_prec_l2:.4f}, Rec: {test_rec_l2:.4f}")
    print(f"  TP: {test_tp_l2}, FP: {test_fp_l2}, TN: {test_tn_l2}, FN: {test_fn_l2}")

    # ADP for Level 2 (attack nodes only, with remapped indices after masking)
    test_adp_l2 = 0.0
    if val_idx_to_uuid and test_idx_to_uuid and test_node_to_attack_chain_attack_only:
        # Build UUID->attack_chain mapping for attack nodes only
        test_uuid_to_chain_attack_only = {}
        for idx, chain in test_node_to_attack_chain_attack_only.items():
            if idx in test_idx_to_uuid:
                uuid = test_idx_to_uuid[idx]
                test_uuid_to_chain_attack_only[uuid] = chain

        # Create index mapping for the masked Level 2 data
        # test_mask tells us which entities from test_uuid_list are kept
        masked_uuid_list = [
            test_uuid_list[i] for i in range(len(test_uuid_list)) if test_mask[i]
        ]
        test_l2_idx_to_chain = {}
        for new_idx, uuid in enumerate(masked_uuid_list):
            if uuid in test_uuid_to_chain_attack_only:
                test_l2_idx_to_chain[new_idx] = test_uuid_to_chain_attack_only[uuid]

        if test_l2_idx_to_chain:
            test_adp_l2 = compute_adp_score(
                test_scores_l2,
                test_labels_l2,
                test_l2_idx_to_chain,
                out_file=os.path.join(_EVAL_RESULT_DIR, f"{dataset}_adp_level2.png"),
            )

    results["level2_strict"] = {
        "threshold": threshold_l2,
        "val_auc": val_auc_l2,
        "test_auc": test_auc_l2,
        "test_prauc": test_prauc_l2,
        "test_f1": test_f1_l2,
        "test_precision": test_prec_l2,
        "test_recall": test_rec_l2,
        "test_fpr": test_fpr_l2,
        "test_mcc": test_mcc_l2,
        "test_tp": int(test_tp_l2),
        "test_fp": int(test_fp_l2),
        "test_tn": int(test_tn_l2),
        "test_fn": int(test_fn_l2),
        "test_adp": test_adp_l2,
    }

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Metric':<15} {'Causal Scope':<15} {'Strict Attack Chain':<15}")
    print(f"{'F1':<15} {test_f1_l1:<15.4f} {test_f1_l2:<15.4f}")
    print(f"{'PR-AUC':<15} {test_prauc_l1:<15.4f} {test_prauc_l2:<15.4f}")
    print(f"{'FPR':<15} {test_fpr_l1:<15.4f} {test_fpr_l2:<15.4f}")
    print(f"{'MCC':<15} {test_mcc_l1:<15.4f} {test_mcc_l2:<15.4f}")
    print(f"{'ADP':<15} {test_adp_l1:<15.4f} {test_adp_l2:<15.4f}")

    os.makedirs(_EVAL_RESULT_DIR, exist_ok=True)
    with open(
        os.path.join(_EVAL_RESULT_DIR, f"{dataset}_two_level_results.pkl"), "wb"
    ) as f:
        pkl.dump(results, f)

    return results
