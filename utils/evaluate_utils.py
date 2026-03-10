import numpy as np
from matplotlib import pyplot as plt

from utils.ground_truth import (
    get_excluded_node_ids,
    get_ground_truth,
    get_uuid_to_index_id_mapping,
)
from utils.plotting import (
    log_image_to_wandb,
    plot_anomaly_score_distribution,
    plot_scores_neat,
)
from utils.utils import log

__all__ = [
    "compute_adp_score",
    "get_excluded_node_ids",
    "get_ground_truth",
    "get_uuid_to_index_id_mapping",
    "log_image_to_wandb",
    "plot_anomaly_score_distribution",
    "plot_scores_neat",
]


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
