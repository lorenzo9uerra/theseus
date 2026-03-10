import os

import numpy as np
from sklearn.metrics import average_precision_score

from tasks.evaluate_support import (
    aggregate_to_entity_level,
    build_node_to_attack_mappings,
    compute_binary_metrics,
    find_threshold,
    inference_loop,
)
from utils.evaluate_utils import compute_adp_score
from utils.ground_truth import get_excluded_node_ids
from utils.plotting import plot_anomaly_score_distribution, plot_scores_neat
from utils.utils import log, timed_execution


@timed_execution
def evaluate(model, val_data, test_data, config, ground_truth):
    """
    Evaluate at ENTITY level (one score per unique node via max-pooling).

    Two evaluation scopes:
    - Causal: attack + contaminated nodes as positives (logged as final_test_* + test_adp_causal)
    - Strict Attack Chain: only attack nodes are positive (contaminated excluded; logged as final_strict_test_* + test_adp_strict)
    """
    excluded_node_ids = get_excluded_node_ids(config)

    # Validation: aggregate to entity level, then find threshold
    val_scores_raw, val_labels_raw, val_nodes_raw, _ = inference_loop(
        model, val_data, config
    )
    log(
        f"Validation: {len(val_scores_raw)} instances -> {len(set(val_nodes_raw))} entities"
    )

    if excluded_node_ids and val_nodes_raw.size > 0:
        keep = ~np.isin(val_nodes_raw, list(excluded_node_ids))
        dropped_instances = int((~keep).sum())
        dropped_entities = len(set(val_nodes_raw[~keep].tolist()))
        if dropped_instances > 0:
            log(
                f"Validation: excluding {dropped_entities} entities ({dropped_instances} instances) from metrics"
            )
        val_scores_raw = val_scores_raw[keep]
        val_labels_raw = val_labels_raw[keep]
        val_nodes_raw = val_nodes_raw[keep]

    val_scores, val_labels, _ = aggregate_to_entity_level(
        val_scores_raw, val_labels_raw, val_nodes_raw
    )

    val_ap = (
        average_precision_score(val_labels, val_scores) if val_labels.size > 0 else 0.0
    )
    threshold, val_mcc = find_threshold(val_scores, val_labels)

    # Test: aggregate to entity level
    test_scores_raw, test_labels_raw, test_nodes_raw, _ = inference_loop(
        model, test_data, config
    )
    log(
        f"Test: {len(test_scores_raw)} instances -> {len(set(test_nodes_raw))} entities"
    )

    if excluded_node_ids and test_nodes_raw.size > 0:
        keep = ~np.isin(test_nodes_raw, list(excluded_node_ids))
        dropped_instances = int((~keep).sum())
        dropped_entities = len(set(test_nodes_raw[~keep].tolist()))
        if dropped_instances > 0:
            log(
                f"Test: excluding {dropped_entities} entities ({dropped_instances} instances) from metrics"
            )
        test_scores_raw = test_scores_raw[keep]
        test_labels_raw = test_labels_raw[keep]
        test_nodes_raw = test_nodes_raw[keep]

    test_scores, test_labels, test_nodes = aggregate_to_entity_level(
        test_scores_raw, test_labels_raw, test_nodes_raw
    )

    test_ap = (
        average_precision_score(test_labels, test_scores)
        if test_labels.size > 0
        else 0.0
    )
    test_predictions = (
        np.zeros_like(test_labels)
        if np.isinf(threshold)
        else (test_scores >= threshold).astype(int)
    )

    metrics = {
        "final_val_ap": val_ap,
        "final_val_mcc": val_mcc,
        "threshold": threshold,
        "final_test_ap": test_ap,
    }
    metrics.update(
        compute_binary_metrics(
            test_labels, test_predictions, "final_test", "confusion_matrix"
        )
    )

    # Calculate strict metrics (excluding contaminated nodes)
    all_contaminated_nids = set()
    if ground_truth:
        for metadata in ground_truth.values():
            all_contaminated_nids.update(metadata.get("contaminated_nids", []) or [])

    if all_contaminated_nids and test_nodes.size > 0:
        is_not_contaminated = ~np.isin(test_nodes, list(all_contaminated_nids))

        if is_not_contaminated.any():
            strict_scores = test_scores[is_not_contaminated]
            strict_labels = test_labels[is_not_contaminated]

            if strict_labels.size > 0:
                strict_ap = average_precision_score(strict_labels, strict_scores)

                strict_predictions = (
                    np.zeros_like(strict_labels)
                    if np.isinf(threshold)
                    else (strict_scores >= threshold).astype(int)
                )

                metrics.update(
                    {
                        "final_strict_test_ap": strict_ap,
                    }
                )
                metrics.update(
                    compute_binary_metrics(
                        strict_labels,
                        strict_predictions,
                        "final_strict_test",
                        "strict_confusion_matrix",
                    )
                )

    if not all_contaminated_nids:
        metrics.setdefault("final_strict_test_ap", metrics.get("final_test_ap"))
        metrics.setdefault(
            "final_strict_test_binary_f1", metrics.get("final_test_binary_f1")
        )
        metrics.setdefault("final_strict_test_mcc", metrics.get("final_test_mcc"))
        metrics.setdefault("final_strict_test_fpr", metrics.get("final_test_fpr"))

    out_dir = config.outputs_dir
    os.makedirs(out_dir, exist_ok=True)

    node_to_attack_ids_strict, node_to_attack_ids_causal = (
        build_node_to_attack_mappings(ground_truth)
    )

    if node_to_attack_ids_causal and test_scores.size > 0:
        threshold_for_plot = None if np.isinf(threshold) else float(threshold)

        dist_plot_path = os.path.join(out_dir, "anomaly_score_distribution.pdf")
        try:
            plot_anomaly_score_distribution(
                test_scores.tolist(),
                test_labels.tolist(),
                dist_plot_path,
                threshold_for_plot,
                nodes=test_nodes.tolist(),
                node_to_attack_ids=node_to_attack_ids_causal,
            )
        except Exception as exc:
            log(f"Anomaly score distribution plot failed: {exc}")

        neat_scores_path = os.path.join(out_dir, "scores_neat.png")
        try:
            plot_scores_neat(
                test_scores.tolist(),
                test_labels.tolist(),
                test_nodes.tolist(),
                node_to_attack_ids_causal,
                neat_scores_path,
                threshold_for_plot,
            )
        except Exception as exc:
            log(f"Neat score plot failed: {exc}")

        # ADP curves for both scopes
        adp_strict_path = os.path.join(out_dir, "adp_strict_curve.png")
        try:
            adp_strict_score = compute_adp_score(
                test_scores.tolist(),
                test_nodes.tolist(),
                node_to_attack_ids_strict,
                test_labels.tolist(),
                out_file=adp_strict_path,
            )
            metrics["test_adp_strict"] = adp_strict_score
        except Exception as exc:
            log(f"Strict ADP computation failed: {exc}")

        adp_causal_path = os.path.join(out_dir, "adp_causal_curve.png")
        try:
            adp_causal_score = compute_adp_score(
                test_scores.tolist(),
                test_nodes.tolist(),
                node_to_attack_ids_causal,
                test_labels.tolist(),
                out_file=adp_causal_path,
            )
            metrics["test_adp_causal"] = adp_causal_score
        except Exception as exc:
            log(f"Causal ADP computation failed: {exc}")

    if not all_contaminated_nids:
        metrics.setdefault("test_adp_strict", metrics.get("test_adp_causal"))

    for key, value in metrics.items():
        log(f"{key}: {value}")

    return metrics


def validate(model, val_data, config):
    """Validate with entity-level aggregation for consistency with evaluate()."""
    scores_raw, labels_raw, node_ids_raw, avg_loss = inference_loop(
        model, val_data, config
    )
    scores, labels, _ = aggregate_to_entity_level(scores_raw, labels_raw, node_ids_raw)
    val_ap = average_precision_score(labels, scores)
    threshold, _ = find_threshold(scores, labels)
    return val_ap, avg_loss, threshold
