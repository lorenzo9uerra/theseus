import os

import numpy as np
from sklearn.metrics import average_precision_score

from tasks.evaluate_support import (
    aggregate_to_entity_level,
    build_node_to_attack_mappings,
    compute_auroc,
    compute_binary_metrics,
    compute_threshold_metrics,
    inference_loop,
    select_max_benign_threshold,
    threshold_predictions,
)
from utils.evaluate_utils import compute_adp_score
from utils.ground_truth import get_excluded_node_ids
from utils.plotting import plot_anomaly_score_distribution, plot_scores_neat
from utils.utils import log, timed_execution


def _get_plot_stem(config) -> str:
    model_name = getattr(config, "model", "theseus")
    dataset_name = str(getattr(config, "dataset", "dataset")).lower()
    seed = getattr(config, "seed", "unknown")
    return f"{model_name}_{dataset_name}_seed{seed}"


def _get_metric_excluded_node_ids(config):
    try:
        return get_excluded_node_ids(config)
    except AttributeError:
        return set()


def _get_contaminated_node_ids(ground_truth):
    contaminated_node_ids = set()
    if ground_truth:
        for metadata in ground_truth.values():
            contaminated_node_ids.update(metadata.get("contaminated_nids", []) or [])
    return contaminated_node_ids


def _exclude_contaminated_entities(
    scores, labels, nodes, contaminated_node_ids, *, split_name
):
    if not contaminated_node_ids or nodes.size == 0:
        return scores, labels, nodes

    keep = ~np.isin(nodes, list(contaminated_node_ids))
    dropped = int((~keep).sum())
    if dropped > 0:
        log(
            f"{split_name}: excluding {dropped} contaminated entities from thresholding and metrics"
        )

    return scores[keep], labels[keep], nodes[keep]


def _collect_entity_scores(model, dataset, config, *, split_name, excluded_node_ids):
    scores_raw, labels_raw, nodes_raw, avg_loss = inference_loop(model, dataset, config)
    log(f"{split_name}: {len(scores_raw)} instances -> {len(set(nodes_raw))} entities")

    if excluded_node_ids and nodes_raw.size > 0:
        keep = ~np.isin(nodes_raw, list(excluded_node_ids))
        dropped_instances = int((~keep).sum())
        dropped_entities = len(set(nodes_raw[~keep].tolist()))
        if dropped_instances > 0:
            log(
                f"{split_name}: excluding {dropped_entities} entities ({dropped_instances} instances) from metrics"
            )
        scores_raw = scores_raw[keep]
        labels_raw = labels_raw[keep]
        nodes_raw = nodes_raw[keep]

    scores, labels, nodes = aggregate_to_entity_level(scores_raw, labels_raw, nodes_raw)
    return scores, labels, nodes, avg_loss


def _calibrate_threshold(
    model, dataset, config, *, split_name, excluded_node_ids, contaminated_node_ids
):
    scores, labels, nodes, avg_loss = _collect_entity_scores(
        model,
        dataset,
        config,
        split_name=split_name,
        excluded_node_ids=excluded_node_ids,
    )
    scores, labels, _ = _exclude_contaminated_entities(
        scores, labels, nodes, contaminated_node_ids, split_name=split_name
    )

    ap = average_precision_score(labels, scores) if np.any(labels == 1) else 0.0
    threshold = select_max_benign_threshold(scores, labels)
    threshold_source = f"{split_name.lower()}_max_benign"
    mcc = compute_threshold_metrics(scores, labels, threshold)["mcc"]

    benign_count = int(np.sum(labels == 0))
    if benign_count > 0:
        log(
            f"{split_name}: threshold set to maximum benign calibration score "
            f"across {benign_count} entities."
        )
    elif labels.size > 0:
        log(
            f"{split_name}: no benign entities available for calibration; "
            "falling back to the maximum finite calibration score."
        )

    return ap, avg_loss, threshold, mcc, threshold_source


@timed_execution
def evaluate(
    model,
    val_data,
    test_data,
    config,
    ground_truth,
    *,
    calibration_data=None,
    calibration_split_name="Validation",
):
    """Evaluate at entity level under strict attack-chain semantics only."""
    excluded_node_ids = _get_metric_excluded_node_ids(config)
    contaminated_node_ids = _get_contaminated_node_ids(ground_truth)

    if calibration_data is None:
        calibration_data = val_data

    calibration_ap, _, threshold, calibration_mcc, threshold_source = (
        _calibrate_threshold(
            model,
            calibration_data,
            config,
            split_name=calibration_split_name,
            excluded_node_ids=excluded_node_ids,
            contaminated_node_ids=contaminated_node_ids,
        )
    )

    # Test: aggregate to entity level
    test_scores, test_labels, test_nodes, _ = _collect_entity_scores(
        model, test_data, config, split_name="Test", excluded_node_ids=excluded_node_ids
    )
    test_scores, test_labels, test_nodes = _exclude_contaminated_entities(
        test_scores, test_labels, test_nodes, contaminated_node_ids, split_name="Test"
    )

    test_ap = (
        average_precision_score(test_labels, test_scores)
        if test_labels.size > 0
        else 0.0
    )
    test_auroc = compute_auroc(test_labels, test_scores)
    test_predictions = threshold_predictions(test_scores, threshold)

    metrics = {
        "final_val_ap": calibration_ap,
        "final_val_mcc": calibration_mcc,
        "threshold": threshold,
        "threshold_source": threshold_source,
        "final_test_ap": test_ap,
        "final_test_auroc": test_auroc,
    }
    metrics.update(
        compute_binary_metrics(
            test_labels, test_predictions, "final_test", "confusion_matrix"
        )
    )

    out_dir = config.outputs_dir
    os.makedirs(out_dir, exist_ok=True)

    node_to_attack_ids = build_node_to_attack_mappings(ground_truth)

    if node_to_attack_ids and test_scores.size > 0:
        threshold_for_plot = None if np.isinf(threshold) else float(threshold)
        plot_stem = _get_plot_stem(config)

        dist_plot_path = os.path.join(
            out_dir, f"{plot_stem}_anomaly_score_distribution.pdf"
        )
        try:
            plot_anomaly_score_distribution(
                test_scores.tolist(),
                test_labels.tolist(),
                dist_plot_path,
                threshold_for_plot,
                nodes=test_nodes.tolist(),
                node_to_attack_ids=node_to_attack_ids,
            )
        except Exception as exc:
            log(f"Anomaly score distribution plot failed: {exc}")

        neat_scores_path = os.path.join(out_dir, f"{plot_stem}_scores_neat.png")
        try:
            plot_scores_neat(
                test_scores.tolist(),
                test_labels.tolist(),
                test_nodes.tolist(),
                node_to_attack_ids,
                neat_scores_path,
                threshold_for_plot,
            )
        except Exception as exc:
            log(f"Neat score plot failed: {exc}")

        adp_path = os.path.join(out_dir, f"{plot_stem}_adp_curve.png")
        try:
            adp_score = compute_adp_score(
                test_scores.tolist(),
                test_nodes.tolist(),
                node_to_attack_ids,
                test_labels.tolist(),
                out_file=adp_path,
            )
            metrics["test_adp"] = adp_score
        except Exception as exc:
            log(f"ADP computation failed: {exc}")

    for key, value in metrics.items():
        log(f"{key}: {value}")

    return metrics


def validate(model, val_data, config, ground_truth=None):
    """Validate with entity-level aggregation for consistency with evaluate()."""
    scores, labels, nodes, avg_loss = _collect_entity_scores(
        model,
        val_data,
        config,
        split_name="Validation",
        excluded_node_ids=_get_metric_excluded_node_ids(config),
    )
    scores, labels, _ = _exclude_contaminated_entities(
        scores,
        labels,
        nodes,
        _get_contaminated_node_ids(ground_truth),
        split_name="Validation",
    )
    val_ap = average_precision_score(labels, scores) if np.any(labels == 1) else 0.0
    threshold = select_max_benign_threshold(scores, labels)
    return val_ap, avg_loss, threshold
