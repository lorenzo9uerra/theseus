import os
from collections import defaultdict

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
)

from utils.evaluate_utils import (
    compute_adp_score,
    get_excluded_node_ids,
    plot_anomaly_score_distribution,
    plot_scores_neat,
)
from utils.utils import log, timed_execution


def aggregate_to_entity_level(scores, labels, node_ids):
    """
    Max-pooling aggregation to entity level for fair comparison with baselines.
    Each node gets its maximum score; label=1 if positive in any snapshot.
    """
    node_map = {}

    for score, label, nid in zip(scores, labels, node_ids, strict=True):
        nid_key = int(nid)
        if nid_key not in node_map:
            node_map[nid_key] = {"score": -float("inf"), "label": 0}

        if score > node_map[nid_key]["score"]:
            node_map[nid_key]["score"] = score

        if label > node_map[nid_key]["label"]:
            node_map[nid_key]["label"] = label

    agg_scores = []
    agg_labels = []
    agg_nodes = []

    for nid, data in node_map.items():
        agg_scores.append(data["score"])
        agg_labels.append(data["label"])
        agg_nodes.append(nid)

    return (
        np.array(agg_scores),
        np.array(agg_labels, dtype=int),
        np.array(agg_nodes, dtype=int),
    )


@torch.inference_mode()
def inference_loop(model, dataset, config):
    """Compute per-node anomaly scores for process nodes only."""
    device = config.device

    all_scores = []
    all_labels = []
    all_node_ids = []
    total_loss = 0.0
    total_samples = 0

    for graph in dataset:
        graph = graph.to(device=device)
        outputs, encoded_target = model(graph)
        losses = model.loss(outputs, encoded_target, "none")

        if losses.dim() > 1:
            losses = losses.mean(dim=tuple(range(1, losses.dim())))

        labels_tensor = graph.y
        if labels_tensor.dim() > 1:
            labels_tensor = labels_tensor.view(-1)

        process_mask = graph.x[:, 0] == 1

        if process_mask.any():
            losses = losses[process_mask]
            labels_tensor = labels_tensor[process_mask]
            node_ids_tensor = graph.original_n_id[process_mask]

            all_scores.extend(losses.detach().cpu().tolist())
            all_labels.extend(labels_tensor.detach().cpu().tolist())
            all_node_ids.extend(node_ids_tensor.detach().cpu().tolist())
            total_loss += losses.sum().item()
            total_samples += losses.shape[0]

    avg_loss = total_loss / total_samples if total_samples else 0.0

    return (
        np.asarray(all_scores, dtype=float),
        np.asarray(all_labels, dtype=int),
        np.asarray(all_node_ids, dtype=int),
        avg_loss,
    )


def _find_threshold(scores, labels):
    """Find optimal threshold by maximizing MCC on the validation set."""
    if labels.size == 0 or scores.size == 0:
        return np.inf, 0.0

    if not np.any(labels == 1):
        return np.inf, 0.0

    _, _, thresholds = precision_recall_curve(labels, scores)
    if thresholds.size == 0:
        best_threshold = scores.max() + 1e-6 if scores.size else 0.0
        return best_threshold, 0.0

    best_mcc = -1.0
    best_idx = 0

    for idx, thresh in enumerate(thresholds):
        predictions = (scores >= thresh).astype(int)
        mcc = matthews_corrcoef(labels, predictions)

        if mcc > best_mcc:
            best_mcc = mcc
            best_idx = idx

    return thresholds[best_idx], best_mcc


def _build_node_to_attack_mappings(attack_metadata):
    """Build node->attack mappings for strict (attack only) and causal (attack+contaminated) scopes."""
    if not attack_metadata:
        return {}, {}

    node_to_attack_ids_strict = defaultdict(set)
    node_to_attack_ids_causal = defaultdict(set)

    for attack_id, metadata in attack_metadata.items():
        attack_nodes = set(metadata.get("nids", []) or [])
        contaminated_nodes = set(metadata.get("contaminated_nids", []) or [])

        for node_id in attack_nodes:
            node_to_attack_ids_strict[node_id].add(attack_id)
            node_to_attack_ids_causal[node_id].add(attack_id)

        for node_id in contaminated_nodes:
            node_to_attack_ids_causal[node_id].add(attack_id)

    strict_dict = dict(node_to_attack_ids_strict)
    causal_dict = dict(node_to_attack_ids_causal)

    log(
        f"ADP mapping - Strict: {len(strict_dict)} nodes, Causal: {len(causal_dict)} nodes"
    )

    return strict_dict, causal_dict


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
    threshold, val_mcc = _find_threshold(val_scores, val_labels)

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

    # Compute confusion matrix components
    cm = confusion_matrix(test_labels, test_predictions)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    test_precision = precision_score(
        test_labels, test_predictions, average="binary", zero_division=0
    )
    test_recall = recall_score(
        test_labels, test_predictions, average="binary", zero_division=0
    )

    test_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    test_tpr = test_recall
    test_fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    test_tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    test_binary_f1 = f1_score(
        test_labels, test_predictions, average="binary", zero_division=0
    )
    test_mcc = matthews_corrcoef(test_labels, test_predictions)

    metrics = {
        "final_val_ap": val_ap,
        "final_val_mcc": val_mcc,
        "threshold": threshold,
        "final_test_ap": test_ap,
        "final_test_binary_f1": test_binary_f1,
        "final_test_mcc": test_mcc,
        "final_test_precision": test_precision,
        "final_test_recall": test_recall,
        "final_test_tpr": test_tpr,
        "final_test_fpr": test_fpr,
        "final_test_tnr": test_tnr,
        "final_test_fnr": test_fnr,
        "confusion_matrix_tn": int(tn),
        "confusion_matrix_fp": int(fp),
        "confusion_matrix_fn": int(fn),
        "confusion_matrix_tp": int(tp),
    }

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

                strict_cm = confusion_matrix(strict_labels, strict_predictions)
                s_tn, s_fp, s_fn, s_tp = (
                    strict_cm.ravel() if strict_cm.size == 4 else (0, 0, 0, 0)
                )

                strict_precision = precision_score(
                    strict_labels, strict_predictions, average="binary", zero_division=0
                )
                strict_recall = recall_score(
                    strict_labels, strict_predictions, average="binary", zero_division=0
                )

                strict_fpr = s_fp / (s_fp + s_tn) if (s_fp + s_tn) > 0 else 0.0
                strict_tpr = strict_recall
                strict_fnr = s_fn / (s_fn + s_tp) if (s_fn + s_tp) > 0 else 0.0
                strict_tnr = s_tn / (s_tn + s_fp) if (s_tn + s_fp) > 0 else 0.0
                strict_binary_f1 = f1_score(
                    strict_labels, strict_predictions, average="binary", zero_division=0
                )
                strict_mcc = matthews_corrcoef(strict_labels, strict_predictions)

                metrics.update(
                    {
                        "final_strict_test_ap": strict_ap,
                        "final_strict_test_binary_f1": strict_binary_f1,
                        "final_strict_test_mcc": strict_mcc,
                        "final_strict_test_precision": strict_precision,
                        "final_strict_test_recall": strict_recall,
                        "final_strict_test_tpr": strict_tpr,
                        "final_strict_test_fpr": strict_fpr,
                        "final_strict_test_tnr": strict_tnr,
                        "final_strict_test_fnr": strict_fnr,
                        "strict_confusion_matrix_tn": int(s_tn),
                        "strict_confusion_matrix_fp": int(s_fp),
                        "strict_confusion_matrix_fn": int(s_fn),
                        "strict_confusion_matrix_tp": int(s_tp),
                    }
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
        _build_node_to_attack_mappings(ground_truth)
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
    threshold, _ = _find_threshold(scores, labels)
    return val_ap, avg_loss, threshold
