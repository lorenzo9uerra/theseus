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

from utils.utils import log


def _safe_mcc(labels, predictions):
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)

    if labels.size == 0:
        return 0.0
    if np.unique(labels).size < 2 or np.unique(predictions).size < 2:
        return 0.0
    return float(matthews_corrcoef(labels, predictions))


def aggregate_to_entity_level(scores, labels, node_ids):
    """Max-pool per-node scores to a single entity-level score."""
    node_map = {}

    for score, label, nid in zip(scores, labels, node_ids, strict=True):
        node_id = int(nid)
        if node_id not in node_map:
            node_map[node_id] = {"score": -float("inf"), "label": 0}

        if score > node_map[node_id]["score"]:
            node_map[node_id]["score"] = score

        if label > node_map[node_id]["label"]:
            node_map[node_id]["label"] = label

    agg_scores = []
    agg_labels = []
    agg_nodes = []

    for node_id, data in node_map.items():
        agg_scores.append(data["score"])
        agg_labels.append(data["label"])
        agg_nodes.append(node_id)

    return (
        np.array(agg_scores),
        np.array(agg_labels, dtype=int),
        np.array(agg_nodes, dtype=int),
    )


@torch.inference_mode()
def score_process_nodes(model, graph, config, *, return_node_indices=False):
    """Compute process-node anomaly scores for a single graph."""
    graph = graph.to(device=config.device)
    outputs, encoded_target = model(graph)
    losses = model.loss(outputs, encoded_target, "none")

    if losses.dim() > 1:
        losses = losses.mean(dim=tuple(range(1, losses.dim())))

    labels_tensor = graph.y
    if labels_tensor.dim() > 1:
        labels_tensor = labels_tensor.view(-1)

    process_mask = graph.x[:, 0] == 1
    if not process_mask.any():
        empty_float = np.empty(0, dtype=np.float64)
        empty_int = np.empty(0, dtype=np.int64)
        if return_node_indices:
            return empty_float, empty_int, empty_int, empty_int
        return empty_float, empty_int, empty_int

    scores = losses[process_mask].detach().cpu().numpy().astype(np.float64, copy=False)
    labels = (
        labels_tensor[process_mask].detach().cpu().numpy().astype(np.int64, copy=False)
    )
    node_ids = (
        graph.original_n_id[process_mask]
        .detach()
        .cpu()
        .numpy()
        .astype(np.int64, copy=False)
    )

    if return_node_indices:
        node_indices = (
            torch.where(process_mask)[0]
            .detach()
            .cpu()
            .numpy()
            .astype(np.int64, copy=False)
        )
        return scores, labels, node_ids, node_indices

    return scores, labels, node_ids


@torch.inference_mode()
def inference_loop(model, dataset, config):
    """Compute per-process-node anomaly scores."""
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


def find_threshold(scores, labels):
    """Find the MCC-maximizing decision threshold on validation scores."""
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

    for idx, threshold in enumerate(thresholds):
        predictions = (scores >= threshold).astype(int)
        mcc = _safe_mcc(labels, predictions)

        if mcc > best_mcc:
            best_mcc = mcc
            best_idx = idx

    return thresholds[best_idx], best_mcc


def select_max_benign_threshold(scores, labels):
    """Select the maximum finite calibration score among benign examples."""
    if labels.size == 0 or scores.size == 0:
        return np.inf

    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels)
    finite_mask = np.isfinite(scores)
    if not finite_mask.any():
        return np.inf

    scores = scores[finite_mask]
    labels = labels[finite_mask]
    benign_scores = scores[labels == 0]
    if benign_scores.size > 0:
        return float(benign_scores.max())
    return float(scores.max())


def threshold_predictions(scores, threshold):
    """Apply the shared exclusive decision rule used by all methods."""
    if np.isinf(threshold):
        return np.zeros_like(scores, dtype=int)
    return (scores > threshold).astype(int)


def compute_binary_metrics(labels, predictions, prefix, confusion_prefix):
    """Compute thresholded binary metrics under a shared metric schema."""
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()

    precision = precision_score(labels, predictions, average="binary", zero_division=0)
    recall = recall_score(labels, predictions, average="binary", zero_division=0)
    binary_f1 = f1_score(labels, predictions, average="binary", zero_division=0)
    mcc = _safe_mcc(labels, predictions)

    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    tpr = recall
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        f"{prefix}_binary_f1": binary_f1,
        f"{prefix}_mcc": mcc,
        f"{prefix}_precision": precision,
        f"{prefix}_recall": recall,
        f"{prefix}_tpr": tpr,
        f"{prefix}_fpr": fpr,
        f"{prefix}_tnr": tnr,
        f"{prefix}_fnr": fnr,
        f"{confusion_prefix}_tn": int(tn),
        f"{confusion_prefix}_fp": int(fp),
        f"{confusion_prefix}_fn": int(fn),
        f"{confusion_prefix}_tp": int(tp),
    }


def compute_threshold_metrics(scores, labels, threshold):
    """Compute thresholded binary metrics plus AP on scored entities."""
    if labels.size == 0 or scores.size == 0:
        return {
            "ap": 0.0,
            "mcc": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "fpr": 0.0,
            "fnr": 0.0,
            "tn": 0,
            "fp": 0,
            "fn": 0,
            "tp": 0,
            "entity_count": 0,
            "positive_entities": 0,
            "predicted_positive_entities": 0,
        }

    predictions = threshold_predictions(scores, threshold)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    precision = float(precision_score(labels, predictions, zero_division=0))
    recall = float(recall_score(labels, predictions, zero_division=0))
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
    mcc = _safe_mcc(labels, predictions)
    f1 = float(f1_score(labels, predictions, average="binary", zero_division=0))
    ap = float(average_precision_score(labels, scores)) if np.any(labels == 1) else 0.0

    return {
        "ap": ap,
        "mcc": mcc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fpr,
        "fnr": fnr,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "entity_count": int(labels.size),
        "positive_entities": int(labels.sum()),
        "predicted_positive_entities": int(predictions.sum()),
    }


def build_node_to_attack_mappings(attack_metadata):
    """Build a strict node-to-attack map for attack-chain ADP evaluation."""
    if not attack_metadata:
        return {}

    node_to_attack_ids = defaultdict(set)

    for attack_id, metadata in attack_metadata.items():
        attack_nodes = set(metadata.get("nids", []) or [])

        for node_id in attack_nodes:
            node_to_attack_ids[node_id].add(attack_id)

    mapping = dict(node_to_attack_ids)
    log(f"ADP mapping: {len(mapping)} attack nodes")
    return mapping
