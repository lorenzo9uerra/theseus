from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tasks.evaluate_support import (
    aggregate_to_entity_level,
    build_node_to_attack_mappings,
    compute_binary_metrics,
    find_threshold,
    inference_loop,
)


class DummyGraph:
    def __init__(self):
        self.x = torch.tensor([[1.0], [0.0], [1.0]], dtype=torch.float32)
        self.y = torch.tensor([1, 0, 0], dtype=torch.long)
        self.original_n_id = torch.tensor([10, 20, 30], dtype=torch.long)

    def to(self, device):
        return self


class DummyModel:
    def __call__(self, graph):
        outputs = torch.tensor([[1.0, 3.0], [5.0, 7.0], [2.0, 4.0]])
        encoded_target = torch.zeros_like(outputs)
        return outputs, encoded_target

    def loss(self, outputs, encoded_target, reduction):
        assert reduction == "none"
        return outputs


def test_aggregate_to_entity_level_max_pools_scores_and_labels():
    scores = np.array([0.1, 0.7, 0.2, 0.3])
    labels = np.array([0, 1, 0, 1])
    node_ids = np.array([5, 5, 7, 7])

    agg_scores, agg_labels, agg_nodes = aggregate_to_entity_level(
        scores, labels, node_ids
    )

    assert agg_nodes.tolist() == [5, 7]
    assert agg_scores.tolist() == [0.7, 0.3]
    assert agg_labels.tolist() == [1, 1]


def test_find_threshold_returns_inf_without_positive_labels():
    threshold, best_mcc = find_threshold(np.array([0.2, 0.5]), np.array([0, 0]))

    assert np.isinf(threshold)
    assert best_mcc == 0.0


def test_compute_binary_metrics_returns_expected_schema():
    metrics = compute_binary_metrics(
        np.array([0, 0, 1, 1]),
        np.array([0, 1, 0, 1]),
        "final_test",
        "confusion_matrix",
    )

    assert metrics["final_test_precision"] == pytest.approx(0.5)
    assert metrics["final_test_recall"] == pytest.approx(0.5)
    assert metrics["confusion_matrix_tn"] == 1
    assert metrics["confusion_matrix_fp"] == 1
    assert metrics["confusion_matrix_fn"] == 1
    assert metrics["confusion_matrix_tp"] == 1


def test_build_node_to_attack_mappings_separates_strict_and_causal():
    strict_map, causal_map = build_node_to_attack_mappings(
        {"A-1": {"nids": [1], "contaminated_nids": [2, 3]}}
    )

    assert strict_map == {1: {"A-1"}}
    assert causal_map == {1: {"A-1"}, 2: {"A-1"}, 3: {"A-1"}}


def test_inference_loop_scores_process_nodes_only():
    config = SimpleNamespace(device="cpu")

    scores, labels, node_ids, avg_loss = inference_loop(
        DummyModel(), [DummyGraph()], config
    )

    assert scores.tolist() == [2.0, 3.0]
    assert labels.tolist() == [1, 0]
    assert node_ids.tolist() == [10, 30]
    assert avg_loss == pytest.approx(2.5)
