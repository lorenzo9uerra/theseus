from types import SimpleNamespace

import pytest
import torch

from tasks.evaluate import evaluate, validate


class ScoredGraph:
    def __init__(self, losses, labels, node_ids):
        self.losses = torch.tensor(losses, dtype=torch.float32).unsqueeze(1)
        self.x = torch.ones((len(losses), 1), dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.long)
        self.original_n_id = torch.tensor(node_ids, dtype=torch.long)

    def to(self, device):
        return self


class LossOnlyModel:
    def __call__(self, graph):
        return graph.losses, torch.zeros_like(graph.losses)

    def loss(self, outputs, encoded_target, reduction):
        assert reduction == "none"
        return outputs


def test_validate_uses_entity_level_aggregation():
    config = SimpleNamespace(device="cpu")
    val_data = [ScoredGraph([0.2, 0.6, 0.1], [0, 1, 0], [10, 10, 20])]

    val_ap, avg_loss, threshold = validate(LossOnlyModel(), val_data, config, {})

    assert val_ap == pytest.approx(1.0)
    assert avg_loss == pytest.approx(0.3)
    assert threshold == pytest.approx(0.1)


def test_evaluate_excludes_contaminated_nodes_from_metrics(monkeypatch, tmp_path):
    config = SimpleNamespace(device="cpu", outputs_dir=str(tmp_path))
    val_data = [ScoredGraph([0.9, 0.1], [1, 0], [10, 20])]
    test_data = [
        ScoredGraph([0.95, 0.85, 0.99, 0.1], [1, 1, 0, 0], [100, 200, 300, 400])
    ]
    ground_truth = {"A-1": {"nids": [100], "contaminated_nids": [200]}}

    monkeypatch.setattr("tasks.evaluate.get_excluded_node_ids", lambda config: {300})
    monkeypatch.setattr(
        "tasks.evaluate.plot_anomaly_score_distribution", lambda *args, **kwargs: None
    )
    monkeypatch.setattr("tasks.evaluate.plot_scores_neat", lambda *args, **kwargs: None)

    def fake_compute_adp_score(*args, **kwargs):
        return 0.4

    monkeypatch.setattr("tasks.evaluate.compute_adp_score", fake_compute_adp_score)

    metrics = evaluate(LossOnlyModel(), val_data, test_data, config, ground_truth)

    assert metrics["threshold"] == pytest.approx(0.1)
    assert metrics["final_test_precision"] == pytest.approx(1.0)
    assert metrics["final_test_recall"] == pytest.approx(1.0)
    assert metrics["confusion_matrix_fp"] == 0
    assert metrics["confusion_matrix_tn"] == 1
    assert metrics["confusion_matrix_tp"] == 1
    assert metrics["test_adp"] == pytest.approx(0.4)


def test_validate_excludes_contaminated_nodes_from_threshold_selection(monkeypatch):
    config = SimpleNamespace(device="cpu")
    val_data = [ScoredGraph([0.4, 0.2, 0.1], [1, 1, 0], [100, 200, 300])]
    ground_truth = {"A-1": {"nids": [100], "contaminated_nids": [200]}}

    monkeypatch.setattr("tasks.evaluate.get_excluded_node_ids", lambda config: set())

    val_ap, avg_loss, threshold = validate(
        LossOnlyModel(), val_data, config, ground_truth
    )

    assert val_ap == pytest.approx(1.0)
    assert avg_loss == pytest.approx((0.4 + 0.2 + 0.1) / 3)
    assert threshold == pytest.approx(0.1)


def test_evaluate_uses_train_calibration_when_validation_is_empty(
    monkeypatch, tmp_path
):
    config = SimpleNamespace(device="cpu", outputs_dir=str(tmp_path))
    calibration_data = [ScoredGraph([0.1, 0.1, 0.1], [0, 0, 0], [10, 20, 30])]
    test_data = [ScoredGraph([0.1, 0.5, 0.9], [0, 1, 1], [100, 200, 300])]

    monkeypatch.setattr("tasks.evaluate.get_excluded_node_ids", lambda config: set())
    monkeypatch.setattr(
        "tasks.evaluate.plot_anomaly_score_distribution", lambda *args, **kwargs: None
    )
    monkeypatch.setattr("tasks.evaluate.plot_scores_neat", lambda *args, **kwargs: None)
    monkeypatch.setattr("tasks.evaluate.compute_adp_score", lambda *args, **kwargs: 0.5)

    metrics = evaluate(
        LossOnlyModel(),
        [],
        test_data,
        config,
        {},
        calibration_data=calibration_data,
        calibration_split_name="Train",
    )

    assert metrics["final_val_ap"] == 0.0
    assert metrics["threshold_source"] == "train_max_benign"
    assert metrics["threshold"] == pytest.approx(0.1)
    assert metrics["final_test_recall"] == pytest.approx(1.0)
