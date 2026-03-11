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

    val_ap, avg_loss, threshold = validate(LossOnlyModel(), val_data, config)

    assert val_ap == pytest.approx(1.0)
    assert avg_loss == pytest.approx(0.3)
    assert threshold == pytest.approx(0.6)


def test_evaluate_filters_excluded_nodes_and_reports_strict_metrics(
    monkeypatch, tmp_path
):
    config = SimpleNamespace(device="cpu", outputs_dir=str(tmp_path))
    val_data = [ScoredGraph([0.9, 0.1], [1, 0], [10, 20])]
    test_data = [ScoredGraph([0.95, 0.85, 0.99], [1, 1, 0], [100, 200, 300])]
    ground_truth = {"A-1": {"nids": [100], "contaminated_nids": [200]}}

    monkeypatch.setattr("tasks.evaluate.get_excluded_node_ids", lambda config: {300})
    monkeypatch.setattr(
        "tasks.evaluate.plot_anomaly_score_distribution", lambda *args, **kwargs: None
    )
    monkeypatch.setattr("tasks.evaluate.plot_scores_neat", lambda *args, **kwargs: None)

    def fake_compute_adp_score(*args, **kwargs):
        out_file = kwargs["out_file"]
        return 0.4 if "strict" in out_file else 0.7

    monkeypatch.setattr("tasks.evaluate.compute_adp_score", fake_compute_adp_score)

    metrics = evaluate(LossOnlyModel(), val_data, test_data, config, ground_truth)

    assert metrics["threshold"] == pytest.approx(0.9)
    assert metrics["final_test_precision"] == pytest.approx(1.0)
    assert metrics["final_test_recall"] == pytest.approx(0.5)
    assert metrics["confusion_matrix_fp"] == 0
    assert metrics["final_strict_test_recall"] == pytest.approx(1.0)
    assert metrics["strict_confusion_matrix_tp"] == 1
    assert metrics["test_adp_strict"] == pytest.approx(0.4)
    assert metrics["test_adp_causal"] == pytest.approx(0.7)
