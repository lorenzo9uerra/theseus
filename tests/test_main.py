from types import SimpleNamespace

import pytest

import main


class DummyModel:
    def __init__(self):
        self.eval_called = False

    def eval(self):
        self.eval_called = True


def make_config(**overrides):
    config = SimpleNamespace(seed=123, test=False, checkpoint=None)
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def test_main_training_path_dispatches_tasks(monkeypatch):
    config = make_config()
    call_log = []
    dummy_model = DummyModel()
    graphs = {"train": ["train_graph"], "val": ["val_graph"], "test": ["test_graph"]}
    ground_truth = {"attack": {"nids": [1], "contaminated_nids": [2]}}
    metrics = {"final_test_ap": 0.5}

    monkeypatch.setattr(
        "main.set_seed", lambda seed: call_log.append(("set_seed", seed))
    )
    monkeypatch.setattr("main.build_graphs", lambda cfg: (graphs, ground_truth))
    monkeypatch.setattr(
        "main.train",
        lambda cfg, train_data, val_data, test_data, ground_truth: (
            call_log.append(("train", train_data, val_data, test_data, ground_truth))
            or dummy_model
        ),
    )
    monkeypatch.setattr(
        "main.evaluate",
        lambda model, val_data, test_data, cfg, gt, **kwargs: (
            call_log.append(("evaluate", model, val_data, test_data, gt, kwargs))
            or metrics
        ),
    )
    monkeypatch.setattr(
        "main.wandb.log",
        lambda logged_metrics: call_log.append(("wandb.log", logged_metrics)),
    )
    monkeypatch.setattr(
        "main.initialize_model",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("initialize_model should not run in training mode")
        ),
    )

    main.main(config)

    assert call_log == [
        ("set_seed", 123),
        ("train", ["train_graph"], ["val_graph"], ["test_graph"], ground_truth),
        (
            "evaluate",
            dummy_model,
            ["val_graph"],
            ["test_graph"],
            ground_truth,
            {"calibration_data": ["val_graph"], "calibration_split_name": "Validation"},
        ),
        ("wandb.log", metrics),
    ]


def test_main_test_path_loads_checkpoint_and_evaluates(monkeypatch, tmp_path):
    checkpoint_path = tmp_path / "checkpoint.pt"
    checkpoint_path.write_text("ok")
    config = make_config(test=True, checkpoint=str(checkpoint_path))
    call_log = []
    dummy_model = DummyModel()
    graphs = {"train": ["train_graph"], "val": ["val_graph"], "test": ["test_graph"]}
    ground_truth = {"attack": {"nids": [1], "contaminated_nids": [2]}}
    metrics = {"final_test_ap": 0.7}

    monkeypatch.setattr(
        "main.set_seed", lambda seed: call_log.append(("set_seed", seed))
    )
    monkeypatch.setattr("main.build_graphs", lambda cfg: (graphs, ground_truth))
    monkeypatch.setattr(
        "main.initialize_model",
        lambda cfg, train_data, checkpoint: (
            call_log.append(("initialize_model", train_data, checkpoint)) or dummy_model
        ),
    )
    monkeypatch.setattr(
        "main.evaluate",
        lambda model, val_data, test_data, cfg, gt, **kwargs: (
            call_log.append(("evaluate", model, val_data, test_data, gt, kwargs))
            or metrics
        ),
    )
    monkeypatch.setattr(
        "main.wandb.log",
        lambda logged_metrics: call_log.append(("wandb.log", logged_metrics)),
    )
    monkeypatch.setattr(
        "main.train",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("train should not run in test mode")
        ),
    )

    main.main(config)

    assert dummy_model.eval_called is True
    assert call_log == [
        ("set_seed", 123),
        ("initialize_model", ["train_graph"], str(checkpoint_path)),
        (
            "evaluate",
            dummy_model,
            ["val_graph"],
            ["test_graph"],
            ground_truth,
            {"calibration_data": ["val_graph"], "calibration_split_name": "Validation"},
        ),
        ("wandb.log", metrics),
    ]


def test_main_uses_train_calibration_when_validation_split_is_empty(monkeypatch):
    config = make_config()
    call_log = []
    dummy_model = DummyModel()
    graphs = {"train": ["train_graph"], "val": [], "test": ["test_graph"]}
    ground_truth = {"attack": {"nids": [1], "contaminated_nids": [2]}}
    metrics = {"final_test_ap": 0.5}

    monkeypatch.setattr(
        "main.set_seed", lambda seed: call_log.append(("set_seed", seed))
    )
    monkeypatch.setattr("main.build_graphs", lambda cfg: (graphs, ground_truth))
    monkeypatch.setattr(
        "main.train",
        lambda cfg, train_data, val_data, test_data, ground_truth: (
            call_log.append(("train", train_data, val_data, test_data, ground_truth))
            or dummy_model
        ),
    )
    monkeypatch.setattr(
        "main.evaluate",
        lambda model, val_data, test_data, cfg, gt, **kwargs: (
            call_log.append(("evaluate", model, val_data, test_data, gt, kwargs))
            or metrics
        ),
    )
    monkeypatch.setattr(
        "main.wandb.log",
        lambda logged_metrics: call_log.append(("wandb.log", logged_metrics)),
    )

    main.main(config)

    assert call_log == [
        ("set_seed", 123),
        ("train", ["train_graph"], [], ["test_graph"], ground_truth),
        (
            "evaluate",
            dummy_model,
            [],
            ["test_graph"],
            ground_truth,
            {"calibration_data": ["train_graph"], "calibration_split_name": "Train"},
        ),
        ("wandb.log", metrics),
    ]


def test_main_test_path_requires_checkpoint(monkeypatch):
    config = make_config(test=True, checkpoint=None)

    monkeypatch.setattr(
        "main.build_graphs", lambda cfg: ({"train": [], "val": [], "test": []}, {})
    )

    with pytest.raises(
        ValueError, match="config.checkpoint must be specified when config.test is True"
    ):
        main.main(config)


def test_main_test_path_rejects_missing_checkpoint(monkeypatch, tmp_path):
    missing_checkpoint = tmp_path / "missing.pt"
    config = make_config(test=True, checkpoint=str(missing_checkpoint))

    monkeypatch.setattr(
        "main.build_graphs", lambda cfg: ({"train": [], "val": [], "test": []}, {})
    )

    with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
        main.main(config)
