#!/usr/bin/env python3
from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import Any, Callable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

PIDS_ROOT = Path(__file__).resolve().parents[1]
if str(PIDS_ROOT) not in sys.path:
    sys.path.insert(0, str(PIDS_ROOT))

from pidsmaker.config import get_runtime_required_args, get_yml_cfg
from pidsmaker.detection.evaluation_methods import (
    edge_evaluation,
    node_evaluation,
    node_tw_evaluation,
    tw_evaluation,
)
from pidsmaker.detection.evaluation_methods.evaluation_utils import (
    compute_tw_labels,
    listdir_sorted,
)

SelectionRule = str
StatsDict = dict[str, Any]
EpochRecord = dict[str, Any]

SELECTION_RULES: dict[SelectionRule, tuple[str, str]] = {
    "val_ap": ("val", "ap"),
    "val_pr_auc": ("val", "ap"),
    "val_mcc": ("val", "mcc"),
    "val_adp": ("val", "adp_score"),
    "val_discrimination": ("val", "discrimination"),
    "test_adp": ("test", "adp_score"),
}


def build_cfg(
    model: str,
    dataset: str,
    seed: int,
    artifact_dir: str | os.PathLike[str],
    csv_base_dir: str | os.PathLike[str],
    *,
    tuned: bool = True,
    cpu: bool = True,
) -> Any:
    args_list = [
        model,
        dataset,
        "--artifact_dir",
        str(artifact_dir),
        "--csv_base_dir",
        str(csv_base_dir),
        "--detection.gnn_training.seed",
        str(seed),
    ]
    if tuned:
        args_list.append("--tuned")
    if cpu:
        args_list.append("--cpu")
    args = get_runtime_required_args(args=args_list)
    return get_yml_cfg(args)


def get_evaluation_fn(cfg: Any) -> Callable[..., StatsDict]:
    method = cfg.detection.evaluation.used_method.strip()
    if method == "node_evaluation":
        return node_evaluation.main
    if method == "tw_evaluation":
        return tw_evaluation.main
    if method == "node_tw_evaluation":
        return node_tw_evaluation.main
    if method == "edge_evaluation":
        return edge_evaluation.main
    raise ValueError(
        f"Unsupported evaluation method for analysis-only replay: {method}. "
        "Only standard non-queue evaluation methods are supported."
    )


def ensure_tw_labels(cfg: Any) -> Any:
    tw_labels_file = Path(cfg.preprocessing.build_graphs._tw_labels) / "tw_to_malicious_nodes.pkl"
    if tw_labels_file.exists():
        return compute_tw_labels(cfg)
    return compute_tw_labels(cfg)


def get_epoch_names(cfg: Any) -> list[str]:
    val_losses_dir = Path(cfg.detection.gnn_training._edge_losses_dir) / "val"
    if not val_losses_dir.exists():
        raise FileNotFoundError(f"Validation edge-loss directory not found: {val_losses_dir}")
    return listdir_sorted(str(val_losses_dir))


def get_epoch_paths(cfg: Any, epoch_name: str) -> tuple[Path, Path]:
    val_tw_path = Path(cfg.detection.gnn_training._edge_losses_dir) / "val" / epoch_name
    test_tw_path = Path(cfg.detection.gnn_training._edge_losses_dir) / "test" / epoch_name
    if not val_tw_path.exists():
        raise FileNotFoundError(f"Validation epoch directory not found: {val_tw_path}")
    if not test_tw_path.exists():
        raise FileNotFoundError(f"Test epoch directory not found: {test_tw_path}")
    return val_tw_path, test_tw_path


def retarget_evaluation_outputs(cfg: Any, output_dir: Path) -> Any:
    cfg = cfg.clone()
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg.detection.evaluation._task_path = str(output_dir)
    cfg.detection.evaluation._precision_recall_dir = str(output_dir / "precision_recall_dir")
    cfg.detection.evaluation._uncertainty_exp_dir = str(output_dir / "uncertainty_exp")
    cfg.detection.evaluation._results_dir = str(output_dir / "results")

    cfg.detection.evaluation.queue_evaluation._precision_recall_dir = str(
        output_dir / "precision_recall_dir"
    )
    cfg.detection.evaluation.queue_evaluation._queues_dir = str(output_dir / "queues_dir")
    cfg.detection.evaluation.queue_evaluation._predicted_queues_dir = str(
        output_dir / "predicted_queues_dir"
    )
    cfg.detection.evaluation.queue_evaluation._kairos_dir = str(output_dir / "kairos_dir")

    for path in (
        cfg.detection.evaluation._precision_recall_dir,
        cfg.detection.evaluation._results_dir,
        cfg.detection.evaluation.queue_evaluation._precision_recall_dir,
        cfg.detection.evaluation.queue_evaluation._queues_dir,
        cfg.detection.evaluation.queue_evaluation._predicted_queues_dir,
        cfg.detection.evaluation.queue_evaluation._kairos_dir,
    ):
        os.makedirs(path, exist_ok=True)

    return cfg


def parse_epoch_index(epoch_name: str) -> int:
    try:
        return int(epoch_name.rsplit("_", 1)[-1])
    except ValueError:
        return 10**9


def _selection_metric(record: EpochRecord, selection_rule: SelectionRule) -> float:
    if selection_rule not in SELECTION_RULES:
        raise ValueError(
            f"Unknown selection rule '{selection_rule}'. "
            f"Expected one of: {', '.join(sorted(SELECTION_RULES))}"
        )
    split_name, metric_name = SELECTION_RULES[selection_rule]
    value = record.get(split_name, {}).get(metric_name, float("-inf"))
    return safe_float(value)


def safe_float(value: Any) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float("-inf")
    if math.isnan(value):
        return float("-inf")
    return value


def select_epoch_record(
    records: list[EpochRecord], selection_rule: SelectionRule
) -> EpochRecord:
    if not records:
        raise ValueError("No epoch records were provided for selection.")

    def key_fn(record: EpochRecord) -> tuple[float, float, float, float]:
        return (
            _selection_metric(record, selection_rule),
            safe_float(record.get("val", {}).get("ap")),
            safe_float(record.get("val", {}).get("mcc")),
            -parse_epoch_index(record["epoch"]),
        )

    return max(records, key=key_fn)


def sanitize_for_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): sanitize_for_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_for_json(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
    return value
