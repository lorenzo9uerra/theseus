#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from pathlib import Path
from typing import Any, Union

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import matplotlib.pyplot as plt
import numpy as np
import torch
from _analysis_common import build_cfg, get_epoch_paths

plt.rcParams.update(
    {
        "font.size": 21,
        "axes.titlesize": 31,
        "axes.labelsize": 26,
        "xtick.labelsize": 21,
        "ytick.labelsize": 21,
        "legend.fontsize": 21,
        "figure.titlesize": 29,
    }
)

from pidsmaker.detection.evaluation_methods.evaluation_utils import (
    classifier_evaluation,
    get_max_benign_threshold,
)

DEFAULT_DATASETS = ("CADETS_E3", "FIVEDIRECTIONS_E3", "THEIA_E3", "TRACE_E3")
DEFAULT_SEEDS = (111, 333, 828, 0, 433)
DEFAULT_MULTIPLIERS = (
    0.25,
    0.3,
    0.35,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
    1.0,
    1.1,
    1.2,
    1.4,
    1.6,
    1.8,
    2.0,
    2.5,
    3.0,
    3.5,
    4.0,
)
PLOT_METRICS = (
    ("precision", "Precision", "#b23a48", "-"),
    ("recall", "Recall", "#1d3557", "-"),
    ("f1", "F1", "#2a9d8f", "-"),
    ("mcc", "MCC", "#222222", "-"),
    ("attack_coverage", "Attack Coverage", "#e76f51", "--"),
)
PLOT_X_TICKS = (0.25, 0.5, 1.0, 2.0, 4.0)
PLOT_X_TICKLABELS = ("0.25x", "0.5x", "1.0x", "2.0x", "4.0x")
DATASET_TITLES = {
    "CADETS_E3": "Cadets",
    "FIVEDIRECTIONS_E3": "FiveDirections",
    "THEIA_E3": "Theia",
    "TRACE_E3": "Trace",
}
ScalarValue = Union[float, int, str]
SweepRow = dict[str, ScalarValue]
SweepSummary = dict[str, ScalarValue]
AggregateRow = dict[str, ScalarValue]


def _canonical_relative_threshold(value: float) -> float:
    return round(float(value), 10)

BEST_EPOCH_RE = re.compile(r"Best epoch selected from validation:\s+(model_epoch_\d+)")
SCORES_FILE_RE = re.compile(r"scores_file:\s+(.+scores_model_epoch_\d+\.pkl)")


def _parse_csv_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _safe_torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _load_score_payload(scores_file: Path) -> dict[str, Any]:
    payload = _safe_torch_load(scores_file)
    required = {"pred_scores", "y_truth", "nodes", "node2attacks"}
    missing = required - set(payload)
    if missing:
        raise KeyError(f"Score file {scores_file} is missing keys: {sorted(missing)}")
    return payload


def _threshold_grid(
    base_threshold: float, multipliers: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    thresholds = np.unique(base_threshold * multipliers)
    return thresholds.astype(np.float64), (thresholds / base_threshold).astype(np.float64)


def _compute_attack_coverage(
    nodes: np.ndarray,
    predictions: np.ndarray,
    node2attacks: dict[int, set[str] | list[str] | tuple[str, ...]],
) -> float:
    attacks_in_eval = set()
    for node in nodes.tolist():
        attacks_in_eval.update(node2attacks.get(int(node), []))

    if not attacks_in_eval:
        return 0.0

    detected = set()
    for node, pred in zip(nodes.tolist(), predictions.tolist()):
        if pred != 1:
            continue
        detected.update(node2attacks.get(int(node), []))

    return len(detected) / len(attacks_in_eval)


def _compute_adp(
    scores: np.ndarray,
    nodes: np.ndarray,
    labels: np.ndarray,
    node2attacks: dict[int, set[str] | list[str] | tuple[str, ...]],
) -> float:
    attacks_in_eval = set()
    for node in nodes.tolist():
        attacks_in_eval.update(node2attacks.get(int(node), []))

    total_attacks = len(attacks_in_eval)
    if total_attacks == 0:
        return 0.0

    order = np.argsort(scores)[::-1]
    sorted_scores = scores[order]
    sorted_nodes = nodes[order]
    sorted_labels = labels[order]

    detected_attacks = set()
    tp = 0
    fp = 0
    points: list[tuple[float, float]] = []

    last_score = None
    for score, node, label in zip(sorted_scores, sorted_nodes, sorted_labels):
        if last_score is not None and score != last_score and tp + fp > 0:
            precision = tp / (tp + fp)
            attack_pct = len(detected_attacks) / total_attacks
            points.append((precision, attack_pct))

        if int(label) == 1:
            tp += 1
            detected_attacks.update(node2attacks.get(int(node), []))
        else:
            fp += 1
        last_score = float(score)

    if tp + fp > 0:
        precision = tp / (tp + fp)
        attack_pct = len(detected_attacks) / total_attacks
        points.append((precision, attack_pct))

    if not points:
        return 0.0

    points = sorted(points)
    precisions = np.array([p for p, _ in points], dtype=np.float64)
    detected = np.array([d for _, d in points], dtype=np.float64)
    if len(points) == 1:
        return float(detected[0] * precisions[0])

    deltas = np.diff(precisions)
    heights = (detected[1:] + detected[:-1]) * 0.5
    return float(np.sum(deltas * heights))


def _parse_log_for_selection(log_file: Path, project_root: Path) -> tuple[str, Path]:
    selected_epoch = None
    scores_file = None

    with log_file.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = BEST_EPOCH_RE.search(line)
            if match:
                selected_epoch = match.group(1)
            match = SCORES_FILE_RE.search(line)
            if match:
                scores_file = match.group(1).strip()

    if selected_epoch is None:
        raise ValueError(f"Could not find selected epoch in {log_file}")
    if scores_file is None:
        raise ValueError(f"Could not find scores_file in {log_file}")

    scores_path = Path(scores_file)
    if not scores_path.is_absolute():
        scores_path = (project_root / "baselines" / "PIDSMaker" / scores_path).resolve()
    return selected_epoch, scores_path


def _dataset_slug(dataset: str) -> str:
    return dataset.lower()


def _run_single_sweep(
    *,
    model: str,
    dataset: str,
    seed: int,
    project_root: Path,
    artifact_dir: Path,
    csv_base_dir: Path,
    log_dir: Path,
    multipliers: np.ndarray,
    output_dir: Path,
) -> tuple[list[SweepRow], SweepSummary]:
    log_file = log_dir / f"{model}_{dataset}_seed{seed}.log"
    selected_epoch, scores_file = _parse_log_for_selection(log_file, project_root)
    payload = _load_score_payload(scores_file)

    cfg = build_cfg(model, dataset, seed, artifact_dir, csv_base_dir)
    val_tw_path, _ = get_epoch_paths(cfg, selected_epoch)
    base_threshold = float(
        get_max_benign_threshold(
            str(val_tw_path),
            cfg,
            use_dst_node_loss=cfg.detection.evaluation.node_evaluation.use_dst_node_loss,
        )
    )

    scores = np.asarray(payload["pred_scores"], dtype=np.float64)
    labels = np.asarray(payload["y_truth"], dtype=np.int32)
    nodes = np.asarray(payload["nodes"], dtype=np.int64)
    node2attacks: dict[int, set[str] | list[str] | tuple[str, ...]] = {
        int(node): set(str(attack) for attack in attacks)
        for node, attacks in payload["node2attacks"].items()
    }

    thresholds, relative = _threshold_grid(base_threshold, multipliers)
    adp = _compute_adp(scores, nodes, labels, node2attacks)

    rows: list[SweepRow] = []
    for thr, rel in zip(thresholds.tolist(), relative.tolist()):
        predictions = (scores > thr).astype(np.int32)
        stats = classifier_evaluation(labels.tolist(), predictions.tolist(), scores.tolist())
        tp = int(stats["tp"])
        tn = int(stats["tn"])
        fp = int(stats["fp"])
        fn = int(stats["fn"])

        row: SweepRow = {
            "model": model,
            "dataset": dataset,
            "seed": seed,
            "selected_epoch": selected_epoch,
            "scores_file": str(scores_file),
            "base_threshold": base_threshold,
            "threshold": float(thr),
            "threshold_relative": float(rel),
            "precision": float(stats["precision"]),
            "recall": float(stats["recall"]),
            "f1": float(stats["fscore"]),
            "mcc": float(stats["mcc"]),
            "fpr": float(stats["fpr"]),
            "fnr": (fn / (fn + tp)) if (fn + tp) > 0 else 0.0,
            "predicted_positive_fraction": float((tp + fp) / max(1, tp + tn + fp + fn)),
            "attack_coverage": _compute_attack_coverage(nodes, predictions, node2attacks),
            "adp": adp,
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
        }
        rows.append(row)

    summary: SweepSummary = {
        "model": model,
        "dataset": dataset,
        "seed": seed,
        "selected_epoch": selected_epoch,
        "scores_file": str(scores_file),
        "base_threshold": base_threshold,
        "n_nodes": int(labels.size),
        "n_attacks": int(labels.sum()),
        "adp": adp,
    }

    per_seed_dir = output_dir / "per_seed"
    per_seed_dir.mkdir(parents=True, exist_ok=True)
    out_path = per_seed_dir / f"{model}_{_dataset_slug(dataset)}_seed{seed}.csv"
    _write_csv(rows, out_path)
    return rows, summary


def _aggregate_rows(rows: list[SweepRow]) -> list[AggregateRow]:
    grouped: dict[tuple[str, str, float], list[SweepRow]] = {}
    for row in rows:
        key = (
            str(row["model"]),
            str(row["dataset"]),
            _canonical_relative_threshold(float(row["threshold_relative"])),
        )
        grouped.setdefault(key, []).append(row)

    metrics = (
        "base_threshold",
        "threshold",
        "precision",
        "recall",
        "f1",
        "mcc",
        "fpr",
        "fnr",
        "predicted_positive_fraction",
        "attack_coverage",
        "adp",
    )
    aggregated: list[AggregateRow] = []
    for (model, dataset, threshold_relative), group in sorted(grouped.items()):
        row: AggregateRow = {
            "model": model,
            "dataset": dataset,
            "threshold_relative": threshold_relative,
            "n_seeds": len(group),
        }
        for metric in metrics:
            values = np.asarray([float(item[metric]) for item in group], dtype=np.float64)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=0))
        aggregated.append(row)
    return aggregated


def _plot_aggregate(aggregate_rows: list[AggregateRow], output_dir: Path, model: str) -> None:
    datasets = sorted({str(row["dataset"]) for row in aggregate_rows})
    fig, axes = plt.subplots(2, 2, figsize=(17.6, 12.2), sharex=True, sharey=True)
    axes = axes.flatten()

    for ax, dataset in zip(axes, datasets):
        dataset_rows = [row for row in aggregate_rows if row["dataset"] == dataset]
        dataset_rows.sort(key=lambda row: float(row["threshold_relative"]))

        x = np.asarray([float(row["threshold_relative"]) for row in dataset_rows], dtype=np.float64)
        adp = float(dataset_rows[0]["adp_mean"]) if dataset_rows else 0.0

        for metric, label, color, linestyle in PLOT_METRICS:
            y = np.asarray([float(row[f"{metric}_mean"]) for row in dataset_rows], dtype=np.float64)
            ax.plot(x, y, label=label, color=color, linestyle=linestyle, linewidth=2.8)

        ax.axvline(1.0, color="#666666", linestyle=":", linewidth=1.5)
        ax.set_xscale("log")
        ax.set_xlim(0.25, 4.0)
        ax.set_xticks(PLOT_X_TICKS)
        ax.set_xticklabels(PLOT_X_TICKLABELS)
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(DATASET_TITLES.get(dataset, dataset))
        ax.grid(True, alpha=0.25)
        ax.text(
            0.97,
            0.95,
            f"ADP={adp:.3f}",
            transform=ax.transAxes,
            fontsize=21,
            ha="right",
            va="top",
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
        )

    for ax in axes[2:]:
        ax.set_xlabel("Threshold / reported threshold")
    for ax in axes[::2]:
        ax.set_ylabel("Metric value")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=len(PLOT_METRICS),
        frameon=False,
    )
    fig.supxlabel("Threshold / reported threshold (max benign calibration score)")
    fig.supylabel("Metric value")
    fig.tight_layout(rect=(0, 0, 1, 0.92))

    png_path = output_dir / f"{model}_threshold_sensitivity_e3.png"
    pdf_path = output_dir / f"{model}_threshold_sensitivity_e3.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _write_csv(rows: list[dict[str, Any]], out_path: Path) -> None:
    if not rows:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _load_existing_per_seed_rows(per_seed_dir: Path, model: str) -> list[SweepRow]:
    rows: list[SweepRow] = []
    if not per_seed_dir.is_dir():
        return rows
    for path in sorted(per_seed_dir.glob(f"{model}_*.csv")):
        rows.extend(_read_csv_rows(path))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep threshold multipliers around the current max-benign validation threshold "
            "for PIDSMaker methods using frozen strict score payloads from final evaluation logs."
        )
    )
    parser.add_argument("model", choices=("velox", "orthrus"))
    parser.add_argument(
        "--project-root",
        type=Path,
        default=PROJECT_ROOT,
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=None,
        help="Defaults to <project-root>/baselines/PIDSMaker/artifacts",
    )
    parser.add_argument(
        "--csv-base-dir",
        type=Path,
        default=None,
        help="Defaults to <project-root>/data/DARPA",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="Defaults to <project-root>/baselines/PIDSMaker/results_eval_maxbenign_20260325",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/threshold_sensitivity_velox_e3"),
    )
    parser.add_argument(
        "--datasets",
        default=",".join(DEFAULT_DATASETS),
    )
    parser.add_argument(
        "--seeds",
        default=",".join(str(seed) for seed in DEFAULT_SEEDS),
    )
    parser.add_argument(
        "--multipliers",
        default=",".join(str(m) for m in DEFAULT_MULTIPLIERS),
    )
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    artifact_dir = (
        args.artifact_dir.resolve()
        if args.artifact_dir is not None
        else (project_root / "baselines" / "PIDSMaker" / "artifacts")
    )
    csv_base_dir = (
        args.csv_base_dir.resolve()
        if args.csv_base_dir is not None
        else (project_root / "data" / "DARPA")
    )
    log_dir = (
        args.log_dir.resolve()
        if args.log_dir is not None
        else (project_root / "baselines" / "PIDSMaker" / "results_eval_maxbenign_20260325")
    )
    output_dir = args.output_dir.resolve()

    datasets = _parse_csv_list(args.datasets)
    seeds = [int(seed) for seed in _parse_csv_list(args.seeds)]
    multipliers = np.asarray(
        [float(mult) for mult in _parse_csv_list(args.multipliers)], dtype=np.float64
    )

    all_rows: list[SweepRow] = []
    summaries: list[SweepSummary] = []
    per_seed_dir = output_dir / "per_seed"
    per_seed_dir.mkdir(parents=True, exist_ok=True)

    for dataset in datasets:
        for seed in seeds:
            existing_csv = per_seed_dir / f"{args.model}_{_dataset_slug(dataset)}_seed{seed}.csv"
            if existing_csv.exists():
                print(f"Reusing existing sweep for {args.model} {dataset} seed={seed}...")
                cached_rows = _read_csv_rows(existing_csv)
                all_rows.extend(cached_rows)
                if cached_rows:
                    first = cached_rows[0]
                    summaries.append(
                        {
                            "model": first["model"],
                            "dataset": first["dataset"],
                            "seed": int(first["seed"]),
                            "selected_epoch": first["selected_epoch"],
                            "scores_file": first["scores_file"],
                            "base_threshold": float(first["base_threshold"]),
                            "n_nodes": "",
                            "n_attacks": "",
                            "adp": float(first["adp"]),
                        }
                    )
                continue
            print(f"Sweeping {args.model} {dataset} seed={seed}...")
            rows, summary = _run_single_sweep(
                model=args.model,
                dataset=dataset,
                seed=seed,
                project_root=project_root,
                artifact_dir=artifact_dir,
                csv_base_dir=csv_base_dir,
                log_dir=log_dir,
                multipliers=multipliers,
                output_dir=output_dir,
            )
            all_rows.extend(rows)
            summaries.append(summary)

    full_rows = _load_existing_per_seed_rows(per_seed_dir, args.model)
    if not full_rows:
        full_rows = all_rows

    summary_by_key: dict[tuple[str, int], SweepSummary] = {}
    for summary in summaries:
        summary_by_key[(str(summary["dataset"]), int(summary["seed"]))] = summary
    for row in full_rows:
        key = (str(row["dataset"]), int(row["seed"]))
        if key not in summary_by_key:
            summary_by_key[key] = {
                "model": row["model"],
                "dataset": row["dataset"],
                "seed": int(row["seed"]),
                "selected_epoch": row["selected_epoch"],
                "scores_file": row["scores_file"],
                "base_threshold": float(row["base_threshold"]),
                "n_nodes": "",
                "n_attacks": "",
                "adp": float(row["adp"]),
            }

    aggregate_rows = _aggregate_rows(full_rows)
    _write_csv(full_rows, output_dir / f"{args.model}_threshold_sensitivity_full.csv")
    _write_csv(aggregate_rows, output_dir / f"{args.model}_threshold_sensitivity_aggregate.csv")
    ordered_summaries = [
        summary_by_key[key] for key in sorted(summary_by_key, key=lambda item: (item[0], item[1]))
    ]
    _write_csv(
        ordered_summaries, output_dir / f"{args.model}_threshold_sensitivity_run_summary.csv"
    )
    _plot_aggregate(aggregate_rows, output_dir, args.model)

    print(f"Wrote threshold sensitivity outputs to {output_dir}")


if __name__ == "__main__":
    main()
