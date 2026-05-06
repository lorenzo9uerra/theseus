#!/usr/bin/env python3
# ruff: noqa: E402
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np

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

from tasks.build_graphs import build_graphs
from tasks.evaluate import (
    _collect_entity_scores,
    _exclude_contaminated_entities,
    _get_contaminated_node_ids,
    _get_metric_excluded_node_ids,
)
from tasks.evaluate_support import (
    build_node_to_attack_mappings,
    compute_threshold_metrics,
    select_max_benign_threshold,
    threshold_predictions,
)
from tasks.training import initialize_model
from utils.evaluate_utils import compute_adp_score
from utils.ground_truth import get_excluded_node_ids
from utils.parsers import parse_args, parse_config
from utils.utils import log

DEFAULT_DATASETS = ("CADETS_E3", "FIVEDIRECTIONS_E3", "THEIA_E3", "TRACE_E3")
DEFAULT_SEEDS = (65129, 923457, 56604, 9382, 58371)
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
ScalarValue = float | int | str
SweepRow = dict[str, ScalarValue]
SweepSummary = dict[str, ScalarValue]
AggregateRow = dict[str, ScalarValue]


def _canonical_relative_threshold(value: float) -> float:
    return round(float(value), 10)


def set_seed(seed: int) -> None:
    np.random.seed(seed)


def _parse_csv_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _dataset_lower(dataset: str) -> str:
    return dataset.lower()


def _checkpoint_path(artifact_root: Path, dataset: str, seed: int) -> Path:
    return (
        artifact_root
        / "checkpoints"
        / "theseus"
        / f"checkpoint_theseus_{_dataset_lower(dataset)}_seed_{seed}_paper.pt"
    )


def _build_config(dataset: str, seed: int, artifact_root: Path, run_output_dir: Path):
    checkpoint = _checkpoint_path(artifact_root, dataset, seed)
    args = parse_args(
        [
            dataset,
            "--seed",
            str(seed),
            "--test",
            "--checkpoint",
            str(checkpoint),
            "--cache_dir",
            str(artifact_root / "cache"),
            "--checkpoint_dir",
            str(artifact_root / "checkpoints"),
            "--data_dir",
            str(artifact_root / "data" / "DARPA"),
            "--outputs_dir",
            str(run_output_dir),
        ]
    )
    return parse_config(args)


def _compute_attack_coverage(
    nodes: np.ndarray, predictions: np.ndarray, node_to_attack_ids: dict[int, set[str]]
) -> float:
    if not node_to_attack_ids:
        return 0.0

    total_attacks = set()
    for attack_ids in node_to_attack_ids.values():
        total_attacks.update(attack_ids)

    if not total_attacks:
        return 0.0

    detected_attacks = set()
    for node, pred in zip(nodes.tolist(), predictions.tolist()):  # noqa: B905
        if pred != 1:
            continue
        detected_attacks.update(node_to_attack_ids.get(int(node), set()))

    return len(detected_attacks) / len(total_attacks)


def _threshold_grid(
    base_threshold: float, calibration_scores: np.ndarray, multipliers: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    if np.isfinite(base_threshold) and base_threshold > 0:
        thresholds = np.unique(base_threshold * multipliers)
        return thresholds, thresholds / base_threshold

    finite_scores = calibration_scores[np.isfinite(calibration_scores)]
    if finite_scores.size == 0:
        thresholds = np.array([np.inf], dtype=np.float64)
        relative = np.array([1.0], dtype=np.float64)
        return thresholds, relative

    quantile_levels = np.linspace(0.5, 1.0, len(multipliers))
    thresholds = np.unique(np.quantile(finite_scores, quantile_levels))
    base = float(np.max(finite_scores))
    relative = np.divide(
        thresholds,
        base if base != 0 else 1.0,
        out=np.ones_like(thresholds, dtype=np.float64),
        where=np.isfinite(thresholds),
    )
    return thresholds.astype(np.float64), relative.astype(np.float64)


def _collect_scores(config):
    set_seed(config.seed)
    graphs, ground_truth = build_graphs(config)

    train_data = graphs["train"]
    val_data = graphs["val"]
    test_data = graphs["test"]
    calibration_data = val_data if val_data else train_data
    calibration_split_name = "Validation" if val_data else "Train"

    model = initialize_model(config, train_data, config.checkpoint)
    model.eval()

    excluded_node_ids = _get_metric_excluded_node_ids(config)
    contaminated_node_ids = _get_contaminated_node_ids(ground_truth)

    calibration_scores, calibration_labels, calibration_nodes, _ = (
        _collect_entity_scores(
            model,
            calibration_data,
            config,
            split_name=calibration_split_name,
            excluded_node_ids=excluded_node_ids,
        )
    )
    calibration_scores, calibration_labels, calibration_nodes = (
        _exclude_contaminated_entities(
            calibration_scores,
            calibration_labels,
            calibration_nodes,
            contaminated_node_ids,
            split_name=calibration_split_name,
        )
    )

    test_scores, test_labels, test_nodes, _ = _collect_entity_scores(
        model, test_data, config, split_name="Test", excluded_node_ids=excluded_node_ids
    )
    test_scores, test_labels, test_nodes = _exclude_contaminated_entities(
        test_scores, test_labels, test_nodes, contaminated_node_ids, split_name="Test"
    )

    base_threshold = select_max_benign_threshold(calibration_scores, calibration_labels)
    node_to_attack_ids = build_node_to_attack_mappings(ground_truth)
    adp = (
        compute_adp_score(
            test_scores.tolist(),
            test_nodes.tolist(),
            node_to_attack_ids,
            test_labels.tolist(),
            plot=False,
        )
        if node_to_attack_ids and test_scores.size > 0
        else 0.0
    )

    return {
        "calibration_scores": calibration_scores,
        "calibration_labels": calibration_labels,
        "calibration_nodes": calibration_nodes,
        "test_scores": test_scores,
        "test_labels": test_labels,
        "test_nodes": test_nodes,
        "base_threshold": float(base_threshold),
        "node_to_attack_ids": node_to_attack_ids,
        "adp": float(adp),
    }


def _run_single_sweep(
    dataset: str,
    seed: int,
    artifact_root: Path,
    multipliers: np.ndarray,
    output_dir: Path,
) -> tuple[list[SweepRow], SweepSummary]:
    run_output_dir = output_dir / "run_outputs" / f"{dataset.lower()}_seed{seed}"
    run_output_dir.mkdir(parents=True, exist_ok=True)

    config = _build_config(dataset, seed, artifact_root, run_output_dir)
    payload = _collect_scores(config)

    score_dir = output_dir / "score_cache"
    score_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        score_dir / f"{dataset.lower()}_seed{seed}_scores.npz",
        calibration_scores=payload["calibration_scores"],
        calibration_labels=payload["calibration_labels"],
        calibration_nodes=payload["calibration_nodes"],
        test_scores=payload["test_scores"],
        test_labels=payload["test_labels"],
        test_nodes=payload["test_nodes"],
        base_threshold=np.array([payload["base_threshold"]], dtype=np.float64),
        adp=np.array([payload["adp"]], dtype=np.float64),
    )

    thresholds, relative_thresholds = _threshold_grid(
        payload["base_threshold"], payload["calibration_scores"], multipliers
    )

    rows: list[SweepRow] = []
    for threshold, relative in zip(thresholds, relative_thresholds):  # noqa: B905
        metrics = compute_threshold_metrics(
            payload["test_scores"], payload["test_labels"], float(threshold)
        )
        predictions = threshold_predictions(payload["test_scores"], float(threshold))
        attack_coverage = _compute_attack_coverage(
            payload["test_nodes"], predictions, payload["node_to_attack_ids"]
        )
        rows.append(
            {
                "dataset": dataset,
                "seed": seed,
                "base_threshold": float(payload["base_threshold"]),
                "threshold": float(threshold),
                "threshold_relative": float(relative),
                "precision": float(metrics["precision"]),
                "recall": float(metrics["recall"]),
                "f1": float(metrics["f1"]),
                "mcc": float(metrics["mcc"]),
                "fpr": float(metrics["fpr"]),
                "fnr": float(metrics["fnr"]),
                "predicted_positive_entities": int(
                    metrics["predicted_positive_entities"]
                ),
                "predicted_positive_fraction": (
                    float(metrics["predicted_positive_entities"])
                    / float(metrics["entity_count"])
                    if metrics["entity_count"] > 0
                    else 0.0
                ),
                "attack_coverage": float(attack_coverage),
                "ap": float(metrics["ap"]),
                "adp": float(payload["adp"]),
            }
        )

    per_seed_dir = output_dir / "per_seed"
    per_seed_dir.mkdir(parents=True, exist_ok=True)
    per_seed_path = per_seed_dir / f"{dataset.lower()}_seed{seed}_threshold_sweep.csv"
    _write_csv(per_seed_path, rows)

    summary: SweepSummary = {
        "dataset": dataset,
        "seed": seed,
        "base_threshold": float(payload["base_threshold"]),
        "adp": float(payload["adp"]),
        "excluded_node_count": float(len(get_excluded_node_ids(config))),
    }
    return rows, summary


def _mean_and_std(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return 0.0, 0.0
    return float(arr.mean()), float(arr.std(ddof=0))


def _write_csv(path: Path, rows: list[SweepRow | SweepSummary | AggregateRow]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _build_aggregate_rows(rows: list[SweepRow]) -> list[AggregateRow]:
    grouped: dict[tuple[str, float], list[SweepRow]] = {}
    for row in rows:
        dataset = str(row["dataset"])
        threshold_relative = _canonical_relative_threshold(
            float(row["threshold_relative"])
        )
        grouped.setdefault((dataset, threshold_relative), []).append(row)

    aggregate_rows: list[AggregateRow] = []
    for (dataset, threshold_relative), group_rows in sorted(grouped.items()):
        aggregate_row: AggregateRow = {
            "dataset": dataset,
            "threshold_relative": threshold_relative,
        }
        for metric in (
            "precision",
            "recall",
            "f1",
            "mcc",
            "attack_coverage",
            "fpr",
            "predicted_positive_fraction",
            "adp",
        ):
            mean, std = _mean_and_std([float(row[metric]) for row in group_rows])
            aggregate_row[f"{metric}_mean"] = mean
            aggregate_row[f"{metric}_std"] = std
        aggregate_rows.append(aggregate_row)
    return aggregate_rows


def _read_csv_rows(path: Path) -> list[SweepRow]:
    rows: list[SweepRow] = []
    with path.open("r", newline="", encoding="utf-8") as fh:
        for raw_row in csv.DictReader(fh):
            row: SweepRow = {}
            for key, value in raw_row.items():
                if key is None:
                    continue
                row[key] = "" if value is None else value
            rows.append(row)
    return rows


def _load_existing_per_seed_rows(per_seed_dir: Path) -> list[SweepRow]:
    rows: list[SweepRow] = []
    if not per_seed_dir.is_dir():
        return rows
    for path in sorted(per_seed_dir.glob("*_threshold_sweep.csv")):
        rows.extend(_read_csv_rows(path))
    return rows


def _plot_aggregate(
    rows: list[SweepRow], datasets: list[str], output_dir: Path
) -> None:
    aggregate_rows = _build_aggregate_rows(rows)
    _write_csv(output_dir / "threshold_sensitivity_aggregate.csv", aggregate_rows)

    fig, axes = plt.subplots(2, 2, figsize=(17.6, 12.2), sharex=True, sharey=True)
    axes = axes.flatten()

    for idx, dataset in enumerate(datasets):
        ax = axes[idx]
        subset = [row for row in aggregate_rows if row["dataset"] == dataset]
        if not subset:
            ax.set_visible(False)
            continue

        x = np.asarray(
            [float(row["threshold_relative"]) for row in subset], dtype=np.float64
        )
        for key, label, color, linestyle in PLOT_METRICS:
            y = np.asarray(
                [float(row[f"{key}_mean"]) for row in subset], dtype=np.float64
            )
            ax.plot(x, y, label=label, color=color, linestyle=linestyle, linewidth=2.8)

        ax.axvline(1.0, color="#666666", linestyle=":", linewidth=1.5)
        ax.set_title(DATASET_TITLES.get(dataset, dataset))
        ax.set_xscale("log")
        ax.set_xlim(0.25, 4.0)
        ax.set_xticks(PLOT_X_TICKS)
        ax.set_xticklabels(PLOT_X_TICKLABELS)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.25)

        adp = float(subset[0]["adp_mean"])
        ax.text(
            0.97,
            0.95,
            f"ADP={adp:.3f}",
            transform=ax.transAxes,
            fontsize=21,
            color="#333333",
            ha="right",
            va="top",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.7},
        )

    for ax in axes[len(datasets) :]:
        ax.set_visible(False)

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

    png_path = output_dir / "theseus_threshold_sensitivity_e3.png"
    pdf_path = output_dir / "theseus_threshold_sensitivity_e3.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline threshold sweep for Theseus using frozen checkpoints and cache."
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        required=True,
        help="Root of the frozen reproducibility tree (cache/, checkpoints/, data/).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/threshold_sensitivity"),
        help="Directory for CSVs and figures.",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default=",".join(DEFAULT_DATASETS),
        help="Comma-separated dataset list.",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=",".join(str(seed) for seed in DEFAULT_SEEDS),
        help="Comma-separated seed list.",
    )
    args = parser.parse_args()

    artifact_root = args.artifact_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = _parse_csv_list(args.datasets)
    seeds = [int(seed) for seed in _parse_csv_list(args.seeds)]
    multipliers = np.array(DEFAULT_MULTIPLIERS, dtype=np.float64)

    all_rows: list[SweepRow] = []
    summaries: list[SweepSummary] = []
    per_seed_dir = output_dir / "per_seed"

    for dataset in datasets:
        for seed in seeds:
            existing_csv = (
                per_seed_dir / f"{dataset.lower()}_seed{seed}_threshold_sweep.csv"
            )
            if existing_csv.exists():
                log(f"Reusing existing threshold sweep: {dataset} seed={seed}")
                cached_rows = _read_csv_rows(existing_csv)
                all_rows.extend(cached_rows)
                if cached_rows:
                    first = cached_rows[0]
                    summaries.append(
                        {
                            "dataset": first["dataset"],
                            "seed": int(first["seed"]),
                            "base_threshold": float(first["base_threshold"]),
                            "adp": float(first["adp"]),
                            "excluded_node_count": "",
                        }
                    )
                continue
            log(f"Threshold sweep: {dataset} seed={seed}")
            rows, summary = _run_single_sweep(
                dataset=dataset,
                seed=seed,
                artifact_root=artifact_root,
                multipliers=multipliers,
                output_dir=output_dir,
            )
            all_rows.extend(rows)
            summaries.append(summary)

    full_rows = _load_existing_per_seed_rows(per_seed_dir)
    if not full_rows:
        full_rows = all_rows

    summary_by_key: dict[tuple[str, int], SweepSummary] = {}
    for summary in summaries:
        summary_by_key[(str(summary["dataset"]), int(summary["seed"]))] = summary
    for row in full_rows:
        key = (str(row["dataset"]), int(row["seed"]))
        if key not in summary_by_key:
            summary_by_key[key] = {
                "dataset": row["dataset"],
                "seed": int(row["seed"]),
                "base_threshold": float(row["base_threshold"]),
                "adp": float(row["adp"]),
                "excluded_node_count": "",
            }

    _write_csv(output_dir / "threshold_sensitivity_full.csv", full_rows)
    ordered_summaries = [
        summary_by_key[key]
        for key in sorted(summary_by_key, key=lambda item: (item[0], item[1]))
    ]
    _write_csv(output_dir / "threshold_sensitivity_run_summary.csv", ordered_summaries)
    _plot_aggregate(full_rows, datasets, output_dir)

    log(f"Wrote threshold sensitivity outputs to {output_dir}")


if __name__ == "__main__":
    main()
