#!/usr/bin/env python3
"""Aggregate allowlist diagnostic CSV outputs (mean ± std across matching rows)."""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

METRICS: tuple[str, ...] = ("Precision", "Recall", "F1", "MCC", "FPR")
DEFAULT_PATTERNS = ("*allowlist_diagnostic*.csv", "*binary_allowlist*.csv")

CSV_KEY_TO_METRIC = {
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1",
    "mcc": "MCC",
    "fpr": "FPR",
}


def _default_outputs_dir() -> Path:
    project_root = Path(__file__).resolve().parents[1]
    return project_root / "outputs"


def _std(vals: list[float]) -> float:
    return statistics.stdev(vals) if len(vals) > 1 else 0.0


def _collect_csv_paths(outputs_dir: Path, patterns: list[str]) -> list[Path]:
    return sorted({path for pattern in patterns for path in outputs_dir.glob(pattern)})


def _print_group(
    dataset: str, parsed: int, found: int, values: dict[str, list[float]]
) -> None:
    line = "-" * 88
    metric_width = max(len("METRIC"), max(len(metric) for metric in METRICS))
    print(line)
    print(f"SYSTEM: ALLOWLIST | DATASET: {dataset}")
    print(f"Rows: {parsed}/{found}")
    print(line)
    print(f"{'METRIC':<{metric_width}} |       MEAN |        STD |   N")
    print(f"{'-' * metric_width}+------------+------------+----")

    for metric in METRICS:
        vals = values.get(metric, [])
        if not vals:
            print(f"{metric:<{metric_width}} | {'NA':>10} | {'NA':>10} | {0:>3}")
            continue
        mean_val = statistics.mean(vals)
        std_val = _std(vals)
        print(
            f"{metric:<{metric_width}} | {mean_val:>10.5f} | {std_val:>10.5f} | {len(vals):>3}"
        )
    print()


def parse_csv(path: Path) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    try:
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                dataset = str(row.get("dataset", "")).strip()
                if not dataset:
                    continue
                parsed: dict[str, float | str] = {"dataset": dataset}
                for csv_key, metric_name in CSV_KEY_TO_METRIC.items():
                    value = row.get(csv_key)
                    if value in (None, ""):
                        continue
                    try:
                        parsed[metric_name] = float(value)
                    except ValueError:
                        continue
                rows.append(parsed)
    except Exception:
        return []
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate allowlist diagnostic CSV outputs (mean ± std across rows)."
    )
    parser.add_argument(
        "--outputs_dir",
        type=Path,
        default=_default_outputs_dir(),
        help="Directory containing allowlist diagnostic CSV files (default: outputs).",
    )
    parser.add_argument(
        "--pattern",
        action="append",
        default=None,
        help="Glob pattern for allowlist CSV files. Can be repeated.",
    )
    args = parser.parse_args()

    outputs_dir = args.outputs_dir.resolve()
    if not outputs_dir.is_dir():
        raise SystemExit(f"ERROR: outputs_dir not found: {outputs_dir}")

    patterns = args.pattern or list(DEFAULT_PATTERNS)
    files = _collect_csv_paths(outputs_dir, patterns)
    line = "=" * 88
    print(line)
    print("Aggregate Results (Allowlist)")
    print(f"Outputs dir: {outputs_dir}")
    print(f"Files: {len(files)}")
    print(f"Metrics: {', '.join(METRICS)}")
    print(line)
    print()

    dataset_to_rows: dict[str, list[dict[str, float | str]]] = {}
    parsed_files = 0
    for path in files:
        rows = parse_csv(path)
        if not rows:
            continue
        parsed_files += 1
        for row in rows:
            dataset = str(row["dataset"])
            dataset_to_rows.setdefault(dataset, []).append(row)

    for dataset in sorted(dataset_to_rows):
        rows = dataset_to_rows[dataset]
        values: dict[str, list[float]] = {}
        for row in rows:
            for metric in METRICS:
                value = row.get(metric)
                if isinstance(value, float):
                    values.setdefault(metric, []).append(value)
        _print_group(dataset=dataset, parsed=len(rows), found=len(rows), values=values)

    skipped = len(files) - parsed_files
    if skipped:
        print(f"WARN: skipped {skipped} file(s) with no parseable rows.")


if __name__ == "__main__":
    main()
