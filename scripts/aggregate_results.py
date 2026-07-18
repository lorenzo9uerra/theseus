#!/usr/bin/env python3
"""Aggregate Theseus strict attack-chain evaluation logs (mean ± std across seeds)."""

import argparse
import re
import statistics
from pathlib import Path

METRICS: tuple[str, ...] = ("AP", "AUROC", "Precision", "F1", "MCC", "FPR", "ADP")

LOG_NAME_RE = re.compile(r"^theseus_(?P<dataset>[A-Z]+_E\d+)_seed(?P<seed>\d+)\.log$")

KEY_VALUE_RE = re.compile(
    r"\b(?P<key>[A-Za-z0-9_]+)\b[:\s]+"
    r"(?P<value>[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)"
)

KEY_PRIORITIES: dict[str, tuple[str, ...]] = {
    "AP": ("final_test_ap", "final_strict_test_ap", "final_ap", "final_strict_ap"),
    "AUROC": (
        "final_test_auroc",
        "final_strict_test_auroc",
        "final_auroc",
        "final_strict_auroc",
        "final_test_auc",
        "final_auc",
    ),
    "Precision": (
        "final_test_precision",
        "final_strict_test_precision",
        "final_precision",
        "final_strict_precision",
    ),
    "F1": (
        "final_test_binary_f1",
        "final_strict_test_binary_f1",
        "final_fscore",
        "final_strict_fscore",
    ),
    "MCC": ("final_test_mcc", "final_strict_test_mcc", "final_mcc", "final_strict_mcc"),
    "FPR": ("final_test_fpr", "final_strict_test_fpr", "final_fpr", "final_strict_fpr"),
    "ADP": ("test_adp", "test_adp_strict", "final_adp_score", "final_strict_adp_score"),
}

ALLOWED_KEYS = {key for keys in KEY_PRIORITIES.values() for key in keys}


def _default_log_dir() -> Path:
    project_root = Path(__file__).resolve().parents[1]
    return project_root / "results"


def _std(vals: list[float]) -> float:
    return statistics.stdev(vals) if len(vals) > 1 else 0.0


def _print_group(dataset: str, parsed: int, found: int, values) -> None:
    line = "-" * 88
    metric_width = max(len("METRIC"), max(len(metric) for metric in METRICS))
    print(line)
    print(f"SYSTEM: THESEUS | DATASET: {dataset}")
    print(f"Files: {parsed}/{found}")
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


def parse_file(path: Path) -> dict[str, float] | None:
    metrics: dict[str, float] = {}
    raw: dict[str, float] = {}

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return None

    for line in lines:
        for match in KEY_VALUE_RE.finditer(line):
            key = match.group("key")
            if key not in ALLOWED_KEYS:
                continue
            try:
                raw[key] = float(match.group("value"))
            except ValueError:
                continue

    for metric in METRICS:
        for key in KEY_PRIORITIES.get(metric, ()):
            if key in raw:
                metrics[metric] = raw[key]
                break

    if not metrics:
        return None
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate Theseus strict evaluation logs (mean ± std across seeds)."
    )
    parser.add_argument(
        "--log_dir",
        type=Path,
        default=_default_log_dir(),
        help="Directory containing Theseus evaluation logs (default: results).",
    )
    args = parser.parse_args()

    log_dir = args.log_dir.resolve()
    if not log_dir.is_dir():
        raise SystemExit(f"ERROR: log_dir not found: {log_dir}")

    line = "=" * 88
    files = sorted(log_dir.glob("*.log"))
    print(line)
    print("Aggregate Results (Theseus)")
    print(f"Log dir: {log_dir}")
    print(f"Files: {len(files)}")
    print(f"Metrics: {', '.join(METRICS)}")
    print(line)
    print()

    dataset_to_files: dict[str, list[Path]] = {}
    skipped = 0
    for path in files:
        match = LOG_NAME_RE.match(path.name)
        if not match:
            skipped += 1
            continue
        dataset = match.group("dataset")
        dataset_to_files.setdefault(dataset, []).append(path)

    dataset_order = ["CADETS_E3", "FIVEDIRECTIONS_E3", "THEIA_E3", "TRACE_E3"]
    dataset_order_map = {name: idx for idx, name in enumerate(dataset_order)}
    datasets = sorted(
        dataset_to_files.keys(),
        key=lambda d: (0, dataset_order_map[d]) if d in dataset_order_map else (1, d),
    )

    for dataset in datasets:
        paths = dataset_to_files[dataset]
        values: dict[str, list[float]] = {}
        parsed = 0
        for path in paths:
            data = parse_file(path)
            if data is None:
                continue
            parsed += 1
            for metric, val in data.items():
                values.setdefault(metric, []).append(val)

        _print_group(dataset=dataset, parsed=parsed, found=len(paths), values=values)

    if skipped:
        print(f"WARN: skipped {skipped} file(s) with unexpected name.")


if __name__ == "__main__":
    main()
