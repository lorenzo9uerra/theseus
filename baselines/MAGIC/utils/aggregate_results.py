#!/usr/bin/env python3
"""Aggregate strict MAGIC evaluation logs (mean ± std across seeds)."""

import argparse
import re
import statistics
from contextlib import suppress
from pathlib import Path
from typing import Optional

METRICS: tuple[str, ...] = ("AP", "Precision", "F1", "MCC", "FPR", "ADP")

DATASET_ALIASES = {
    "cadets": "CADETS_E3",
    "fivedirections": "FIVEDIRECTIONS_E3",
    "theia": "THEIA_E3",
    "trace": "TRACE_E3",
}

MAGIC_LOG_RE = re.compile(r"^(?P<dataset>[a-z]+)_seed(?P<seed>\d+)\.log$")
SINGLE_VALUE_RE = re.compile(
    r"^(?P<key>AP|PR-AUC|PRECISION|F1|FPR|MCC|ADP)\b[:\s]+"
    r"(?P<value>[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)$"
)
STRICT_F1_LINE_RE = re.compile(
    r"^F1:\s*(?P<f1>[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)"
    r",\s*Prec:\s*(?P<precision>[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)"
    r",\s*Rec:\s*(?P<recall>[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)$"
)


def _normalize_metric_name(name: str) -> str:
    normalized = name.strip()
    if normalized == "PR-AUC":
        return "AP"
    if normalized == "PRECISION":
        return "Precision"
    return normalized


def _default_log_dir() -> Path:
    magic_root = Path(__file__).resolve().parents[1]
    return magic_root / "results"


def _std(vals: list[float]) -> float:
    return statistics.stdev(vals) if len(vals) > 1 else 0.0


def _print_group(
    system: str, dataset: str, parsed: int, found: int, values: dict[str, list[float]]
) -> None:
    line = "-" * 88
    metric_width = max(len("METRIC"), max(len(metric) for metric in METRICS))
    print(line)
    print(f"SYSTEM: {system.upper()} | DATASET: {dataset}")
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


def parse_file(path: Path) -> Optional[dict[str, float]]:  # noqa: UP045, Magic uses a different environment
    metrics: dict[str, float] = {}

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return None

    for line in lines:
        match = SINGLE_VALUE_RE.match(line.strip())
        if match is None:
            continue
        name = _normalize_metric_name(match.group("key"))
        if name not in METRICS:
            continue
        try:
            metrics[name] = float(match.group("value"))
        except ValueError:
            continue

    in_summary = False
    saw_header = False
    in_strict_section = False
    for line in lines:
        s = line.strip()
        if "LEVEL 2: STRICT ATTACK CHAIN" in s:
            in_strict_section = True
            continue
        if "LEVEL 1: CAUSAL SCOPE" in s or s == "SUMMARY":
            in_strict_section = False
        if in_strict_section:
            strict_match = STRICT_F1_LINE_RE.match(s)
            if strict_match is not None:
                with suppress(ValueError):
                    metrics["Precision"] = float(strict_match.group("precision"))
        if s == "SUMMARY":
            in_summary = True
            continue
        if not in_summary:
            continue
        if s.startswith("=") or not s:
            continue
        if s.startswith("Metric"):
            saw_header = True
            continue
        if not saw_header:
            continue

        parts = re.split(r"\s{2,}", s)
        if len(parts) < 2:
            continue

        name = parts[0].strip()
        name = _normalize_metric_name(name)
        name = "ADP" if name == "ADP Score" else name
        if name not in METRICS:
            continue

        try:
            metrics[name] = float(parts[-1])
        except ValueError:
            continue

    if not metrics:
        return None
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate strict MAGIC evaluation logs (mean ± std across seeds)."
    )
    parser.add_argument(
        "--log_dir",
        type=Path,
        default=_default_log_dir(),
        help="Directory containing MAGIC evaluation logs (default: baselines/MAGIC/results).",
    )
    args = parser.parse_args()

    log_dir = args.log_dir.resolve()
    if not log_dir.is_dir():
        raise SystemExit(f"ERROR: log_dir not found: {log_dir}")

    line = "=" * 88
    files = sorted(log_dir.glob("*.log"))
    print(line)
    print("Aggregate Results (MAGIC)")
    print(f"Log dir: {log_dir}")
    print(f"Files: {len(files)}")
    print(f"Metrics: {', '.join(METRICS)}")
    print(line)
    print()

    dataset_to_files: dict[str, list[Path]] = {}
    skipped = 0
    for path in files:
        match = MAGIC_LOG_RE.match(path.name)
        if not match:
            skipped += 1
            continue
        raw_ds = match.group("dataset")
        dataset = DATASET_ALIASES.get(raw_ds, raw_ds.upper())
        dataset_to_files.setdefault(dataset, []).append(path)

    dataset_order = ["CADETS_E3", "FIVEDIRECTIONS_E3", "THEIA_E3", "TRACE_E3"]
    dataset_order_map = {name: idx for idx, name in enumerate(dataset_order)}
    datasets = sorted(
        dataset_to_files.keys(),
        key=lambda d: (0, dataset_order_map[d]) if d in dataset_order_map else (1, d),
    )

    for dataset in datasets:
        ds_files = dataset_to_files[dataset]
        values: dict[str, list[float]] = {}
        parsed = 0
        for path in ds_files:
            data = parse_file(path)
            if data is None:
                continue
            parsed += 1
            for metric, val in data.items():
                values.setdefault(metric, []).append(val)

        _print_group(
            system="magic",
            dataset=dataset,
            parsed=parsed,
            found=len(ds_files),
            values=values,
        )

    if skipped:
        print(f"WARN: skipped {skipped} file(s) with unexpected name.")


if __name__ == "__main__":
    main()
