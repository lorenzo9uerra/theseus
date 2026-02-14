#!/usr/bin/env python3
"""
Aggregate MAGIC evaluation logs (mean ± std across seeds).

Logs include two scopes:
- Causal: attack + contaminated are positive.
- Strict: attack-only is positive (contaminated masked).
"""

import argparse
import re
import statistics
from pathlib import Path
from typing import Optional

METRICS: tuple[str, ...] = ("PR-AUC", "F1", "MCC", "FPR", "ADP")
SCOPES: tuple[tuple[str, str], ...] = (("causal", "Causal"), ("strict", "Strict"))

DATASET_ALIASES = {
    "cadets": "CADETS_E3",
    "fivedirections": "FIVEDIRECTIONS_E3",
    "theia": "THEIA_E3",
    "trace": "TRACE_E3",
}

MAGIC_LOG_RE = re.compile(r"^(?P<dataset>[a-z]+)_seed(?P<seed>\d+)\.log$")


def _default_log_dir() -> Path:
    magic_root = Path(__file__).resolve().parents[1]
    return magic_root / "results"


def _std(vals: list[float]) -> float:
    return statistics.stdev(vals) if len(vals) > 1 else 0.0


def _print_group(system: str, dataset: str, parsed: int, found: int, values) -> None:
    line = "-" * 88
    print(line)
    print(f"SYSTEM: {system.upper()} | DATASET: {dataset}")
    print(f"Files: {parsed}/{found}")
    print(line)
    print("METRIC  | SCOPE  |       MEAN |        STD |   N")
    print("--------+--------+------------+------------+----")

    for metric in METRICS:
        for scope_key, scope_name in SCOPES:
            vals = values[scope_key].get(metric, [])
            if not vals:
                print(
                    f"{metric:<7} | {scope_name:<6} | {'NA':>10} | {'NA':>10} | {0:>3}"
                )
                continue
            mean_val = statistics.mean(vals)
            std_val = _std(vals)
            print(
                f"{metric:<7} | {scope_name:<6} | {mean_val:>10.5f} | {std_val:>10.5f} | {len(vals):>3}"
            )
    print()


def parse_file(path: Path) -> Optional[dict[str, dict[str, float]]]:  # noqa: UP045, Magic uses a different environment
    metrics: dict[str, dict[str, float]] = {"causal": {}, "strict": {}}

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return None

    in_summary = False
    saw_header = False
    for line in lines:
        s = line.strip()
        if s == "SUMMARY":
            in_summary = True
            continue
        if not in_summary:
            continue
        if s.startswith("=") or not s:
            continue
        if s.startswith("Metric") and "Causal Scope" in s and "Strict" in s:
            saw_header = True
            continue
        if not saw_header:
            continue

        parts = re.split(r"\s{2,}", s)
        if len(parts) < 3:
            continue

        name = parts[0].strip()
        name = "ADP" if name == "ADP Score" else name
        if name not in METRICS:
            continue

        try:
            metrics["causal"][name] = float(parts[1])
            metrics["strict"][name] = float(parts[2])
        except ValueError:
            continue

    if not metrics["causal"] and not metrics["strict"]:
        return None
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate MAGIC evaluation logs (mean ± std across seeds)."
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
    print("Scopes: Causal (attack+contaminated), Strict (attack-only)")
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
        values: dict[str, dict[str, list[float]]] = {"causal": {}, "strict": {}}
        parsed = 0
        for path in ds_files:
            data = parse_file(path)
            if data is None:
                continue
            parsed += 1
            for scope_key, _ in SCOPES:
                for metric, val in data[scope_key].items():
                    values[scope_key].setdefault(metric, []).append(val)

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
