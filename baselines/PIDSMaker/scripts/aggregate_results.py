#!/usr/bin/env python3
"""
Aggregate PIDSMaker evaluation logs (mean ± std across seeds).

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

RAW_KEY_TO_METRIC = {
    "ap": "PR-AUC",
    "fscore": "F1",
    "mcc": "MCC",
    "fpr": "FPR",
    "adp_score": "ADP",
}

PIDSM_LOG_RE = re.compile(
    r"^(?P<system>[A-Za-z0-9]+)_(?P<dataset>[A-Z]+_E\d+)_seed(?P<seed>\d+)\.log$"
)

FINAL_METRIC_RE = re.compile(
    r"final_(?P<strict>strict_)?(?P<key>ap|fscore|mcc|fpr|adp_score)\b[:\s]+"
    r"(?P<value>[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)"
)


def _default_log_dir() -> Path:
    pids_root = Path(__file__).resolve().parents[1]
    return pids_root / "results" / "paper"


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
                print(f"{metric:<7} | {scope_name:<6} | {'NA':>10} | {'NA':>10} | {0:>3}")
                continue
            mean_val = statistics.mean(vals)
            std_val = _std(vals)
            print(
                f"{metric:<7} | {scope_name:<6} | {mean_val:>10.5f} | {std_val:>10.5f} | {len(vals):>3}"
            )
    print()


def parse_file(path: Path) -> Optional[dict[str, dict[str, float]]]:
    metrics: dict[str, dict[str, float]] = {"causal": {}, "strict": {}}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return None

    for line in lines:
        match = FINAL_METRIC_RE.search(line)
        if not match:
            continue
        scope_key = "strict" if match.group("strict") else "causal"
        metric = RAW_KEY_TO_METRIC[match.group("key")]
        try:
            metrics[scope_key][metric] = float(match.group("value"))
        except ValueError:
            continue

    if not metrics["causal"] and not metrics["strict"]:
        return None
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate PIDSMaker evaluation logs (mean ± std across seeds)."
    )
    parser.add_argument(
        "--log_dir",
        type=Path,
        default=_default_log_dir(),
        help="Directory containing PIDSMaker evaluation logs (default: baselines/PIDSMaker/results/paper).",
    )
    args = parser.parse_args()

    log_dir = args.log_dir.resolve()
    if not log_dir.is_dir():
        raise SystemExit(f"ERROR: log_dir not found: {log_dir}")

    line = "=" * 88
    files = sorted(log_dir.glob("*.log"))
    print(line)
    print("Aggregate Results (PIDSMaker)")
    print(f"Log dir: {log_dir}")
    print(f"Files: {len(files)}")
    print("Scopes: Causal (attack+contaminated), Strict (attack-only)")
    print(f"Metrics: {', '.join(METRICS)}")
    print(line)
    print()

    groups: dict[tuple[str, str], list[Path]] = {}
    skipped = 0
    for path in files:
        match = PIDSM_LOG_RE.match(path.name)
        if not match:
            skipped += 1
            continue
        system = match.group("system").lower()
        dataset = match.group("dataset")
        groups.setdefault((system, dataset), []).append(path)

    system_order = ["orthrus", "velox"]
    dataset_order = ["CADETS_E3", "FIVEDIRECTIONS_E3", "THEIA_E3", "TRACE_E3"]
    system_order_map = {name: idx for idx, name in enumerate(system_order)}
    dataset_order_map = {name: idx for idx, name in enumerate(dataset_order)}

    def _sort_key(item: tuple[str, str]):
        system, dataset = item
        sys_key = (0, system_order_map[system]) if system in system_order_map else (1, system)
        ds_key = (0, dataset_order_map[dataset]) if dataset in dataset_order_map else (1, dataset)
        return (sys_key, ds_key)

    for system, dataset in sorted(groups.keys(), key=_sort_key):
        paths = groups[(system, dataset)]
        values: dict[str, dict[str, list[float]]] = {"causal": {}, "strict": {}}
        parsed = 0
        for path in paths:
            data = parse_file(path)
            if data is None:
                continue
            parsed += 1
            for scope_key, _ in SCOPES:
                for metric, val in data[scope_key].items():
                    values[scope_key].setdefault(metric, []).append(val)

        _print_group(
            system=system,
            dataset=dataset,
            parsed=parsed,
            found=len(paths),
            values=values,
        )

    if skipped:
        print(f"WARN: skipped {skipped} file(s) with unexpected name.")


if __name__ == "__main__":
    main()
