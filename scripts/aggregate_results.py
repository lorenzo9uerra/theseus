#!/usr/bin/env python3
"""
Aggregate Theseus evaluation logs (mean ± std across seeds).

Logs include two scopes:
- Causal: attack + contaminated are positive.
- Strict: attack-only is positive (contaminated masked).
"""

import argparse
import re
import statistics
from pathlib import Path

METRICS: tuple[str, ...] = ("PR-AUC", "F1", "MCC", "FPR", "ADP")
SCOPES: tuple[tuple[str, str], ...] = (("causal", "Causal"), ("strict", "Strict"))

LOG_NAME_RE = re.compile(r"^theseus_(?P<dataset>[A-Z]+_E\d+)_seed(?P<seed>\d+)\.log$")

KEY_VALUE_RE = re.compile(
    r"\b(?P<key>[A-Za-z0-9_]+)\b[:\s]+"
    r"(?P<value>[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)"
)

KEY_PRIORITIES: dict[tuple[str, str], tuple[str, ...]] = {
    ("causal", "PR-AUC"): ("final_test_ap", "final_ap"),
    ("causal", "F1"): ("final_test_binary_f1", "final_fscore"),
    ("causal", "MCC"): ("final_test_mcc", "final_mcc"),
    ("causal", "FPR"): ("final_test_fpr", "final_fpr"),
    ("causal", "ADP"): ("test_adp_causal", "final_adp_score"),
    ("strict", "PR-AUC"): ("final_strict_test_ap", "final_strict_ap"),
    ("strict", "F1"): ("final_strict_test_binary_f1", "final_strict_fscore"),
    ("strict", "MCC"): ("final_strict_test_mcc", "final_strict_mcc"),
    ("strict", "FPR"): ("final_strict_test_fpr", "final_strict_fpr"),
    ("strict", "ADP"): ("test_adp_strict", "final_strict_adp_score"),
}

ALLOWED_KEYS = {key for keys in KEY_PRIORITIES.values() for key in keys}


def _default_log_dir() -> Path:
    project_root = Path(__file__).resolve().parents[1]
    return project_root / "results"


def _std(vals: list[float]) -> float:
    return statistics.stdev(vals) if len(vals) > 1 else 0.0


def _print_group(dataset: str, parsed: int, found: int, values) -> None:
    line = "-" * 88
    print(line)
    print(f"SYSTEM: THESEUS | DATASET: {dataset}")
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


def parse_file(path: Path) -> dict[str, dict[str, float]] | None:
    metrics: dict[str, dict[str, float]] = {"causal": {}, "strict": {}}
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

    for scope_key, _ in SCOPES:
        for metric in METRICS:
            for key in KEY_PRIORITIES.get((scope_key, metric), ()):
                if key in raw:
                    metrics[scope_key][metric] = raw[key]
                    break

    if not metrics["causal"] and not metrics["strict"]:
        return None
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate Theseus evaluation logs (mean ± std across seeds)."
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
    print("Scopes: Causal (attack+contaminated), Strict (attack-only)")
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

        _print_group(dataset=dataset, parsed=parsed, found=len(paths), values=values)

    if skipped:
        print(f"WARN: skipped {skipped} file(s) with unexpected name.")


if __name__ == "__main__":
    main()
