#!/usr/bin/env python3
"""Aggregate the ATLASv2 secondary-benchmark table from logs and CSVs."""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from collections.abc import Mapping
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent

ATLAS_DATASETS = ("atlasv2_h1", "atlasv2_h2")
ALLOWLIST_CSV_SUFFIXES = ("allowlist_diagnostic", "binary_allowlist")
DEFAULT_THESEUS_GLOBS = {
    "atlasv2_h1": "mseed_atlasv2_h1_mseed_theseus_*_*.out",
    "atlasv2_h2": "mseed_atlasv2_h2_mseed_theseus_*_*.out",
}
DEFAULT_VELOX_GLOBS = {
    "atlasv2_h1": "mseed_atlasv2_h1_mseed_velox_*_*.out",
    "atlasv2_h2": "mseed_atlasv2_h2_mseed_velox_*_*.out",
}
DEFAULT_THESEUS_RESULT_GLOBS = {
    "atlasv2_h1": "theseus_atlasv2_h1_seed*.log",
    "atlasv2_h2": "theseus_atlasv2_h2_seed*.log",
}
DEFAULT_VELOX_RESULT_GLOBS = {
    "atlasv2_h1": "velox_atlasv2_h1_seed*.log",
    "atlasv2_h2": "velox_atlasv2_h2_seed*.log",
}

FLOAT_RE = r"([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)"
MetricKey = str | tuple[str, ...]
THESEUS_METRIC_KEYS: dict[str, MetricKey] = {
    "ap": "final_test_ap",
    "auroc": "final_test_auroc",
    "precision": "final_test_precision",
    "f1": "final_test_binary_f1",
    "mcc": "final_test_mcc",
    "adp": "test_adp",
    "fpr": "final_test_fpr",
}
VELOX_METRIC_KEYS: dict[str, MetricKey] = {
    "ap": ("final_strict_ap", "final_ap"),
    "auroc": ("final_strict_auc", "final_auc"),
    "precision": ("final_strict_precision", "final_precision"),
    "f1": ("final_strict_fscore", "final_fscore"),
    "mcc": ("final_strict_mcc", "final_mcc"),
    "adp": ("final_strict_adp_score", "final_adp_score"),
    "fpr": ("final_strict_fpr", "final_fpr"),
}


def _sample_std(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _extract_last_float(text: str, key: str | tuple[str, ...]) -> float:
    keys = (key,) if isinstance(key, str) else key
    for candidate in keys:
        matches = re.findall(rf"{re.escape(candidate)}:\s*{FLOAT_RE}", text)
        if matches:
            return float(matches[-1])
    joined = ", ".join(keys)
    raise ValueError(f"Missing metric '{joined}'")


def _extract_last_token(text: str, key: str) -> str | None:
    matches = re.findall(rf"{re.escape(key)}:\s*(\S+)", text)
    return matches[-1] if matches else None


def _split_job_ids(job_id: str | None) -> list[str]:
    if not job_id:
        return []
    return [part.strip() for part in str(job_id).split(",") if part.strip()]


def _build_globs(
    *, base: dict[str, str], h1_job_id: str | None, h2_job_id: str | None, prefix: str
) -> dict[str, str | list[str]]:
    globs: dict[str, str | list[str]] = dict(base)
    overrides = {
        "atlasv2_h1": _split_job_ids(h1_job_id),
        "atlasv2_h2": _split_job_ids(h2_job_id),
    }
    for dataset, job_ids in overrides.items():
        if not job_ids:
            continue
        globs[dataset] = [
            pattern
            for job_id in job_ids
            for pattern in (
                f"mseed_{dataset}_{prefix}_{job_id}_*.out",
                f"mseed_{dataset}_holdout_{prefix}_{job_id}_*.out",
            )
        ]
    return globs


def _resolve_log_paths(slurm_dir: Path, pattern: str | list[str]) -> list[Path]:
    patterns = [pattern] if isinstance(pattern, str) else pattern

    paths: list[Path] = []
    for single_pattern in patterns:
        paths.extend(sorted(slurm_dir.glob(single_pattern)))

    paths = sorted(set(paths))
    if not paths:
        raise FileNotFoundError(f"No logs found for pattern: {pattern}")
    return paths


def _aggregate_logs(
    paths: list[Path], key_map: Mapping[str, MetricKey]
) -> dict[str, Any]:
    values: dict[str, list[float]] = {metric: [] for metric in key_map}
    threshold_sources: list[str] = []
    seeds: list[int] = []
    used_files: list[str] = []
    skipped_files: list[str] = []

    for path in paths:
        text = _read_text(path)
        extracted_metrics: dict[str, float] = {}
        try:
            for metric, key in key_map.items():
                try:
                    extracted_metrics[metric] = _extract_last_float(text, key)
                except ValueError:
                    if metric != "auroc":
                        raise
        except ValueError:
            skipped_files.append(str(path))
            continue

        for metric, value in extracted_metrics.items():
            values[metric].append(value)

        used_files.append(str(path))
        threshold_source = _extract_last_token(text, "threshold_source")
        if threshold_source is not None:
            threshold_sources.append(threshold_source)
        seed = _extract_last_token(text, "Running with seed")
        if seed and seed.isdigit():
            seeds.append(int(seed))

    if not used_files:
        raise ValueError("No complete logs found after filtering incomplete runs.")

    summary = {
        "count": len(used_files),
        "files": used_files,
        "skipped_files": skipped_files,
        "seeds": seeds,
        "threshold_sources": threshold_sources,
        "metrics": {},
    }
    for metric, vals in values.items():
        if not vals:
            continue
        summary["metrics"][metric] = {
            "values": vals,
            "mean": statistics.mean(vals),
            "std": _sample_std(vals),
        }
    return summary


def _aggregate_datasets(
    log_dir: Path, globs: dict[str, str | list[str]], key_map: Mapping[str, MetricKey]
) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for dataset, pattern in globs.items():
        paths = _resolve_log_paths(log_dir, pattern)
        report[dataset] = _aggregate_logs(paths, key_map)
    return report


def aggregate_theseus(
    slurm_dir: Path, globs: dict[str, str | list[str]]
) -> dict[str, Any]:
    return _aggregate_datasets(slurm_dir, globs, THESEUS_METRIC_KEYS)


def aggregate_velox(
    slurm_dir: Path, globs: dict[str, str | list[str]]
) -> dict[str, Any]:
    return _aggregate_datasets(slurm_dir, globs, VELOX_METRIC_KEYS)


def aggregate_theseus_results(
    results_dir: Path, globs: dict[str, str | list[str]]
) -> dict[str, Any]:
    return _aggregate_datasets(results_dir, globs, THESEUS_METRIC_KEYS)


def aggregate_velox_results(
    results_dir: Path, globs: dict[str, str | list[str]]
) -> dict[str, Any]:
    return _aggregate_datasets(results_dir, globs, VELOX_METRIC_KEYS)


def _resolve_allowlist_csv_path(outputs_dir: Path, dataset: str) -> Path:
    candidates = [
        outputs_dir / f"{dataset}_{suffix}.csv" for suffix in ALLOWLIST_CSV_SUFFIXES
    ]
    for csv_path in candidates:
        if csv_path.exists():
            return csv_path

    candidate_list = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Missing allowlist CSV. Tried: {candidate_list}")


def aggregate_allowlist(outputs_dir: Path) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for dataset in ATLAS_DATASETS:
        csv_path = _resolve_allowlist_csv_path(outputs_dir, dataset)
        with csv_path.open(newline="") as file:
            row = next(csv.DictReader(file), None)
        if row is None:
            raise ValueError(f"Empty allowlist CSV: {csv_path}")
        report[dataset] = {
            "csv": str(csv_path),
            "precision": float(row["precision"]),
            "recall": float(row["recall"]),
            "f1": float(row["f1"]),
            "mcc": float(row["mcc"]),
            "fpr": float(row["fpr"]),
            "n_test_process": int(row["n_test_process"]),
            "n_attack": int(row["n_attack"]),
            "n_contaminated": int(row["n_contaminated"]),
        }
    return report


def _fmt_mean_std(entry: dict[str, Any]) -> str:
    return f"{entry['mean']:.4f} +- {entry['std']:.4f}"


def _fmt_metric(summary: dict[str, Any], metric: str) -> str:
    entry = summary["metrics"].get(metric)
    return _fmt_mean_std(entry) if entry is not None else "N/A"


def _format_markdown(
    *,
    theseus: dict[str, Any],
    velox: dict[str, Any],
    allowlist: dict[str, Any],
    log_globs: dict[str, dict[str, str | list[str]]],
    source_mode: str,
    log_root: Path,
) -> str:
    def _format_glob_lines(value: str | list[str]) -> list[str]:
        values = [value] if isinstance(value, str) else value
        return [f"  - `{log_root / entry}`" for entry in values]

    lines = [
        "# ATLASv2 Secondary Benchmark Summary",
        "",
        "This summary collects the strict-only ATLASv2 results used in the paper table.",
        "",
        "Protocol:",
        "- `atlasv2_h1` and `atlasv2_h2`",
        "- train: July 15-18, 2022",
        "- val: none",
        "- test: July 19-20, 2022",
        "- contaminated nodes are scored but excluded from metric accounting",
        "",
        "## Main Table",
        "",
        "| Host | Method | AP | AUROC | Precision | F1 | MCC | ADP | FPR |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for dataset in ATLAS_DATASETS:
        lines.append(
            "| "
            + f"`{dataset}` | `Theseus` | "
            + " | ".join(
                _fmt_metric(theseus[dataset], metric)
                for metric in ("ap", "auroc", "precision", "f1", "mcc", "adp", "fpr")
            )
            + " |"
        )
        lines.append(
            "| "
            + f"`{dataset}` | `Velox` | "
            + " | ".join(
                _fmt_metric(velox[dataset], metric)
                for metric in ("ap", "auroc", "precision", "f1", "mcc", "adp", "fpr")
            )
            + " |"
        )
        lines.append(
            "| "
            + f"`{dataset}` | `Allowlist` | `N/A` | `N/A` | "
            + f"`{allowlist[dataset]['precision']:.4f}` | "
            + f"`{allowlist[dataset]['f1']:.4f}` | "
            + f"`{allowlist[dataset]['mcc']:.4f}` | `N/A` | "
            + f"`{allowlist[dataset]['fpr']:.4f}` |"
        )

    lines.extend(
        [
            "",
            "## Caveats",
            "",
            "- ATLASv2 is a secondary benchmark. Each host effectively contains a single engagement, so these results should not be interpreted like the temporally separated E3 benchmarks.",
            "- Theseus and Velox should be interpreted under the current ATLASv2 fallback regime: no attacked validation stage is available, so calibration falls back to training-derived thresholds.",
            "- Use the threshold-source fields in the JSON output to confirm the exact calibration path recorded by each rerun.",
            "- `Allowlist` is deterministic, so no seed variance is reported.",
            "",
            "## Source Pointers",
            "",
            f"- Source mode: `{source_mode}`",
            "- `Theseus` h1 logs:",
            *_format_glob_lines(log_globs["theseus"]["atlasv2_h1"]),
            "- `Theseus` h2 logs:",
            *_format_glob_lines(log_globs["theseus"]["atlasv2_h2"]),
            "- `Velox` h1 logs:",
            *_format_glob_lines(log_globs["velox"]["atlasv2_h1"]),
            "- `Velox` h2 logs:",
            *_format_glob_lines(log_globs["velox"]["atlasv2_h2"]),
            "- Allowlist outputs:",
            *(f"  - `{allowlist[dataset]['csv']}`" for dataset in ATLAS_DATASETS),
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate the ATLASv2 secondary-benchmark table from logs and CSVs."
    )
    parser.add_argument(
        "--slurm-dir",
        type=Path,
        default=PROJECT_ROOT / "slurm",
        help="Directory containing rerun logs when --results-dir is omitted.",
    )
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs",
        help="Default directory for ATLASv2 allowlist diagnostic CSVs.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Directory containing packaged canonical result logs.",
    )
    parser.add_argument(
        "--allowlist-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing ATLASv2 allowlist diagnostic CSVs. "
            "Legacy *_binary_allowlist.csv names are accepted. "
            "Defaults to --outputs-dir."
        ),
    )
    parser.add_argument(
        "--theseus-h1-job-id",
        type=str,
        default=None,
        help="Optional slurm array job id for atlasv2_h1 Theseus reruns.",
    )
    parser.add_argument(
        "--theseus-h2-job-id",
        type=str,
        default=None,
        help="Optional slurm array job id for atlasv2_h2 Theseus reruns.",
    )
    parser.add_argument(
        "--velox-h1-job-id",
        type=str,
        default=None,
        help="Optional slurm array job id for atlasv2_h1 Velox reruns.",
    )
    parser.add_argument(
        "--velox-h2-job-id",
        type=str,
        default=None,
        help="Optional slurm array job id for atlasv2_h2 Velox reruns.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "atlasv2_secondary_benchmark_summary.json",
        help="Machine-readable JSON output path.",
    )
    parser.add_argument(
        "--markdown-out",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "atlasv2_secondary_benchmark_summary.md",
        help="Human-readable Markdown output path.",
    )
    args = parser.parse_args()

    theseus_globs = _build_globs(
        base=DEFAULT_THESEUS_GLOBS,
        h1_job_id=args.theseus_h1_job_id,
        h2_job_id=args.theseus_h2_job_id,
        prefix="mseed_theseus",
    )
    velox_globs = _build_globs(
        base=DEFAULT_VELOX_GLOBS,
        h1_job_id=args.velox_h1_job_id,
        h2_job_id=args.velox_h2_job_id,
        prefix="mseed_velox",
    )

    if args.results_dir is not None:
        source_mode = "results"
        source_root = args.results_dir
        theseus_globs = dict[str, str | list[str]](DEFAULT_THESEUS_RESULT_GLOBS)
        velox_globs = dict[str, str | list[str]](DEFAULT_VELOX_RESULT_GLOBS)
        theseus = aggregate_theseus_results(args.results_dir, theseus_globs)
        velox = aggregate_velox_results(args.results_dir, velox_globs)
    else:
        source_mode = "slurm"
        source_root = args.slurm_dir
        theseus = aggregate_theseus(args.slurm_dir, theseus_globs)
        velox = aggregate_velox(args.slurm_dir, velox_globs)

    allowlist_dir = args.allowlist_dir or args.outputs_dir
    allowlist = aggregate_allowlist(allowlist_dir)

    log_globs: dict[str, dict[str, str | list[str]]] = {
        "theseus": theseus_globs,
        "velox": velox_globs,
    }

    summary = {
        "theseus": theseus,
        "velox": velox,
        "allowlist": allowlist,
        "log_globs": log_globs,
        "source_mode": source_mode,
        "source_root": str(source_root),
    }

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(summary, indent=2) + "\n")
    args.markdown_out.write_text(
        _format_markdown(
            theseus=theseus,
            velox=velox,
            allowlist=allowlist,
            log_globs=log_globs,
            source_mode=source_mode,
            log_root=source_root,
        )
        + "\n"
    )
    print(f"Wrote {args.json_out}")
    print(f"Wrote {args.markdown_out}")


if __name__ == "__main__":
    main()
