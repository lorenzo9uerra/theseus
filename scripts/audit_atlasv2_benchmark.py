#!/usr/bin/env python3
"""Reproducible audit for ATLASv2 benchmark suitability."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.create_csv_files_atlasv2 import (  # noqa: E402
    _infer_attack_id_from_edr_file,
    _parse_cbc_timestamp_to_ns,
)
from utils.ground_truth import load_atlasv2_process_labels  # noqa: E402
from utils.utils import read_node_table  # noqa: E402

ATLAS_DATASETS = ["atlasv2_h1", "atlasv2_h2"]
DARPA_DATASETS = ["CADETS_E3", "THEIA_E3", "FIVEDIRECTIONS_E3", "TRACE_E3"]
TABLE_NAMES = [
    "event_table",
    "process_node_table",
    "file_node_table",
    "netflow_node_table",
]
ALLOWLIST_CSV_SUFFIXES = ("allowlist_diagnostic", "binary_allowlist")


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(str(value).strip())
    except Exception:
        return None


def _resolve_allowlist_csv_path(outputs_dir: Path, dataset_name: str) -> Path | None:
    for suffix in ALLOWLIST_CSV_SUFFIXES:
        csv_path = outputs_dir / f"{dataset_name}_{suffix}.csv"
        if csv_path.exists():
            return csv_path
    return None


@dataclass(frozen=True)
class AtlasLabelRow:
    attack_id: str
    host: str
    pid: int
    path: str
    label: str
    process_uuid: str = ""


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    try:
        string_value = str(value).strip()
        if string_value == "":
            return None
        if string_value.endswith(".0"):
            string_value = string_value[:-2]
        return int(string_value)
    except Exception:
        return None


def _normalize_windows_path(path: str) -> str:
    if not path:
        return ""
    normalized = str(path).strip().lower().replace("\\", "/")
    while "//" in normalized:
        normalized = normalized.replace("//", "/")
    return normalized


def _infer_host_from_device_name(device_name: str) -> str | None:
    device = (device_name or "").lower()
    if device.endswith("h1") or "-h1" in device:
        return "h1"
    if device.endswith("h2") or "-h2" in device:
        return "h2"
    return None


def _infer_host_from_dataset_name(dataset_name: str) -> str | None:
    name = (dataset_name or "").lower()
    if name.endswith("_h1") or name.endswith("-h1") or name.endswith("h1"):
        return "h1"
    if name.endswith("_h2") or name.endswith("-h2") or name.endswith("h2"):
        return "h2"
    return None


def _infer_host_from_attack_id(attack_id: str) -> str | None:
    attack_id = (attack_id or "").strip().lower()
    if not attack_id.startswith("atlasv2/"):
        return None
    host_and_scenario = attack_id.split("/", 1)[1]
    host = host_and_scenario.split("-", 1)[0]
    return host if host in {"h1", "h2"} else None


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open() as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping in {path}, got {type(data).__name__}")
    return data


def _find_table_path(dataset_dir: Path, table_name: str) -> Path | None:
    parquet_path = dataset_dir / f"{table_name}.parquet"
    if parquet_path.exists():
        return parquet_path
    csv_path = dataset_dir / f"{table_name}.csv"
    if csv_path.exists():
        return csv_path
    return None


def _scan_table(path: Path):
    if path.suffix == ".parquet":
        return pl.scan_parquet(path)
    if path.suffix == ".csv":
        return pl.scan_csv(path, infer_schema_length=1000)
    raise ValueError(f"Unsupported table format: {path}")


def _count_rows(path: Path) -> int:
    result = _scan_table(path).select(pl.len().alias("rows")).collect()
    return int(result["rows"][0])


def _ns_to_utc_day(timestamp_ns: int) -> str:
    return datetime.fromtimestamp(timestamp_ns / 1_000_000_000, tz=UTC).strftime(
        "%Y-%m-%d"
    )


def _ns_to_utc_iso(timestamp_ns: int) -> str:
    return (
        datetime.fromtimestamp(timestamp_ns / 1_000_000_000, tz=UTC)
        .isoformat()
        .replace("+00:00", "Z")
    )


def collect_dataset_table_counts(dataset_dir: Path) -> dict[str, int | None]:
    counts: dict[str, int | None] = {}
    for table_name in TABLE_NAMES:
        table_path = _find_table_path(dataset_dir, table_name)
        counts[table_name] = _count_rows(table_path) if table_path else None
    return counts


def collect_scale_report(
    atlas_root: Path,
    darpa_root: Path,
    atlas_datasets: list[str],
    darpa_datasets: list[str],
) -> dict[str, Any]:
    datasets: dict[str, dict[str, int | None]] = {}
    atlas_event_total = 0
    darpa_event_counts: dict[str, int] = {}

    for dataset_name in atlas_datasets:
        dataset_dir = atlas_root / dataset_name
        if not dataset_dir.exists():
            continue
        counts = collect_dataset_table_counts(dataset_dir)
        datasets[dataset_name] = counts
        atlas_event_total += counts.get("event_table") or 0

    for dataset_name in darpa_datasets:
        dataset_dir = darpa_root / dataset_name
        if not dataset_dir.exists():
            continue
        counts = collect_dataset_table_counts(dataset_dir)
        datasets[dataset_name] = counts
        event_count = counts.get("event_table")
        if event_count is not None:
            darpa_event_counts[dataset_name] = event_count

    comparison: dict[str, Any] = {}
    if atlas_event_total and darpa_event_counts:
        smallest_darpa_name, smallest_darpa_events = min(
            darpa_event_counts.items(), key=lambda item: item[1]
        )
        comparison = {
            "combined_atlas_events": atlas_event_total,
            "smallest_darpa_dataset": smallest_darpa_name,
            "smallest_darpa_events": smallest_darpa_events,
            "smallest_darpa_over_combined_atlas": round(
                smallest_darpa_events / atlas_event_total, 3
            ),
        }

    return {"datasets": datasets, "comparison": comparison}


def _format_event_day_label(year_month: str, day: int) -> str:
    return f"{year_month}-{int(day):02d}"


def collect_event_day_counts(dataset_dir: Path) -> dict[str, int]:
    table_path = _find_table_path(dataset_dir, "event_table")
    if table_path is None:
        return {}

    timestamp_col = pl.col("timestamp_rec").cast(pl.Int64, strict=False)
    frame = (
        _scan_table(table_path)
        .select(timestamp_col.alias("timestamp_rec"))
        .drop_nulls()
        .with_columns(
            pl.from_epoch(pl.col("timestamp_rec"), time_unit="ns")
            .dt.strftime("%Y-%m-%d")
            .alias("event_day")
        )
        .group_by("event_day")
        .len()
        .sort("event_day")
        .collect()
    )
    return {row["event_day"]: int(row["len"]) for row in frame.iter_rows(named=True)}


def collect_split_report(
    atlas_root: Path, atlas_config_path: Path, atlas_datasets: list[str]
) -> dict[str, Any]:
    config_data = _load_yaml(atlas_config_path)
    report: dict[str, Any] = {}

    for dataset_name in atlas_datasets:
        dataset_config = config_data.get(dataset_name)
        dataset_dir = atlas_root / dataset_name
        if not isinstance(dataset_config, dict) or not dataset_dir.exists():
            continue

        year_month = str(dataset_config["year_month"])
        train_days = [int(day) for day in dataset_config.get("train_days", [])]
        val_days = [int(day) for day in dataset_config.get("val_days", [])]
        test_days = [int(day) for day in dataset_config.get("test_days", [])]

        train_labels = [_format_event_day_label(year_month, day) for day in train_days]
        val_labels = [_format_event_day_label(year_month, day) for day in val_days]
        test_labels = [_format_event_day_label(year_month, day) for day in test_days]

        event_day_counts = collect_event_day_counts(dataset_dir)
        report[dataset_name] = {
            "train_days": train_days,
            "val_days": val_days,
            "test_days": test_days,
            "split_overlap": {
                "train_val": sorted(set(train_days) & set(val_days)),
                "train_test": sorted(set(train_days) & set(test_days)),
                "val_test": sorted(set(val_days) & set(test_days)),
            },
            "event_day_counts": event_day_counts,
            "events_per_split": {
                "train": sum(event_day_counts.get(label, 0) for label in train_labels),
                "val": sum(event_day_counts.get(label, 0) for label in val_labels),
                "test": sum(event_day_counts.get(label, 0) for label in test_labels),
            },
        }

    return report


def collect_semantic_signal_report(
    atlas_root: Path, atlas_datasets: list[str]
) -> dict[str, Any]:
    from scripts.analyze_semantic_signal_quality import analyze_dataset

    report: dict[str, Any] = {}
    for dataset_name in atlas_datasets:
        try:
            stats = analyze_dataset(dataset_name, str(atlas_root), verbose=False)
        except Exception:
            continue
        report[dataset_name] = {
            "total_nodes": stats.total_nodes,
            "completeness": stats.overall_completeness,
            "entropy": stats.overall_entropy,
        }
    return report


def collect_allowlist_report(
    outputs_dir: Path, atlas_datasets: list[str]
) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for dataset_name in atlas_datasets:
        csv_path = _resolve_allowlist_csv_path(outputs_dir, dataset_name)
        if csv_path is None:
            continue

        with csv_path.open(newline="") as file:
            reader = csv.DictReader(file)
            row = next(reader, None)
        if not row:
            continue

        report[dataset_name] = {
            "precision": _safe_float(row.get("precision")),
            "recall": _safe_float(row.get("recall")),
            "f1": _safe_float(row.get("f1")),
            "mcc": _safe_float(row.get("mcc")),
            "fpr": _safe_float(row.get("fpr")),
            "n_test_process": _safe_int(row.get("n_test_process")),
            "n_attack": _safe_int(row.get("n_attack")),
            "n_contaminated": _safe_int(row.get("n_contaminated")),
            "n_excluded": _safe_int(row.get("n_excluded")),
        }
    return report


def load_atlas_label_rows(labels_source: Path) -> list[AtlasLabelRow]:
    if labels_source.is_dir():
        return [
            AtlasLabelRow(
                attack_id=label.attack_id,
                host=_infer_host_from_attack_id(label.attack_id) or "",
                pid=label.pid if label.pid is not None else -1,
                path=label.path,
                label=label.label,
                process_uuid=label.process_uuid,
            )
            for label in load_atlasv2_process_labels(labels_source)
            if _infer_host_from_attack_id(label.attack_id) is not None
        ]

    rows: list[AtlasLabelRow] = []
    with labels_source.open(newline="") as file:
        reader = csv.DictReader(file, skipinitialspace=True)
        for raw_row in reader:
            if not raw_row:
                continue

            row = {
                (key or "").strip().lower(): (value or "").strip()
                for key, value in raw_row.items()
            }
            attack_id = row.get("attack", "").lower()
            if attack_id and not attack_id.startswith("atlasv2/"):
                attack_id = f"atlasv2/{attack_id}"
            host = _infer_host_from_attack_id(attack_id)
            label = row.get("label", "").lower()
            pid = _safe_int(row.get("process_id"))
            path = _normalize_windows_path(row.get("process_name", ""))
            process_uuid = row.get("process_uuid", "").lower()

            if not attack_id or host is None or label == "":
                continue
            if not process_uuid and (pid is None or not path):
                continue

            rows.append(
                AtlasLabelRow(
                    attack_id=attack_id,
                    host=host,
                    pid=pid if pid is not None else -1,
                    path=path,
                    label=label,
                    process_uuid=process_uuid,
                )
            )

    return rows


def collect_label_inventory(label_rows: list[AtlasLabelRow]) -> dict[str, Any]:
    per_host: dict[str, dict[str, int]] = {}
    for host in sorted({row.host for row in label_rows}):
        host_rows = [row for row in label_rows if row.host == host]
        raw_counts = Counter(row.label for row in host_rows)
        unique_rows = {
            (row.attack_id, row.process_uuid, row.pid, row.path, row.label)
            for row in host_rows
        }
        unique_label_counts = Counter(label for _, _, _, _, label in unique_rows)
        per_host[host] = {
            "raw_rows": len(host_rows),
            "unique_rows": len(unique_rows),
            "duplicates": len(host_rows) - len(unique_rows),
            "raw_attack": raw_counts.get("attack", 0),
            "raw_contaminated": raw_counts.get("contaminated", 0),
            "raw_benign": raw_counts.get("benign", 0),
            "unique_attack": unique_label_counts.get("attack", 0),
            "unique_contaminated": unique_label_counts.get("contaminated", 0),
            "unique_benign": unique_label_counts.get("benign", 0),
            "attack_chains": len({row.attack_id for row in host_rows}),
        }

    return {"per_host": per_host, "raw_rows": len(label_rows)}


def _build_process_lookup(
    dataset_dir: Path,
) -> tuple[
    dict[str, set[int]],
    dict[tuple[str, int, str], set[int]],
    dict[tuple[str, int, str], set[int]],
]:
    process_df = read_node_table(
        str(dataset_dir),
        "process_node_table",
        columns=["index_id", "node_uuid", "path", "pid", "attack"],
    )
    if process_df is None:
        raise ValueError(f"process_node_table not found under {dataset_dir}")

    by_process_uuid: dict[str, set[int]] = defaultdict(set)
    by_attack_pid_path: dict[tuple[str, int, str], set[int]] = defaultdict(set)
    by_host_pid_path: dict[tuple[str, int, str], set[int]] = defaultdict(set)

    for row in process_df.iter_rows(named=True):
        index_id = _safe_int(row.get("index_id"))
        pid = _safe_int(row.get("pid"))
        path = _normalize_windows_path(row.get("path", ""))
        attack_id = (row.get("attack") or "").strip()
        node_uuid = (row.get("node_uuid") or "").strip()
        device_name = node_uuid.split("|", 1)[0] if "|" in node_uuid else ""
        host = _infer_host_from_device_name(device_name)

        if index_id is None:
            continue

        process_uuid = node_uuid.split("|")[-1].strip().lower()
        if process_uuid:
            by_process_uuid[process_uuid].add(index_id)
        if attack_id and pid is not None and path:
            by_attack_pid_path[(attack_id, pid, path)].add(index_id)
        if host and pid is not None and path:
            by_host_pid_path[(host, pid, path)].add(index_id)

    return by_process_uuid, by_attack_pid_path, by_host_pid_path


def _classify_label_match(
    process_uuid: str,
    attack_id: str,
    host: str,
    pid: int,
    path: str,
    by_attack_pid_path: dict[tuple[str, int, str], set[int]],
    by_host_pid_path: dict[tuple[str, int, str], set[int]],
    by_process_uuid: dict[str, set[int]],
) -> tuple[str, set[int]]:
    if process_uuid:
        uuid_matches = by_process_uuid.get(process_uuid, set())
        if uuid_matches:
            if len(uuid_matches) == 1:
                return "uuid_unique", uuid_matches
            return "uuid_ambiguous", uuid_matches
        return "unmatched", set()

    exact_matches = by_attack_pid_path.get((attack_id, pid, path), set())
    if exact_matches:
        if len(exact_matches) == 1:
            return "exact_unique", exact_matches
        return "exact_ambiguous", exact_matches

    host_matches = by_host_pid_path.get((host, pid, path), set())
    if host_matches:
        if len(host_matches) == 1:
            return "host_fallback_unique", host_matches
        return "host_fallback_ambiguous", host_matches

    return "unmatched", set()


MATCH_CATEGORIES = (
    "uuid_unique",
    "uuid_ambiguous",
    "exact_unique",
    "exact_ambiguous",
    "host_fallback_unique",
    "host_fallback_ambiguous",
    "unmatched",
)


def _format_match_counts(counts: Counter) -> dict[str, int]:
    return {category: counts.get(category, 0) for category in MATCH_CATEGORIES}


def collect_label_alignment_report(
    atlas_root: Path, atlas_datasets: list[str], label_rows: list[AtlasLabelRow]
) -> dict[str, Any]:
    report: dict[str, Any] = {}

    for dataset_name in atlas_datasets:
        dataset_dir = atlas_root / dataset_name
        dataset_host = _infer_host_from_dataset_name(dataset_name)
        if dataset_host is None or not dataset_dir.exists():
            continue

        by_process_uuid, by_attack_pid_path, by_host_pid_path = _build_process_lookup(
            dataset_dir
        )
        host_rows = [row for row in label_rows if row.host == dataset_host]
        attack_rows = [row for row in host_rows if row.label == "attack"]
        contaminated_rows = [row for row in host_rows if row.label == "contaminated"]

        raw_categories = Counter()
        raw_node_ids: set[int] = set()
        for row in attack_rows:
            category, node_ids = _classify_label_match(
                row.process_uuid,
                row.attack_id,
                row.host,
                row.pid,
                row.path,
                by_attack_pid_path,
                by_host_pid_path,
                by_process_uuid,
            )
            raw_categories[category] += 1
            raw_node_ids.update(node_ids)

        unique_attack_rows = {
            (row.process_uuid, row.attack_id, row.host, row.pid, row.path)
            for row in attack_rows
        }
        unique_categories = Counter()
        unique_node_ids: set[int] = set()
        for process_uuid, attack_id, host, pid, path in sorted(unique_attack_rows):
            category, node_ids = _classify_label_match(
                process_uuid,
                attack_id,
                host,
                pid,
                path,
                by_attack_pid_path,
                by_host_pid_path,
                by_process_uuid,
            )
            unique_categories[category] += 1
            unique_node_ids.update(node_ids)

        contaminated_categories = Counter()
        contaminated_node_ids: set[int] = set()
        for row in contaminated_rows:
            category, node_ids = _classify_label_match(
                row.process_uuid,
                row.attack_id,
                row.host,
                row.pid,
                row.path,
                by_attack_pid_path,
                by_host_pid_path,
                by_process_uuid,
            )
            contaminated_categories[category] += 1
            contaminated_node_ids.update(node_ids)

        report[dataset_name] = {
            "raw_attack_rows": len(attack_rows),
            "unique_attack_rows": len(unique_attack_rows),
            "duplicate_attack_rows": len(attack_rows) - len(unique_attack_rows),
            "raw_contaminated_rows": len(contaminated_rows),
            "contaminated_match_counts": _format_match_counts(contaminated_categories),
            "matched_node_ids_from_contaminated_rows": len(contaminated_node_ids),
            "raw_match_counts": _format_match_counts(raw_categories),
            "unique_match_counts": _format_match_counts(unique_categories),
            "matched_node_ids_from_raw_attack_rows": len(raw_node_ids),
            "matched_node_ids_from_unique_attack_rows": len(unique_node_ids),
        }

    return report


def _iter_attack_edr_files(raw_dir: Path):
    if not raw_dir.exists():
        return
    for path in sorted(raw_dir.rglob("*.jsonl")):
        parts = {part.lower() for part in path.parts}
        if "attack" not in parts or "cbc-edr" not in parts:
            continue
        attack_id = _infer_attack_id_from_edr_file(path)
        if not attack_id or attack_id.endswith("-benign"):
            continue
        yield path


def _scan_raw_window(path: Path) -> tuple[int | None, int | None, int]:
    min_ts: int | None = None
    max_ts: int | None = None
    count = 0

    with path.open(errors="ignore") as file:
        for line in file:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            timestamp_ns = _parse_cbc_timestamp_to_ns(
                obj.get("device_timestamp") or obj.get("backend_timestamp")
            )
            if timestamp_ns is None:
                continue

            count += 1
            min_ts = timestamp_ns if min_ts is None else min(min_ts, timestamp_ns)
            max_ts = timestamp_ns if max_ts is None else max(max_ts, timestamp_ns)

    return min_ts, max_ts, count


def collect_raw_scenario_report(raw_dir: Path) -> dict[str, Any]:
    scenarios: list[dict[str, Any]] = []
    spillover = []

    for path in _iter_attack_edr_files(raw_dir):
        attack_id = _infer_attack_id_from_edr_file(path)
        start_ns, end_ns, event_count = _scan_raw_window(path)
        if attack_id is None or start_ns is None or end_ns is None:
            continue

        start_day = _ns_to_utc_day(start_ns)
        end_day = _ns_to_utc_day(end_ns)
        scenario = {
            "attack_id": attack_id,
            "path": str(path),
            "event_count": event_count,
            "start_time_utc": _ns_to_utc_iso(start_ns),
            "end_time_utc": _ns_to_utc_iso(end_ns),
            "start_day_utc": start_day,
            "end_day_utc": end_day,
            "spans_multiple_days": start_day != end_day,
            "duration_minutes": round((end_ns - start_ns) / 60_000_000_000, 2),
        }
        scenarios.append(scenario)
        if scenario["spans_multiple_days"]:
            spillover.append(attack_id)

    return {
        "scenario_count": len(scenarios),
        "spillover_scenarios": spillover,
        "scenarios": scenarios,
    }


def _extract_process_paths(obj: dict[str, Any]) -> set[str]:
    paths = set()
    for field in ("process_path", "childproc_name"):
        normalized = _normalize_windows_path(obj.get(field, ""))
        if normalized:
            paths.add(normalized)
    return paths


def _collect_process_paths_in_windows(
    path: Path, prefix_minutes: int
) -> tuple[set[str], set[str]]:
    min_ts, max_ts, _ = _scan_raw_window(path)
    if min_ts is None or max_ts is None:
        return set(), set()

    window_ns = prefix_minutes * 60 * 1_000_000_000
    prefix_cutoff = min_ts + window_ns
    suffix_cutoff = max_ts - window_ns

    prefix_paths: set[str] = set()
    suffix_paths: set[str] = set()

    with path.open(errors="ignore") as file:
        for line in file:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            timestamp_ns = _parse_cbc_timestamp_to_ns(
                obj.get("device_timestamp") or obj.get("backend_timestamp")
            )
            if timestamp_ns is None:
                continue

            paths = _extract_process_paths(obj)
            if not paths:
                continue

            if timestamp_ns <= prefix_cutoff:
                prefix_paths.update(paths)
            if timestamp_ns >= suffix_cutoff:
                suffix_paths.update(paths)

    return prefix_paths, suffix_paths


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 0.0
    return len(left & right) / len(left | right)


def _pairwise_jaccard_report(path_sets: dict[str, set[str]]) -> dict[str, Any]:
    scenario_names = sorted(path_sets)
    pairs = []
    scores = []

    for left_idx, left_name in enumerate(scenario_names):
        for right_name in scenario_names[left_idx + 1 :]:
            score = _jaccard(path_sets[left_name], path_sets[right_name])
            scores.append(score)
            pairs.append(
                {"left": left_name, "right": right_name, "jaccard": round(score, 4)}
            )

    mean_score = round(sum(scores) / len(scores), 4) if scores else None
    return {"mean_jaccard": mean_score, "pairs": pairs}


def collect_startup_footprint_report(
    raw_dir: Path, prefix_minutes: int, min_common_scenarios: int
) -> dict[str, Any]:
    prefix_sets: dict[str, set[str]] = {}
    suffix_sets: dict[str, set[str]] = {}

    for path in _iter_attack_edr_files(raw_dir):
        attack_id = _infer_attack_id_from_edr_file(path)
        if attack_id is None:
            continue
        prefix_paths, suffix_paths = _collect_process_paths_in_windows(
            path, prefix_minutes
        )
        prefix_sets[attack_id] = prefix_paths
        suffix_sets[attack_id] = suffix_paths

    prefix_frequencies = Counter()
    for path_set in prefix_sets.values():
        prefix_frequencies.update(path_set)

    common_prefix_paths = [
        {"path": path, "scenario_count": count}
        for path, count in sorted(
            prefix_frequencies.items(), key=lambda item: (-item[1], item[0])
        )
        if count >= min_common_scenarios
    ]

    return {
        "scenario_count": len(prefix_sets),
        "prefix_minutes": prefix_minutes,
        "prefix_pairwise_jaccard": _pairwise_jaccard_report(prefix_sets),
        "suffix_pairwise_jaccard": _pairwise_jaccard_report(suffix_sets),
        "common_prefix_process_paths": common_prefix_paths,
    }


def build_report(
    atlas_root: Path,
    darpa_root: Path,
    atlas_raw_dir: Path | None,
    atlas_labels: Path | None,
    atlas_config_path: Path,
    atlas_datasets: list[str],
    darpa_datasets: list[str],
    prefix_minutes: int,
    min_common_scenarios: int,
    allowlist_dir: Path | None = None,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "inputs": {
            "atlas_root": str(atlas_root),
            "darpa_root": str(darpa_root),
            "atlas_raw_dir": str(atlas_raw_dir) if atlas_raw_dir else None,
            "atlas_labels": str(atlas_labels) if atlas_labels else None,
            "atlas_config": str(atlas_config_path),
        },
        "scale": collect_scale_report(
            atlas_root, darpa_root, atlas_datasets, darpa_datasets
        ),
        "split_independence": collect_split_report(
            atlas_root, atlas_config_path, atlas_datasets
        ),
        "semantic_signal": collect_semantic_signal_report(atlas_root, atlas_datasets),
    }

    if atlas_labels and atlas_labels.exists():
        label_rows = load_atlas_label_rows(atlas_labels)
        report["label_inventory"] = collect_label_inventory(label_rows)
        report["label_alignment"] = collect_label_alignment_report(
            atlas_root, atlas_datasets, label_rows
        )
    else:
        report["label_inventory"] = None
        report["label_alignment"] = None

    if atlas_raw_dir and atlas_raw_dir.exists():
        report["raw_scenarios"] = collect_raw_scenario_report(atlas_raw_dir)
        report["startup_footprint"] = collect_startup_footprint_report(
            atlas_raw_dir, prefix_minutes, min_common_scenarios
        )
    else:
        report["raw_scenarios"] = None
        report["startup_footprint"] = None

    if allowlist_dir and allowlist_dir.exists():
        report["allowlist"] = collect_allowlist_report(allowlist_dir, atlas_datasets)
    else:
        report["allowlist"] = None

    return report


def format_markdown_summary(report: dict[str, Any]) -> str:
    lines = ["# ATLASv2 Benchmark Audit", ""]

    scale = report.get("scale", {})
    comparison = scale.get("comparison") or {}
    lines.append("## Scale")
    for dataset_name, counts in (scale.get("datasets") or {}).items():
        lines.append(
            "- "
            f"{dataset_name}: "
            f"events={counts.get('event_table')}, "
            f"processes={counts.get('process_node_table')}, "
            f"files={counts.get('file_node_table')}, "
            f"netflows={counts.get('netflow_node_table')}"
        )
    if comparison:
        lines.append(
            "- "
            f"Combined ATLASv2 events={comparison['combined_atlas_events']}, "
            f"smallest DARPA E3 dataset={comparison['smallest_darpa_dataset']} "
            f"with {comparison['smallest_darpa_events']} events "
            f"({comparison['smallest_darpa_over_combined_atlas']}x larger)."
        )
    lines.append("")

    lines.append("## Semantic Signal Quality Inputs")
    for dataset_name, signal in (report.get("semantic_signal") or {}).items():
        lines.append(
            "- "
            f"{dataset_name}: completeness={signal['completeness']:.2f}%, "
            f"entropy={signal['entropy']:.2f}"
        )
    lines.append("")

    allowlist = report.get("allowlist") or {}
    if allowlist:
        lines.append("## Allowlist Diagnostic")
        for dataset_name, metrics in allowlist.items():
            lines.append(
                "- "
                f"{dataset_name}: "
                f"test_proc={metrics['n_test_process']}, "
                f"attack={metrics['n_attack']}, "
                f"contaminated={metrics['n_contaminated']}, "
                f"f1={metrics['f1']:.3f}, "
                f"mcc={metrics['mcc']:.3f}, "
                f"fpr={metrics['fpr']:.4f}"
            )
        lines.append("")

    lines.append("## Split Independence")
    for dataset_name, split_report in (report.get("split_independence") or {}).items():
        lines.append(
            "- "
            f"{dataset_name}: train={split_report['train_days']}, "
            f"val={split_report['val_days']}, test={split_report['test_days']}, "
            f"overlap={split_report['split_overlap']}, "
            f"event_day_counts={split_report['event_day_counts']}"
        )
    lines.append("")

    label_inventory = report.get("label_inventory") or {}
    per_host_inventory = label_inventory.get("per_host") or {}
    lines.append("## Label Inventory")
    for host, inventory in per_host_inventory.items():
        lines.append(
            "- "
            f"{host}: raw_attack={inventory['raw_attack']}, "
            f"raw_contaminated={inventory['raw_contaminated']}, "
            f"raw_benign={inventory['raw_benign']}, "
            f"duplicates={inventory['duplicates']}"
        )
    for dataset_name, alignment in (report.get("label_alignment") or {}).items():
        lines.append(
            "- "
            f"{dataset_name}: unique_match_counts={alignment['unique_match_counts']}, "
            f"contaminated_match_counts={alignment['contaminated_match_counts']}"
        )
    lines.append("")

    raw_scenarios = report.get("raw_scenarios")
    if raw_scenarios:
        lines.append("## Raw Scenario Timing")
        lines.append(
            "- "
            f"scenario_count={raw_scenarios['scenario_count']}, "
            f"spillover_scenarios={raw_scenarios['spillover_scenarios']}"
        )
        lines.append("")

    startup = report.get("startup_footprint")
    if startup:
        lines.append("## Early-Scenario Footprint")
        lines.append(
            "- "
            f"prefix_minutes={startup['prefix_minutes']}, "
            f"mean_prefix_jaccard={startup['prefix_pairwise_jaccard']['mean_jaccard']}, "
            f"mean_suffix_jaccard={startup['suffix_pairwise_jaccard']['mean_jaccard']}"
        )
        common_paths = startup.get("common_prefix_process_paths") or []
        if common_paths:
            top_paths = ", ".join(
                f"{item['path']} ({item['scenario_count']})"
                for item in common_paths[:10]
            )
            lines.append(f"- common_prefix_process_paths={top_paths}")
        lines.append("")

    lines.append("## Paper-Safe Reading")
    lines.append(
        "- ATLASv2 can still be useful for supplementary experiments, but this audit "
        "supports excluding it from a primary benchmark suite when the paper's main "
        "claims target large provenance datasets with established labels and cleaner "
        "split assumptions."
    )

    return "\n".join(lines).strip() + "\n"


def _format_day_span(days: list[int]) -> str:
    if not days:
        return "none"
    if len(days) > 1 and days == list(range(days[0], days[-1] + 1)):
        return f"{days[0]}--{days[-1]}"
    return ",".join(str(day) for day in days)


def _format_k_count(count: int | None) -> str:
    if count is None:
        return "---"
    return f"{round(count / 1000):,}k"


def _positive_row_match(alignment: dict[str, Any]) -> tuple[int, int]:
    raw_attack_rows = int(alignment.get("raw_attack_rows", 0))
    raw_contaminated_rows = int(alignment.get("raw_contaminated_rows", 0))
    raw_unmatched = int((alignment.get("raw_match_counts") or {}).get("unmatched", 0))
    contaminated_unmatched = int(
        (alignment.get("contaminated_match_counts") or {}).get("unmatched", 0)
    )
    matched = (raw_attack_rows - raw_unmatched) + (
        raw_contaminated_rows - contaminated_unmatched
    )
    unmatched = raw_unmatched + contaminated_unmatched
    return matched, unmatched


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit ATLASv2 for benchmark suitability relative to DARPA TC E3."
    )
    parser.add_argument(
        "--atlas_root",
        type=Path,
        default=PROJECT_ROOT / "data" / "ATLASV2",
        help="Root directory containing atlasv2_h1/ and atlasv2_h2/ processed tables.",
    )
    parser.add_argument(
        "--darpa_root",
        type=Path,
        default=PROJECT_ROOT / "data" / "DARPA",
        help="Root directory containing DARPA E3 processed datasets.",
    )
    parser.add_argument(
        "--atlas_raw_dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "atlasv2" / "data",
        help="Raw ATLASv2 directory used for scenario timing and startup-footprint checks.",
    )
    parser.add_argument(
        "--atlas_labels",
        type=Path,
        default=PROJECT_ROOT / "ground_truth" / "reapr-ground-truth" / "atlasv2",
        help="Directory containing revised ATLASv2 labels or the legacy label CSV.",
    )
    parser.add_argument(
        "--atlas_config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "datasets" / "atlasv2.yml",
        help="ATLASv2 dataset config used to define train/val/test day splits.",
    )
    parser.add_argument(
        "--atlas_datasets",
        nargs="+",
        default=ATLAS_DATASETS,
        help="Processed ATLASv2 dataset names to audit.",
    )
    parser.add_argument(
        "--darpa_datasets",
        nargs="+",
        default=DARPA_DATASETS,
        help="Processed DARPA datasets to compare against.",
    )
    parser.add_argument(
        "--prefix_minutes",
        type=int,
        default=2,
        help="Window size for early/late raw-log process-path overlap.",
    )
    parser.add_argument(
        "--min_common_scenarios",
        type=int,
        default=4,
        help="Minimum number of scenarios required to report a common prefix path.",
    )
    parser.add_argument(
        "--allowlist_dir",
        type=Path,
        default=PROJECT_ROOT / "outputs",
        help="Directory containing ATLASv2 allowlist CSV outputs.",
    )
    parser.add_argument(
        "--json_out",
        type=Path,
        default=None,
        help="Optional path for the machine-readable JSON report.",
    )
    parser.add_argument(
        "--markdown_out",
        type=Path,
        default=None,
        help="Optional path for the human-readable Markdown summary.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = build_report(
        atlas_root=args.atlas_root,
        darpa_root=args.darpa_root,
        atlas_raw_dir=args.atlas_raw_dir,
        atlas_labels=args.atlas_labels,
        atlas_config_path=args.atlas_config,
        atlas_datasets=list(args.atlas_datasets),
        darpa_datasets=list(args.darpa_datasets),
        prefix_minutes=args.prefix_minutes,
        min_common_scenarios=args.min_common_scenarios,
        allowlist_dir=args.allowlist_dir,
    )

    markdown = format_markdown_summary(report)
    print(markdown, end="")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2) + "\n")

    if args.markdown_out:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(markdown)


if __name__ == "__main__":
    main()
