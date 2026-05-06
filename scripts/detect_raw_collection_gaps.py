#!/usr/bin/env python3
"""Audit raw DARPA TC E3 event streams for collection gaps and ordering issues."""

from __future__ import annotations

import argparse
import gzip
import json
import tarfile
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import IO, SupportsInt, cast

EVENT_KEY = "com.bbn.tc.schema.avro.cdm18.Event"
EVENT_MARKER = f'{{"datum":{{"{EVENT_KEY}"'
DEFAULT_DATASETS = ("cadets", "theia", "fivedirections", "trace")

type SourceHandle = IO[str] | IO[bytes]
type SourceStream = tuple[Path, str, SourceHandle]


def _iter_source_streams(dataset_dir: Path) -> Iterator[SourceStream]:
    archive_paths = sorted(dataset_dir.glob("*.json.tar.gz"))
    if archive_paths:
        for archive_path in archive_paths:
            with tarfile.open(archive_path, "r:gz") as archive:
                for member in archive:
                    if not member.isfile() or ".json" not in member.name:
                        continue
                    extracted = archive.extractfile(member)
                    if extracted is None:
                        continue
                    yield archive_path, member.name, extracted
        return

    for path in sorted(dataset_dir.iterdir()):
        if not path.is_file() or "json" not in path.name:
            continue
        if path.suffix == ".gz":
            with gzip.open(path, "rt") as handle:
                yield path, path.name, handle
        else:
            with path.open() as handle:
                yield path, path.name, handle


def _iter_decoded_lines(handle: SourceHandle) -> Iterator[str]:
    for raw_line in handle:
        yield raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line


def _get_union_value(field: object | None) -> object | None:
    if field is None:
        return None
    if isinstance(field, dict):
        for value in field.values():
            if value is None:
                return None
            if isinstance(value, dict):
                return _get_union_value(value)
            return value
    return field


def _coerce_int(value: object | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, str | bytes | bytearray):
        try:
            return int(value)
        except ValueError:
            return None
    if isinstance(value, float):
        return int(value)
    try:
        return int(cast(SupportsInt, value))
    except (TypeError, ValueError):
        return None


@dataclass
class CollectionGapAudit:
    dataset: str
    raw_dir: str
    stream_count: int = 0
    total_events: int = 0
    events_with_timestamp: int = 0
    events_with_sequence: int = 0
    timestamp_gap_threshold_seconds: float = 60.0
    timestamp_gap_count: int = 0
    total_timestamp_gap_seconds: float = 0.0
    max_timestamp_gap_seconds: float = 0.0
    out_of_order_timestamp_count: int = 0
    max_timestamp_backward_seconds: float = 0.0
    sequence_jump_count: int = 0
    total_sequence_jump_size: int = 0
    max_sequence_jump_size: int = 0
    sequence_regression_count: int = 0
    max_sequence_backward: int = 0
    parse_error_count: int = 0


def audit_dataset(
    dataset: str, raw_root: Path, gap_threshold_seconds: float
) -> CollectionGapAudit:
    dataset_dir = raw_root / dataset
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Raw dataset directory not found: {dataset_dir}")

    audit = CollectionGapAudit(
        dataset=dataset,
        raw_dir=str(dataset_dir),
        timestamp_gap_threshold_seconds=gap_threshold_seconds,
    )
    gap_threshold_nanos = int(gap_threshold_seconds * 1e9)

    for _path, _member, handle in _iter_source_streams(dataset_dir):
        audit.stream_count += 1
        previous_timestamp: int | None = None
        previous_sequence: int | None = None
        for line in _iter_decoded_lines(handle):
            if EVENT_MARKER not in line:
                continue
            try:
                event = json.loads(line)["datum"][EVENT_KEY]
            except Exception:
                audit.parse_error_count += 1
                continue

            audit.total_events += 1

            timestamp = _coerce_int(event.get("timestampNanos"))
            if timestamp is not None:
                audit.events_with_timestamp += 1
                if previous_timestamp is not None:
                    delta = timestamp - previous_timestamp
                    if delta < 0:
                        audit.out_of_order_timestamp_count += 1
                        audit.max_timestamp_backward_seconds = max(
                            audit.max_timestamp_backward_seconds, abs(delta) / 1e9
                        )
                    elif delta > gap_threshold_nanos:
                        gap_seconds = delta / 1e9
                        audit.timestamp_gap_count += 1
                        audit.total_timestamp_gap_seconds += gap_seconds
                        audit.max_timestamp_gap_seconds = max(
                            audit.max_timestamp_gap_seconds, gap_seconds
                        )
                previous_timestamp = timestamp

            sequence = _coerce_int(_get_union_value(event.get("sequence")))
            if sequence is not None:
                audit.events_with_sequence += 1
                if previous_sequence is not None:
                    delta = sequence - previous_sequence
                    if delta < 0:
                        audit.sequence_regression_count += 1
                        audit.max_sequence_backward = max(
                            audit.max_sequence_backward, abs(delta)
                        )
                    elif delta > 1:
                        jump_size = delta - 1
                        audit.sequence_jump_count += 1
                        audit.total_sequence_jump_size += jump_size
                        audit.max_sequence_jump_size = max(
                            audit.max_sequence_jump_size, jump_size
                        )
                previous_sequence = sequence

    return audit


def write_audit_json(audit: CollectionGapAudit, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{audit.dataset}_collection_gaps.json"
    out_path.write_text(json.dumps(asdict(audit), indent=2))
    return out_path


def print_summary(audits: list[CollectionGapAudit]) -> None:
    print("\nCOLLECTION GAP AUDIT")
    print("=" * 166)
    print(
        f"{'Dataset':<18} {'Streams':>8} {'Events':>14} {'Gap Cnt':>10} {'Avg Gap (s)':>12} "
        f"{'Max Gap (s)':>12} {'TS OOO':>10} {'Seq Jumps':>12} {'Seq Regress':>12} {'Parse Err':>10}"
    )
    print("-" * 166)
    for audit in audits:
        avg_gap_seconds = (
            audit.total_timestamp_gap_seconds / audit.timestamp_gap_count
            if audit.timestamp_gap_count
            else 0.0
        )
        print(
            f"{audit.dataset:<18} {audit.stream_count:>8,} {audit.total_events:>14,} "
            f"{audit.timestamp_gap_count:>10,} {avg_gap_seconds:>12.2f} "
            f"{audit.max_timestamp_gap_seconds:>12.2f} "
            f"{audit.out_of_order_timestamp_count:>10,} "
            f"{audit.sequence_jump_count:>12,} "
            f"{audit.sequence_regression_count:>12,} "
            f"{audit.parse_error_count:>10,}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit raw DARPA TC E3 event streams for collection gaps."
    )
    parser.add_argument(
        "datasets",
        nargs="*",
        default=list(DEFAULT_DATASETS),
        help="Raw dataset directory names under --raw-root.",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("data/raw"),
        help="Root directory containing the raw dataset folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/raw_audits"),
        help="Directory where per-dataset JSON summaries are written.",
    )
    parser.add_argument(
        "--gap-threshold-seconds",
        type=float,
        default=60.0,
        help="Timestamp gap threshold used to count collection gaps.",
    )
    args = parser.parse_args()

    audits = []
    for dataset in args.datasets:
        print(f"Auditing raw collection gaps: {dataset}")
        audits.append(audit_dataset(dataset, args.raw_root, args.gap_threshold_seconds))

    for audit in audits:
        path = write_audit_json(audit, args.output_dir)
        print(f"Wrote {path}")

    print_summary(audits)


if __name__ == "__main__":
    main()
