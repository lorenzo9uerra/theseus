#!/usr/bin/env python3
"""Audit raw E3 CDM logs for graph-linkability and field coverage."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import tarfile
from collections.abc import Iterator, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import IO, cast

SUBJECT_KEY = "com.bbn.tc.schema.avro.cdm18.Subject"
FILE_KEY = "com.bbn.tc.schema.avro.cdm18.FileObject"
NETFLOW_KEY = "com.bbn.tc.schema.avro.cdm18.NetFlowObject"
EVENT_KEY = "com.bbn.tc.schema.avro.cdm18.Event"
UUID_KEY = "com.bbn.tc.schema.avro.cdm18.UUID"

SUBJECT_MARKER = f'{{"datum":{{"{SUBJECT_KEY}"'
FILE_MARKER = f'{{"datum":{{"{FILE_KEY}"'
NETFLOW_MARKER = f'{{"datum":{{"{NETFLOW_KEY}"'
EVENT_MARKER = f'{{"datum":{{"{EVENT_KEY}"'

DEFAULT_DATASETS = ("cadets", "theia", "fivedirections", "trace")

type SourceHandle = IO[str] | IO[bytes]
type SourceStream = tuple[Path, str, SourceHandle]
type EntityCoverageResult = tuple[
    set[bytes], set[bytes], set[bytes], int, "DatasetCoverage", int, int, int
]


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


def _fingerprint(text: str) -> bytes:
    return hashlib.blake2b(text.encode("utf-8"), digest_size=16).digest()


def _normalized_text(value: object | None) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _value_digest_set_add(target: set[bytes], value: str) -> bool:
    if not value:
        return False
    target.add(_fingerprint(value))
    return True


def _extract_uuid(field: object | None) -> str:
    if not isinstance(field, Mapping):
        return ""
    mapping = cast(Mapping[str, object], field)
    value = mapping.get(UUID_KEY)
    return _normalized_text(value)


def _extract_object_uuid(event: dict) -> tuple[str, str]:
    for field_name in ("predicateObject", "predicateObject2", "predicateObject1"):
        uuid = _extract_uuid(event.get(field_name))
        if uuid:
            return uuid, field_name
    return "", ""


def _extract_subject_uuid(event: dict) -> tuple[str, str]:
    for field_name in ("subject", "process"):
        uuid = _extract_uuid(event.get(field_name))
        if uuid:
            return uuid, field_name
    return "", ""


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


@dataclass
class FieldCoverage:
    total_entities: int = 0
    non_empty_entities: int = 0
    unique_values: int = 0


@dataclass
class DatasetCoverage:
    process_cmd: FieldCoverage
    process_path: FieldCoverage
    file_path: FieldCoverage
    netflow_desc: FieldCoverage


@dataclass
class UUIDLinkageStats:
    total_events: int = 0
    events_with_subject_uuid: int = 0
    events_with_object_uuid: int = 0
    events_with_both_uuids: int = 0
    subject_uuid_resolves: int = 0
    object_uuid_resolves: int = 0
    events_with_both_entities_resolved: int = 0
    orphaned_events: int = 0
    missing_subject_uuid: int = 0
    missing_object_uuid: int = 0
    unresolved_subject_uuid: int = 0
    unresolved_object_uuid: int = 0
    predicate_object2_used: int = 0
    parse_error_count: int = 0


@dataclass
class DatasetAudit:
    dataset: str
    raw_dir: str
    total_process_entities: int
    total_file_entities: int
    total_netflow_entities: int
    entity_parse_error_count: int
    coverage: DatasetCoverage
    linkage: UUIDLinkageStats


def collect_entities_and_coverage(dataset_dir: Path) -> EntityCoverageResult:
    process_entities: dict[bytes, list[str]] = {}
    file_entities: dict[bytes, str] = {}
    netflow_entities: dict[bytes, str] = {}

    process_cmd_values: set[bytes] = set()
    process_path_values: set[bytes] = set()
    file_path_values: set[bytes] = set()
    netflow_desc_values: set[bytes] = set()
    event_paths: dict[bytes, str] = {}
    event_execs: dict[bytes, str] = {}
    parse_error_count = 0

    for _path, _member, handle in _iter_source_streams(dataset_dir):
        for line in _iter_decoded_lines(handle):
            try:
                if SUBJECT_MARKER in line:
                    record = json.loads(line)
                    subject = record["datum"][SUBJECT_KEY]

                    uuid = _normalized_text(subject.get("uuid"))
                    uuid_key = _fingerprint(uuid) if uuid else b""

                    cmd = _normalized_text(_get_union_value(subject.get("cmdLine")))
                    proc_path = _normalized_text(_get_union_value(subject.get("path")))

                    properties = subject.get("properties")
                    if isinstance(properties, dict):
                        props_map = properties.get("map")
                        if isinstance(props_map, dict):
                            if not proc_path:
                                proc_path = _normalized_text(props_map.get("path"))
                            if not cmd:
                                cmd = _normalized_text(props_map.get("name"))
                    if uuid_key:
                        existing_path, existing_cmd = process_entities.get(
                            uuid_key, ["", ""]
                        )
                        if not proc_path:
                            proc_path = existing_path
                        if not cmd:
                            cmd = existing_cmd
                        process_entities[uuid_key] = [proc_path, cmd]

                elif FILE_MARKER in line:
                    record = json.loads(line)
                    file_object = record["datum"][FILE_KEY]

                    uuid = _normalized_text(file_object.get("uuid"))
                    uuid_key = _fingerprint(uuid) if uuid else b""

                    filename = _normalized_text(
                        _get_union_value(file_object.get("filename"))
                    )
                    if not filename:
                        base_object = file_object.get("baseObject")
                        if isinstance(base_object, dict):
                            properties = base_object.get("properties")
                            if isinstance(properties, dict):
                                props_map = properties.get("map")
                                if isinstance(props_map, dict):
                                    filename = _normalized_text(
                                        props_map.get("filename")
                                    )
                    if uuid_key:
                        if not filename:
                            filename = file_entities.get(uuid_key, "")
                        file_entities[uuid_key] = filename

                elif NETFLOW_MARKER in line:
                    record = json.loads(line)
                    netflow = record["datum"][NETFLOW_KEY]

                    uuid = _normalized_text(netflow.get("uuid"))
                    uuid_key = _fingerprint(uuid) if uuid else b""

                    src_addr = _normalized_text(
                        _get_union_value(netflow.get("localAddress"))
                    )
                    src_port = _normalized_text(
                        _get_union_value(netflow.get("localPort"))
                    )
                    dst_addr = _normalized_text(
                        _get_union_value(netflow.get("remoteAddress"))
                    )
                    dst_port = _normalized_text(
                        _get_union_value(netflow.get("remotePort"))
                    )

                    if all((src_addr, src_port, dst_addr, dst_port)):
                        description = f"{src_addr} {src_port} {dst_addr} {dst_port}"
                    else:
                        description = ""
                    if uuid_key:
                        if not description:
                            description = netflow_entities.get(uuid_key, "")
                        netflow_entities[uuid_key] = description

                elif EVENT_MARKER in line:
                    if '"predicateObjectPath":null' in line and '"exec"' not in line:
                        continue
                    record = json.loads(line)
                    event = record["datum"][EVENT_KEY]

                    object_uuid, _object_field = _extract_object_uuid(event)
                    object_path = _normalized_text(
                        _get_union_value(event.get("predicateObjectPath"))
                    )
                    if object_uuid and object_path:
                        event_paths[_fingerprint(object_uuid)] = object_path

                    subject_uuid, _subject_field = _extract_subject_uuid(event)
                    properties = event.get("properties")
                    if (
                        subject_uuid
                        and isinstance(properties, dict)
                        and "map" in properties
                    ):
                        props_map = properties["map"]
                        if isinstance(props_map, dict):
                            cmd = _normalized_text(props_map.get("exec"))
                            if not cmd:
                                cmd = _normalized_text(props_map.get("name"))
                            if cmd:
                                event_execs[_fingerprint(subject_uuid)] = cmd
            except Exception:
                parse_error_count += 1
                continue

    for uuid_key, path in event_paths.items():
        if uuid_key in process_entities and not process_entities[uuid_key][0]:
            process_entities[uuid_key][0] = path
        if uuid_key in file_entities and not file_entities[uuid_key]:
            file_entities[uuid_key] = path

    for uuid_key, cmd in event_execs.items():
        if uuid_key in process_entities and not process_entities[uuid_key][1]:
            process_entities[uuid_key][1] = cmd

    for proc_path, cmd in process_entities.values():
        _value_digest_set_add(process_cmd_values, cmd)
        _value_digest_set_add(process_path_values, proc_path)

    for filename in file_entities.values():
        _value_digest_set_add(file_path_values, filename)

    for description in netflow_entities.values():
        _value_digest_set_add(netflow_desc_values, description)

    total_process = len(process_entities)
    total_file = len(file_entities)
    total_netflow = len(netflow_entities)
    process_cmd_non_empty = sum(1 for _path, cmd in process_entities.values() if cmd)
    process_path_non_empty = sum(1 for path, _cmd in process_entities.values() if path)
    file_path_non_empty = sum(1 for path in file_entities.values() if path)
    netflow_desc_non_empty = sum(1 for desc in netflow_entities.values() if desc)

    coverage = DatasetCoverage(
        process_cmd=FieldCoverage(
            total_entities=total_process,
            non_empty_entities=process_cmd_non_empty,
            unique_values=len(process_cmd_values),
        ),
        process_path=FieldCoverage(
            total_entities=total_process,
            non_empty_entities=process_path_non_empty,
            unique_values=len(process_path_values),
        ),
        file_path=FieldCoverage(
            total_entities=total_file,
            non_empty_entities=file_path_non_empty,
            unique_values=len(file_path_values),
        ),
        netflow_desc=FieldCoverage(
            total_entities=total_netflow,
            non_empty_entities=netflow_desc_non_empty,
            unique_values=len(netflow_desc_values),
        ),
    )

    return (
        set(process_entities.keys()),
        set(file_entities.keys()),
        set(netflow_entities.keys()),
        parse_error_count,
        coverage,
        total_process,
        total_file,
        total_netflow,
    )


def audit_event_uuid_linkage(
    dataset_dir: Path,
    process_uuids: set[bytes],
    file_uuids: set[bytes],
    netflow_uuids: set[bytes],
) -> UUIDLinkageStats:
    all_known = process_uuids | file_uuids | netflow_uuids
    stats = UUIDLinkageStats()

    for _path, _member, handle in _iter_source_streams(dataset_dir):
        for line in _iter_decoded_lines(handle):
            if EVENT_MARKER not in line:
                continue
            try:
                record = json.loads(line)
                event = record["datum"][EVENT_KEY]
            except Exception:
                stats.parse_error_count += 1
                continue

            stats.total_events += 1

            subject_uuid, _subject_field = _extract_subject_uuid(event)
            object_uuid, object_field = _extract_object_uuid(event)

            if subject_uuid:
                stats.events_with_subject_uuid += 1
            else:
                stats.missing_subject_uuid += 1

            if object_uuid:
                stats.events_with_object_uuid += 1
                if object_field == "predicateObject2":
                    stats.predicate_object2_used += 1
            else:
                stats.missing_object_uuid += 1

            if subject_uuid and object_uuid:
                stats.events_with_both_uuids += 1

            subject_resolves = False
            if subject_uuid:
                subject_resolves = _fingerprint(subject_uuid) in process_uuids
                if subject_resolves:
                    stats.subject_uuid_resolves += 1
                else:
                    stats.unresolved_subject_uuid += 1

            object_resolves = False
            if object_uuid:
                object_resolves = _fingerprint(object_uuid) in all_known
                if object_resolves:
                    stats.object_uuid_resolves += 1
                else:
                    stats.unresolved_object_uuid += 1

            if subject_resolves and object_resolves:
                stats.events_with_both_entities_resolved += 1

    stats.orphaned_events = (
        stats.total_events - stats.events_with_both_entities_resolved
    )
    return stats


def audit_dataset(dataset: str, raw_root: Path) -> DatasetAudit:
    dataset_dir = raw_root / dataset
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Raw dataset directory not found: {dataset_dir}")

    (
        process_uuids,
        file_uuids,
        netflow_uuids,
        entity_parse_error_count,
        coverage,
        total_process,
        total_file,
        total_netflow,
    ) = collect_entities_and_coverage(dataset_dir)

    linkage = audit_event_uuid_linkage(
        dataset_dir, process_uuids, file_uuids, netflow_uuids
    )

    return DatasetAudit(
        dataset=dataset,
        raw_dir=str(dataset_dir),
        total_process_entities=total_process,
        total_file_entities=total_file,
        total_netflow_entities=total_netflow,
        entity_parse_error_count=entity_parse_error_count,
        coverage=coverage,
        linkage=linkage,
    )


def write_dataset_json(audit: DatasetAudit, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{audit.dataset}_raw_audit.json"
    out_path.write_text(json.dumps(asdict(audit), indent=2))
    return out_path


def print_uuid_summary(audits: list[DatasetAudit]) -> None:
    print("\nUUID LINKAGE AUDIT")
    print("=" * 125)
    print(
        f"{'Dataset':<18} {'Events':>14} {'Subject UUID':>14} {'Object UUID':>14} "
        f"{'Both Resolved':>16} {'Orphaned':>14} {'Parse Err':>10}"
    )
    print("-" * 125)
    for audit in audits:
        linkage = audit.linkage
        print(
            f"{audit.dataset:<18} "
            f"{linkage.total_events:>14,} "
            f"{linkage.events_with_subject_uuid:>14,} "
            f"{linkage.events_with_object_uuid:>14,} "
            f"{linkage.events_with_both_entities_resolved:>16,} "
            f"{linkage.orphaned_events:>14,} "
            f"{linkage.parse_error_count + audit.entity_parse_error_count:>10,}"
        )


def print_coverage_summary(audits: list[DatasetAudit]) -> None:
    print("\nFEATURE COVERAGE BY RAW FIELD")
    print("=" * 130)
    print(
        f"{'Dataset':<18} {'Field':<16} {'Total':>12} {'Non-empty':>12} "
        f"{'Coverage %':>12} {'Unique':>12}"
    )
    print("-" * 130)
    for audit in audits:
        fields = {
            "process_cmd": audit.coverage.process_cmd,
            "process_path": audit.coverage.process_path,
            "file_path": audit.coverage.file_path,
            "netflow_desc": audit.coverage.netflow_desc,
        }
        for field_name, field in fields.items():
            coverage_pct = (
                100.0 * field.non_empty_entities / field.total_entities
                if field.total_entities
                else 0.0
            )
            print(
                f"{audit.dataset:<18} {field_name:<16} {field.total_entities:>12,} "
                f"{field.non_empty_entities:>12,} {coverage_pct:>11.2f}% "
                f"{field.unique_values:>12,}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit raw DARPA TC E3 source logs before preprocessing."
    )
    parser.add_argument(
        "datasets",
        nargs="*",
        default=list(DEFAULT_DATASETS),
        help="Raw dataset directory names under --raw-root "
        "(default: cadets theia fivedirections trace)",
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
    args = parser.parse_args()

    audits = []
    for dataset in args.datasets:
        print(f"Auditing raw dataset: {dataset}")
        audits.append(audit_dataset(dataset, args.raw_root))
    for audit in audits:
        path = write_dataset_json(audit, args.output_dir)
        print(f"Wrote {path}")

    print_uuid_summary(audits)
    print_coverage_summary(audits)


if __name__ == "__main__":
    main()
