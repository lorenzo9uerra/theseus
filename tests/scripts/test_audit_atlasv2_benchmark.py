from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from scripts.audit_atlasv2_benchmark import (
    AtlasLabelRow,
    collect_allowlist_report,
    collect_dataset_table_counts,
    collect_label_alignment_report,
    collect_raw_scenario_report,
    collect_split_report,
    collect_startup_footprint_report,
)


def _write_parquet(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_parquet(path)


def _utc_ns(
    year: int, month: int, day: int, hour: int, minute: int, second: int
) -> int:
    return int(
        datetime(year, month, day, hour, minute, second, tzinfo=UTC).timestamp()
        * 1_000_000_000
    )


def _cbc_ts(
    year: int, month: int, day: int, hour: int, minute: int, second: int
) -> str:
    return (
        f"{year:04d}-{month:02d}-{day:02d} "
        f"{hour:02d}:{minute:02d}:{second:02d}.0000000 +0000 UTC"
    )


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row) + "\n")


def test_collect_dataset_table_counts_reads_processed_tables(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "atlasv2_h1"
    _write_parquet(
        dataset_dir / "event_table.parquet",
        [{"timestamp_rec": 1}, {"timestamp_rec": 2}, {"timestamp_rec": 3}],
    )
    _write_parquet(
        dataset_dir / "process_node_table.parquet", [{"index_id": 1}, {"index_id": 2}]
    )
    _write_parquet(dataset_dir / "file_node_table.parquet", [{"index_id": 3}])
    pl.DataFrame({"index_id": pl.Series([], dtype=pl.Int64)}).write_parquet(
        dataset_dir / "netflow_node_table.parquet"
    )

    counts = collect_dataset_table_counts(dataset_dir)

    assert counts == {
        "event_table": 3,
        "process_node_table": 2,
        "file_node_table": 1,
        "netflow_node_table": 0,
    }


def test_collect_allowlist_report_accepts_legacy_filename(tmp_path: Path) -> None:
    csv_path = tmp_path / "atlasv2_h1_binary_allowlist.csv"
    csv_path.write_text(
        "\n".join(
            [
                "dataset,precision,recall,f1,fpr,mcc,n_test_process,n_attack,n_contaminated,n_excluded",
                "atlasv2_h1,0.7,0.2,0.3,0.01,0.4,10,2,1,0",
            ]
        ),
        encoding="utf-8",
    )

    report = collect_allowlist_report(tmp_path, ["atlasv2_h1"])

    assert report["atlasv2_h1"]["precision"] == 0.7
    assert report["atlasv2_h1"]["n_contaminated"] == 1


def test_collect_split_report_counts_events_without_overlap(tmp_path: Path) -> None:
    atlas_root = tmp_path / "ATLASV2"
    dataset_dir = atlas_root / "atlasv2_h1"
    _write_parquet(
        dataset_dir / "event_table.parquet",
        [
            {"timestamp_rec": _utc_ns(2022, 7, 19, 1, 0, 0)},
            {"timestamp_rec": _utc_ns(2022, 7, 19, 12, 0, 0)},
            {"timestamp_rec": _utc_ns(2022, 7, 20, 3, 0, 0)},
        ],
    )

    config_path = tmp_path / "atlasv2.yml"
    config_path.write_text(
        "\n".join(
            [
                "atlasv2_h1:",
                '  year_month: "2022-07"',
                "  train_days: [15, 16, 17, 18]",
                "  val_days: []",
                "  test_days: [19, 20]",
                "",
            ]
        )
    )

    report = collect_split_report(atlas_root, config_path, ["atlasv2_h1"])
    split = report["atlasv2_h1"]

    assert split["split_overlap"]["val_test"] == []
    assert split["event_day_counts"] == {"2022-07-19": 2, "2022-07-20": 1}
    assert split["events_per_split"] == {"train": 0, "val": 0, "test": 3}


def test_collect_label_alignment_report_tracks_exact_fallback_and_misses(
    tmp_path: Path,
) -> None:
    atlas_root = tmp_path / "ATLASV2"
    dataset_dir = atlas_root / "atlasv2_h1"
    _write_parquet(
        dataset_dir / "process_node_table.parquet",
        [
            {
                "index_id": 1,
                "node_uuid": "WIN-32-H1|guid1",
                "path": r"C:\foo.exe",
                "pid": 10,
                "attack": "atlasv2/h1-s1",
            },
            {
                "index_id": 2,
                "node_uuid": "WIN-32-H1|guid2",
                "path": r"C:\foo.exe",
                "pid": 10,
                "attack": "atlasv2/h1-s1",
            },
            {
                "index_id": 3,
                "node_uuid": "WIN-32-H1|guid3",
                "path": r"C:\bar.exe",
                "pid": 20,
                "attack": "atlasv2/h1-s2",
            },
            {
                "index_id": 4,
                "node_uuid": "WIN-32-H1|guid4",
                "path": r"C:\bar.exe",
                "pid": 20,
                "attack": "",
            },
            {
                "index_id": 5,
                "node_uuid": "WIN-32-H1|guid5",
                "path": r"C:\baz.exe",
                "pid": 30,
                "attack": "",
            },
            {
                "index_id": 6,
                "node_uuid": "WIN-32-H1|guid6",
                "path": r"C:\qux.exe",
                "pid": 60,
                "attack": "atlasv2/h1-s1",
            },
        ],
    )

    label_rows = [
        AtlasLabelRow("atlasv2/h1-s1", "h1", 10, "c:/foo.exe", "attack"),
        AtlasLabelRow("atlasv2/h1-s1", "h1", 20, "c:/bar.exe", "attack"),
        AtlasLabelRow("atlasv2/h1-s1", "h1", 30, "c:/baz.exe", "attack"),
        AtlasLabelRow("atlasv2/h1-s1", "h1", 40, "c:/missing.exe", "attack"),
        AtlasLabelRow("atlasv2/h1-s1", "h1", 60, "c:/qux.exe", "attack"),
        AtlasLabelRow("atlasv2/h1-s1", "h1", 30, "c:/baz.exe", "attack"),
        AtlasLabelRow("atlasv2/h1-s1", "h1", 50, "c:/ignored.exe", "contaminated"),
    ]

    report = collect_label_alignment_report(atlas_root, ["atlasv2_h1"], label_rows)
    host_report = report["atlasv2_h1"]

    assert host_report["raw_attack_rows"] == 6
    assert host_report["unique_attack_rows"] == 5
    assert host_report["duplicate_attack_rows"] == 1
    assert host_report["raw_contaminated_rows"] == 1
    assert host_report["contaminated_match_counts"] == {
        "exact_unique": 0,
        "exact_ambiguous": 0,
        "host_fallback_unique": 0,
        "host_fallback_ambiguous": 0,
        "unmatched": 1,
    }
    assert host_report["matched_node_ids_from_contaminated_rows"] == 0
    assert host_report["unique_match_counts"] == {
        "exact_unique": 1,
        "exact_ambiguous": 1,
        "host_fallback_unique": 1,
        "host_fallback_ambiguous": 1,
        "unmatched": 1,
    }
    assert host_report["matched_node_ids_from_unique_attack_rows"] == 6


def test_collect_raw_reports_capture_spillover_and_repeated_prefixes(
    tmp_path: Path,
) -> None:
    raw_dir = tmp_path / "atlasv2" / "data"

    _write_jsonl(
        raw_dir / "attack" / "h1" / "cbc-edr" / "edr-h1-s1.jsonl",
        [
            {
                "device_timestamp": _cbc_ts(2022, 7, 19, 23, 59, 0),
                "type": "endpoint.event.procstart",
                "process_path": r"C:\windows\system32\svchost.exe",
                "childproc_name": r"C:\program files\confer\repux.exe",
            },
            {
                "device_timestamp": _cbc_ts(2022, 7, 19, 23, 59, 30),
                "type": "endpoint.event.procstart",
                "process_path": r"C:\windows\system32\svchost.exe",
                "childproc_name": r"C:\windows\system32\conhost.exe",
            },
            {
                "device_timestamp": _cbc_ts(2022, 7, 20, 0, 10, 0),
                "type": "endpoint.event.procstart",
                "process_path": r"C:\attack-one.exe",
                "childproc_name": r"C:\attack-one-child.exe",
            },
        ],
    )
    _write_jsonl(
        raw_dir / "attack" / "h1" / "cbc-edr" / "edr-h1-s2.jsonl",
        [
            {
                "device_timestamp": _cbc_ts(2022, 7, 19, 13, 0, 0),
                "type": "endpoint.event.procstart",
                "process_path": r"C:\windows\system32\svchost.exe",
                "childproc_name": r"C:\program files\confer\repux.exe",
            },
            {
                "device_timestamp": _cbc_ts(2022, 7, 19, 13, 1, 0),
                "type": "endpoint.event.procstart",
                "process_path": r"C:\windows\system32\svchost.exe",
                "childproc_name": r"C:\windows\system32\conhost.exe",
            },
            {
                "device_timestamp": _cbc_ts(2022, 7, 19, 13, 10, 0),
                "type": "endpoint.event.procstart",
                "process_path": r"C:\attack-two.exe",
                "childproc_name": r"C:\attack-two-child.exe",
            },
        ],
    )

    raw_report = collect_raw_scenario_report(raw_dir)
    startup_report = collect_startup_footprint_report(
        raw_dir, prefix_minutes=2, min_common_scenarios=2
    )

    assert raw_report["scenario_count"] == 2
    assert raw_report["spillover_scenarios"] == ["atlasv2/h1-s1"]
    assert startup_report["scenario_count"] == 2
    assert startup_report["prefix_pairwise_jaccard"]["mean_jaccard"] == 1.0
    assert startup_report["suffix_pairwise_jaccard"]["mean_jaccard"] == 0.0
    assert startup_report["common_prefix_process_paths"] == sorted(
        [
            {"path": "c:/windows/system32/svchost.exe", "scenario_count": 2},
            {"path": "c:/program files/confer/repux.exe", "scenario_count": 2},
            {"path": "c:/windows/system32/conhost.exe", "scenario_count": 2},
        ],
        key=lambda item: (-item["scenario_count"], item["path"]),
    )
