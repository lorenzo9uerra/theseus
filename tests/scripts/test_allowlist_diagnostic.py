from pathlib import Path

import polars as pl

from scripts.allowlist_diagnostic import load_atlasv2_ground_truth_ids


def test_load_atlasv2_ground_truth_ids_uses_exact_process_uuid(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "data" / "atlasv2_h1"
    dataset_dir.mkdir(parents=True)
    pl.DataFrame(
        [
            {
                "index_id": 1,
                "node_uuid": "WIN-32-H1|guid1",
                "path": r"C:\same.exe",
                "pid": 10,
                "attack": "atlasv2/h1-s1",
            },
            {
                "index_id": 2,
                "node_uuid": "WIN-32-H1|guid2",
                "path": r"C:\same.exe",
                "pid": 10,
                "attack": "atlasv2/h1-s1",
            },
        ]
    ).write_parquet(dataset_dir / "process_node_table.parquet")

    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()
    (labels_dir / "h1-s1.labels").write_text(
        "attack,process_name,process_id,process_uuid,label\n"
        "h1-s1,c:\\same.exe,10,guid2,attack\n",
        encoding="utf-8",
    )

    attack, contaminated, excluded = load_atlasv2_ground_truth_ids(
        "atlasv2_h1", tmp_path / "data", labels_dir
    )

    assert attack == {"2"}
    assert contaminated == set()
    assert excluded == set()
