from types import SimpleNamespace

import polars as pl

from pidsmaker.utils.labelling import parse_atlasv2_ground_truth


def test_parse_atlasv2_ground_truth_uses_exact_process_uuid(tmp_path):
    dataset_dir = tmp_path / "data"
    dataset_dir.mkdir()
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

    ground_truth_dir = tmp_path / "ground_truth"
    label_dir = ground_truth_dir / "atlasv2"
    label_dir.mkdir(parents=True)
    (label_dir / "h1-s1.labels").write_text(
        "attack,process_name,process_id,process_uuid,label\n"
        "h1-s1,c:\\same.exe,10,guid2,attack\n",
        encoding="utf-8",
    )
    cfg = SimpleNamespace(
        dataset=SimpleNamespace(
            name="atlasv2_h1",
            csv_dir=str(dataset_dir),
            reapr_ground_truth_path="atlasv2/atlasv2_labels.csv",
        ),
        _ground_truth_dir=str(ground_truth_dir),
    )

    (
        attack_chains,
        attack_nids,
        contaminated_nids,
        all_nids,
        _,
        _,
        excluded_nids,
    ) = parse_atlasv2_ground_truth(cfg)

    assert set(attack_chains) == {"atlasv2/h1-s1"}
    assert attack_nids == {2}
    assert contaminated_nids == set()
    assert all_nids == {2}
    assert excluded_nids == set()
