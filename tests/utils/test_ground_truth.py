from pathlib import Path

from utils.ground_truth import _get_atlasv2_labels_dir, load_atlasv2_process_labels


def test_atlasv2_label_dir_accepts_artifact_override(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("ATLASV2_LABEL_DIR", str(tmp_path))

    assert _get_atlasv2_labels_dir() == str(tmp_path)


def test_load_atlasv2_process_labels_prefers_revised_uuid_files(tmp_path: Path) -> None:
    (tmp_path / "atlasv2_labels.csv").write_text(
        "attack,process_name,process_id,label\n"
        "atlasv2/h1-s1,c:\\legacy.exe,1,contaminated\n",
        encoding="utf-8",
    )
    (tmp_path / "h1-s1.labels").write_text(
        "attack, process_name, process_id, process_uuid, label\n"
        "h1-s1, c:\\attack.exe, 42.0, GUID-42, attack\n",
        encoding="utf-8",
    )

    labels = load_atlasv2_process_labels(tmp_path)

    assert len(labels) == 1
    assert labels[0].attack_id == "atlasv2/h1-s1"
    assert labels[0].process_uuid == "guid-42"
    assert labels[0].pid == 42
    assert labels[0].path == "c:/attack.exe"
    assert labels[0].label == "attack"


def test_load_atlasv2_process_labels_falls_back_to_legacy_csv(tmp_path: Path) -> None:
    (tmp_path / "atlasv2_labels.csv").write_text(
        "attack,process_name,process_id,label\n"
        "atlasv2/h2-m1,c:\\legacy.exe,7,contaminated\n",
        encoding="utf-8",
    )

    labels = load_atlasv2_process_labels(tmp_path)

    assert len(labels) == 1
    assert labels[0].attack_id == "atlasv2/h2-m1"
    assert labels[0].process_uuid == ""
    assert labels[0].pid == 7
    assert labels[0].label == "contaminated"
