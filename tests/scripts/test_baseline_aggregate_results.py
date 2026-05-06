from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


def _load_module(module_name: str, relative_path: str):
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / relative_path
    spec = spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from {module_path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_magic_aggregate_parser_accepts_strict_only_summary(tmp_path):
    module = _load_module(
        "magic_aggregate_results", "baselines/MAGIC/utils/aggregate_results.py"
    )
    log_path = tmp_path / "theia_seed71.log"
    log_path.write_text(
        "\n".join(
            [
                "SUMMARY",
                "================================================================================",
                "Metric          Strict Attack Chain",
                "Precision       0.1234",
                "F1              0.1111",
                "AP              0.2222",
                "FPR             0.3333",
                "MCC             0.4444",
                "ADP             0.5555",
            ]
        ),
        encoding="utf-8",
    )

    assert module.parse_file(log_path) == {
        "Precision": 0.1234,
        "F1": 0.1111,
        "AP": 0.2222,
        "FPR": 0.3333,
        "MCC": 0.4444,
        "ADP": 0.5555,
    }


def test_magic_aggregate_parser_accepts_precision_from_final_results_block(tmp_path):
    module = _load_module(
        "magic_aggregate_results_precision",
        "baselines/MAGIC/utils/aggregate_results.py",
    )
    log_path = tmp_path / "trace_seed71.log"
    log_path.write_text(
        "\n".join(
            [
                "FINAL TEST RESULTS",
                "AP: 0.2222",
                "PRECISION: 0.3333",
                "SUMMARY",
                "================================================================================",
                "Metric          Strict Attack Chain",
                "F1              0.1111",
                "AP              0.2222",
                "FPR             0.4444",
                "MCC             0.5555",
                "ADP             0.6666",
            ]
        ),
        encoding="utf-8",
    )

    assert module.parse_file(log_path) == {
        "AP": 0.2222,
        "Precision": 0.3333,
        "F1": 0.1111,
        "FPR": 0.4444,
        "MCC": 0.5555,
        "ADP": 0.6666,
    }


def test_pidsmaker_aggregate_parser_prefers_scope_neutral_metrics(tmp_path):
    module = _load_module(
        "pidsmaker_aggregate_results",
        "baselines/PIDSMaker/scripts/aggregate_results.py",
    )
    log_path = tmp_path / "velox_THEIA_E3_seed71.log"
    log_path.write_text(
        "\n".join(
            [
                "final_ap: 0.1111",
                "final_precision: 0.1234",
                "final_fscore: 0.2222",
                "final_mcc: 0.3333",
                "final_fpr: 0.4444",
                "final_adp_score: 0.5555",
                "final_strict_ap: 0.9999",
            ]
        ),
        encoding="utf-8",
    )

    assert module.parse_file(log_path) == {
        "AP": 0.1111,
        "Precision": 0.1234,
        "F1": 0.2222,
        "MCC": 0.3333,
        "FPR": 0.4444,
        "ADP": 0.5555,
    }


def test_pidsmaker_aggregate_parser_accepts_legacy_strict_metrics(tmp_path):
    module = _load_module(
        "pidsmaker_aggregate_results_legacy",
        "baselines/PIDSMaker/scripts/aggregate_results.py",
    )
    log_path = tmp_path / "orthrus_TRACE_E3_seed71.log"
    log_path.write_text(
        "\n".join(
            [
                "final_strict_ap: 0.1010",
                "final_strict_precision: 0.1515",
                "final_strict_fscore: 0.2020",
                "final_strict_mcc: 0.3030",
                "final_strict_fpr: 0.4040",
                "final_strict_adp_score: 0.5050",
            ]
        ),
        encoding="utf-8",
    )

    assert module.parse_file(log_path) == {
        "AP": 0.101,
        "Precision": 0.1515,
        "F1": 0.202,
        "MCC": 0.303,
        "FPR": 0.404,
        "ADP": 0.505,
    }


def test_theseus_aggregate_parser_accepts_precision(tmp_path):
    module = _load_module("theseus_aggregate_results", "scripts/aggregate_results.py")
    log_path = tmp_path / "theseus_THEIA_E3_seed71.log"
    log_path.write_text(
        "\n".join(
            [
                "final_test_ap: 0.1111",
                "final_test_precision: 0.1234",
                "final_test_binary_f1: 0.2222",
                "final_test_mcc: 0.3333",
                "final_test_fpr: 0.4444",
                "test_adp: 0.5555",
            ]
        ),
        encoding="utf-8",
    )

    assert module.parse_file(log_path) == {
        "AP": 0.1111,
        "Precision": 0.1234,
        "F1": 0.2222,
        "MCC": 0.3333,
        "FPR": 0.4444,
        "ADP": 0.5555,
    }


def test_allowlist_aggregate_parser_reads_precision_from_csv(tmp_path):
    module = _load_module(
        "allowlist_aggregate_results", "scripts/aggregate_allowlist_results.py"
    )
    csv_path = tmp_path / "cadets_allowlist_diagnostic.csv"
    csv_path.write_text(
        "\n".join(
            [
                "dataset,precision,recall,f1,fpr,mcc",
                "CADETS_E3,0.1234,0.2345,0.3456,0.4567,0.5678",
            ]
        ),
        encoding="utf-8",
    )

    assert module.parse_csv(csv_path) == [
        {
            "dataset": "CADETS_E3",
            "Precision": 0.1234,
            "Recall": 0.2345,
            "F1": 0.3456,
            "FPR": 0.4567,
            "MCC": 0.5678,
        }
    ]


def test_allowlist_aggregate_default_patterns_include_legacy_names(tmp_path):
    module = _load_module(
        "allowlist_aggregate_results_patterns", "scripts/aggregate_allowlist_results.py"
    )
    current = tmp_path / "cadets_allowlist_diagnostic.csv"
    legacy = tmp_path / "cadets_binary_allowlist.csv"
    current.write_text("dataset,precision\nCADETS_E3,1.0\n", encoding="utf-8")
    legacy.write_text("dataset,precision\nCADETS_E3,1.0\n", encoding="utf-8")

    paths = module._collect_csv_paths(tmp_path, list(module.DEFAULT_PATTERNS))

    assert paths == [current, legacy]


def test_atlasv2_aggregate_accepts_legacy_allowlist_filenames(tmp_path):
    module = _load_module(
        "atlasv2_aggregate_results", "scripts/aggregate_atlasv2_secondary_results.py"
    )
    for dataset in ("atlasv2_h1", "atlasv2_h2"):
        (tmp_path / f"{dataset}_binary_allowlist.csv").write_text(
            "\n".join(
                [
                    "dataset,precision,recall,f1,fpr,mcc,n_test_process,n_attack,n_contaminated",
                    f"{dataset},0.7,0.2,0.3,0.01,0.4,10,2,1",
                ]
            ),
            encoding="utf-8",
        )

    report = module.aggregate_allowlist(tmp_path)

    assert report["atlasv2_h1"]["csv"].endswith("atlasv2_h1_binary_allowlist.csv")
    assert report["atlasv2_h2"]["precision"] == 0.7


def test_atlasv2_markdown_uses_paper_allowlist_label(tmp_path):
    module = _load_module(
        "atlasv2_aggregate_results_markdown",
        "scripts/aggregate_atlasv2_secondary_results.py",
    )

    def method_summary(value):
        return {
            "metrics": {
                metric: {"mean": value, "std": 0.0}
                for metric in ("ap", "precision", "f1", "mcc", "adp", "fpr")
            }
        }

    theseus = {dataset: method_summary(0.1) for dataset in module.ATLAS_DATASETS}
    velox = {dataset: method_summary(0.2) for dataset in module.ATLAS_DATASETS}
    allowlist = {
        dataset: {
            "csv": str(tmp_path / f"{dataset}_binary_allowlist.csv"),
            "precision": 0.7,
            "f1": 0.3,
            "mcc": 0.4,
            "fpr": 0.01,
        }
        for dataset in module.ATLAS_DATASETS
    }
    log_globs = {
        "theseus": dict.fromkeys(module.ATLAS_DATASETS, "theseus.log"),
        "velox": dict.fromkeys(module.ATLAS_DATASETS, "velox.log"),
    }

    markdown = module._format_markdown(
        theseus=theseus,
        velox=velox,
        allowlist=allowlist,
        log_globs=log_globs,
        source_mode="results",
        log_root=tmp_path,
    )

    assert "`Allowlist`" in markdown
    assert "Binary Allowlist" not in markdown
    assert "Allowlist Diagnostic" not in markdown
