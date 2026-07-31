#!/usr/bin/env python3
"""Evaluate the allowlist diagnostic for unseen process executables and paths."""

import argparse
import csv
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.ground_truth import load_atlasv2_process_labels  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


@dataclass
class EvaluationMetrics:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    fpr: float = 0.0
    mcc: float = 0.0


@dataclass
class DatasetStats:
    dataset: str
    cmd_mode: str = "executable"
    vocab_size_train_cmd: int = 0
    vocab_size_train_path: int = 0
    novel_test_cmds: int = 0
    novel_test_paths: int = 0
    n_test_process: int = 0
    n_excluded: int = 0
    n_attack: int = 0
    n_contaminated: int = 0
    n_flagged: int = 0
    metrics: EvaluationMetrics = field(default_factory=EvaluationMetrics)


def load_dataset_config(dataset_name: str, config_dir: Path) -> dict[str, Any]:
    """Load dataset configuration from YAML files."""
    base_name = dataset_name.lower().split("_")[0]
    config_path = config_dir / f"{base_name}.yml"

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        configs = yaml.safe_load(f)

    if dataset_name not in configs:
        raise ValueError(f"Dataset {dataset_name} not found in {config_path}")

    return configs[dataset_name]


def get_all_datasets(config_dir: Path) -> list[str]:
    """Discover all datasets available in config directory."""
    datasets = []
    if not config_dir.exists():
        return []

    for config_file in config_dir.glob("*.yml"):
        if config_file.name == "test.yml":
            continue
        with open(config_file) as f:
            try:
                configs = yaml.safe_load(f)
                if configs:
                    datasets.extend(configs.keys())
            except yaml.YAMLError:
                continue
    return sorted(datasets)


def compute_classification_metrics(
    flagged: set[str], positive: set[str], universe: set[str]
) -> EvaluationMetrics:
    """Compute standard binary classification metrics."""
    tp_set = flagged & positive
    fp_set = flagged - positive
    fn_set = positive - flagged
    tn_set = universe - flagged - positive

    tp, fp, fn, tn = len(tp_set), len(fp_set), len(fn_set), len(tn_set)

    eps = 1e-12
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    fpr = fp / (fp + tn + eps)

    numerator = tp * tn - fp * fn
    denominator = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    mcc = numerator / denominator if denominator > 0 else 0.0

    return EvaluationMetrics(
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
        precision=precision,
        recall=recall,
        f1_score=f1,
        fpr=fpr,
        mcc=mcc,
    )


def load_ground_truth(
    dataset_name: str,
    ground_truth_dir: Path,
    excluded_attack_chains: set[str] | None = None,
) -> tuple[set[str], set[str], set[str]]:
    """
    Parses ground truth CSV to retrieve Attack and Contaminated UUIDs.
    Returns: (attack_uuids, contaminated_uuids, excluded_uuids)
    """
    gt_path = ground_truth_dir / f"{dataset_name.lower()}_labels.csv"

    if not gt_path.exists():
        logger.warning(f"Ground truth file missing: {gt_path}")
        return set(), set(), set()

    attack_uuids = set()
    contaminated_uuids = set()
    excluded_uuids = set()
    excluded_attack_chains = excluded_attack_chains or set()

    with open(gt_path, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or len(row) < 4:
                continue

            cleaned_row = [c.strip() for c in row]
            if (
                cleaned_row[0].lower() == "attack_chain"
                or cleaned_row[1].lower() == "uuid"
            ):
                continue

            chain = cleaned_row[0]
            uuid = cleaned_row[1]
            label = cleaned_row[-1].lower()

            if not uuid:
                continue

            if chain in excluded_attack_chains:
                excluded_uuids.add(uuid)

            if label == "attack":
                attack_uuids.add(uuid)
            elif label == "contaminated":
                contaminated_uuids.add(uuid)

    return attack_uuids, contaminated_uuids, excluded_uuids


def _safe_int(value: str | int | None) -> int | None:
    try:
        return int(value) if value not in {None, ""} else None
    except (TypeError, ValueError):
        return None


def _normalize_windows_path(path: str) -> str:
    return path.strip().replace("/", "\\").lower()


def _infer_atlas_host(node_uuid: str) -> str | None:
    device_name = node_uuid.split("|", 1)[0] if "|" in node_uuid else ""
    if "-H1" in device_name.upper():
        return "h1"
    if "-H2" in device_name.upper():
        return "h2"
    return None


def load_atlasv2_ground_truth_ids(
    dataset_name: str, data_dir: Path, ground_truth_dir: Path
) -> tuple[set[str], set[str], set[str]]:
    try:
        labels = load_atlasv2_process_labels(ground_truth_dir)
    except ValueError as exc:
        logger.warning("%s", exc)
        return set(), set(), set()

    process_path = data_dir / dataset_name / "process_node_table.parquet"
    process_df = pl.read_parquet(
        process_path, columns=["index_id", "node_uuid", "path", "pid", "attack"]
    )

    by_attack_pid_path: dict[tuple[str, int, str], set[str]] = {}
    by_host_pid_path: dict[tuple[str, int, str], set[str]] = {}
    by_process_uuid: dict[str, set[str]] = {}

    for row in process_df.iter_rows(named=True):
        idx = row.get("index_id")
        pid = _safe_int(row.get("pid"))
        attack_id = (row.get("attack") or "").strip()
        path = _normalize_windows_path(row.get("path") or "")
        node_uuid = (row.get("node_uuid") or "").strip()
        host = _infer_atlas_host(node_uuid)

        if idx is None:
            continue

        idx_str = str(idx)
        process_uuid = node_uuid.split("|")[-1].strip().lower()
        if process_uuid:
            by_process_uuid.setdefault(process_uuid, set()).add(idx_str)
        if attack_id and pid is not None and path:
            by_attack_pid_path.setdefault((attack_id, pid, path), set()).add(idx_str)
        if host and pid is not None and path:
            by_host_pid_path.setdefault((host, pid, path), set()).add(idx_str)

    dataset_host = dataset_name.split("_", 1)[1].lower()
    attack_ids: set[str] = set()
    contaminated_ids: set[str] = set()
    missing = 0

    for process_label in labels:
        if not process_label.attack_id.startswith(f"atlasv2/{dataset_host}-"):
            continue

        if process_label.process_uuid:
            matches = by_process_uuid.get(process_label.process_uuid)
            if matches and len(matches) > 1:
                raise ValueError(
                    f"ATLASv2 process UUID '{process_label.process_uuid}' "
                    f"maps to multiple process nodes"
                )
        elif process_label.pid is None:
            matches = None
        else:
            matches = by_attack_pid_path.get(
                (
                    process_label.attack_id,
                    process_label.pid,
                    _normalize_windows_path(process_label.path),
                )
            )
            if not matches:
                matches = by_host_pid_path.get(
                    (
                        dataset_host,
                        process_label.pid,
                        _normalize_windows_path(process_label.path),
                    )
                )

        if not matches:
            missing += 1
            continue

        if process_label.label == "attack":
            attack_ids.update(matches)
        else:
            contaminated_ids.update(matches)

    if missing:
        logger.info("ATLASv2 labels unmatched for %s: %d rows", dataset_name, missing)

    return attack_ids, contaminated_ids, set()


def load_process_metadata(
    dataset_name: str, data_dir: Path
) -> tuple[dict[str, tuple[str, str]], dict[str, str]]:
    """Loads process node table. Returns (index_id -> (path, cmd), uuid -> index_id)."""
    table_candidates = ("process_node_table", "subject_node_table")
    df = None
    chosen_path = None

    for table_name in table_candidates:
        parquet_path = data_dir / dataset_name / f"{table_name}.parquet"
        csv_path = data_dir / dataset_name / f"{table_name}.csv"

        if parquet_path.exists():
            df = pl.read_parquet(
                parquet_path, columns=["node_uuid", "index_id", "path", "cmd"]
            )
            chosen_path = parquet_path
            break
        if csv_path.exists():
            df = pl.read_csv(csv_path, columns=["node_uuid", "index_id", "path", "cmd"])
            chosen_path = csv_path
            break

    if df is None:
        logger.warning(
            "Process/subject table missing: %s",
            data_dir / dataset_name / "process_node_table.csv",
        )
        return {}, {}

    rows = df.with_columns(
        [pl.col("path").fill_null(""), pl.col("cmd").fill_null("")]
    ).rows(named=True)

    meta_map = {str(r["index_id"]): (r["path"], r["cmd"]) for r in rows}
    uuid_map = {r["node_uuid"]: str(r["index_id"]) for r in rows}

    logger.info("Loaded process metadata from %s", chosen_path)
    return meta_map, uuid_map


def get_active_nodes(
    dataset_name: str, days: list[int], year_month: str, data_dir: Path
) -> set[str]:
    """Retrieve set of active node IDs for given days from event table."""
    parquet_path = data_dir / dataset_name / "event_table.parquet"
    csv_path = data_dir / dataset_name / "event_table.csv"

    day_strs = [f"{year_month}-{int(d):02d}" for d in days]

    if parquet_path.exists():
        lf = pl.scan_parquet(parquet_path)
    elif csv_path.exists():
        lf = pl.scan_csv(csv_path)
    else:
        raise FileNotFoundError(f"Event table not found in {data_dir / dataset_name}")

    res = (
        lf.with_columns(
            pl.col("timestamp_rec")
            .cast(pl.Int64)
            .cast(pl.Datetime("ns"))
            .dt.strftime("%Y-%m-%d")
            .alias("event_day")
        )
        .filter(pl.col("event_day").is_in(day_strs))
        .select(["src_index_id", "dst_index_id"])
        .collect()
    )

    assert isinstance(res, pl.DataFrame)

    return set(map(str, res["src_index_id"].to_list())) | set(
        map(str, res["dst_index_id"].to_list())
    )


def analyze_dataset(
    dataset_name: str,
    config: dict,
    data_dir: Path,
    ground_truth_dir: Path,
    *,
    cmd_mode: str = "executable",
) -> DatasetStats | None:
    """Execute analysis pipeline for a single dataset."""
    logger.info(f"Analyzing {dataset_name} ({cmd_mode} command mode)...")

    proc_meta, uuid_to_id = load_process_metadata(dataset_name, data_dir)
    if not proc_meta:
        return None

    try:
        val_days = set(config.get("val_days", []) or [])
        effective_test_days = [
            int(day) for day in config["test_days"] if int(day) not in val_days
        ]
        train_nodes = get_active_nodes(
            dataset_name, config["train_days"], config["year_month"], data_dir
        )
        test_nodes_all = get_active_nodes(
            dataset_name, effective_test_days, config["year_month"], data_dir
        )
    except FileNotFoundError as e:
        logger.error(str(e))
        return None

    train_procs = train_nodes & proc_meta.keys()
    test_procs = test_nodes_all & proc_meta.keys()

    train_paths = {proc_meta[n][0] for n in train_procs if proc_meta[n][0]}

    def get_cmd_feature(cmd_str):
        if not cmd_str:
            return ""
        cmd_str = cmd_str.strip()
        if cmd_mode == "full":
            return cmd_str
        return cmd_str.split(" ")[0]

    train_cmds = {
        get_cmd_feature(proc_meta[n][1]) for n in train_procs if proc_meta[n][1]
    }

    flagged_nodes = set()
    test_cmds = set()
    test_paths = set()

    for nid in test_procs:
        path, cmd = proc_meta[nid]

        cmd_token = get_cmd_feature(cmd)

        if cmd_token:
            test_cmds.add(cmd_token)
        if path:
            test_paths.add(path)

        is_novel_cmd = cmd_token and (cmd_token not in train_cmds)
        is_novel_path = path and (path not in train_paths)

        if is_novel_cmd or is_novel_path:
            flagged_nodes.add(nid)
    novel_test_cmds = len(test_cmds - train_cmds)
    novel_test_paths = len(test_paths - train_paths)

    if dataset_name.lower().startswith("atlasv2_"):
        attack_ids, contam_ids, excluded_ids = load_atlasv2_ground_truth_ids(
            dataset_name, data_dir, ground_truth_dir
        )
    else:
        excluded_attack_chains = set(config.get("excluded_attack_chains", []) or [])
        attack_uuids, contam_uuids, excluded_uuids = load_ground_truth(
            dataset_name.split("_")[0],
            ground_truth_dir,
            excluded_attack_chains=excluded_attack_chains,
        )

        attack_ids = {uuid_to_id[u] for u in attack_uuids if u in uuid_to_id}
        contam_ids = {uuid_to_id[u] for u in contam_uuids if u in uuid_to_id}
        excluded_ids = {uuid_to_id[u] for u in excluded_uuids if u in uuid_to_id}

    excluded_ids = excluded_ids & test_procs
    test_procs_eval = test_procs - excluded_ids
    flagged_eval = flagged_nodes - excluded_ids
    test_attacks = (attack_ids & test_procs_eval) - excluded_ids
    test_contams = (contam_ids & test_procs_eval) - excluded_ids

    metric_universe = test_procs_eval - test_contams
    metric_flagged = flagged_eval - test_contams
    metrics = compute_classification_metrics(
        metric_flagged, test_attacks, metric_universe
    )

    return DatasetStats(
        dataset=dataset_name,
        cmd_mode=cmd_mode,
        vocab_size_train_cmd=len(train_cmds),
        vocab_size_train_path=len(train_paths),
        novel_test_cmds=novel_test_cmds,
        novel_test_paths=novel_test_paths,
        n_test_process=len(test_procs_eval),
        n_excluded=len(excluded_ids),
        n_attack=len(test_attacks),
        n_contaminated=len(test_contams),
        n_flagged=len(flagged_eval),
        metrics=metrics,
    )


def print_report(stats: DatasetStats):
    """Print formatted analysis results."""
    metrics = stats.metrics

    print(f"\n{'=' * 60}")
    print(f"Dataset: {stats.dataset} Evaluation Report ({stats.cmd_mode} command mode)")
    print(f"{'=' * 60}")
    print("Stats:")
    print(f"  Process Nodes (Test): {stats.n_test_process}")
    if stats.n_excluded:
        print(f"  Excluded from metrics: {stats.n_excluded}")
    print(
        f"  Training Vocab:       {stats.vocab_size_train_cmd} cmds, {stats.vocab_size_train_path} paths"
    )
    print(
        f"  Novel in Test:        {stats.novel_test_cmds} cmds, {stats.novel_test_paths} paths"
    )
    print(
        f"  Nodes Flagged:        {stats.n_flagged} ({stats.n_flagged / stats.n_test_process * 100:.1f}%)"
    )

    print("\nAttack-chain metrics")
    print(
        f"  (Excludes {stats.n_contaminated} contaminated nodes from metric accounting)"
    )
    print(f"  Precision: {metrics.precision:.4f}")
    print(f"  Recall:    {metrics.recall:.4f}")
    print(f"  F1 Score:  {metrics.f1_score:.4f}")
    print(f"  FPR:       {metrics.fpr:.4f}")
    print(f"  MCC:       {metrics.mcc:.4f}")
    print(f"{'=' * 60}\n")


def save_csv_results(results: list[DatasetStats], output_path: str):
    """Serialize results to CSV."""
    if not results:
        return

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    flat_results = []
    for r in results:
        row = {
            "dataset": r.dataset,
            "cmd_mode": r.cmd_mode,
            "train_cmd_vocab": r.vocab_size_train_cmd,
            "train_path_vocab": r.vocab_size_train_path,
            "test_novel_cmds": r.novel_test_cmds,
            "test_novel_paths": r.novel_test_paths,
            "n_test_process": r.n_test_process,
            "n_excluded": r.n_excluded,
            "n_attack": r.n_attack,
            "n_contaminated": r.n_contaminated,
            "n_flagged": r.n_flagged,
            "precision": r.metrics.precision,
            "recall": r.metrics.recall,
            "f1": r.metrics.f1_score,
            "fpr": r.metrics.fpr,
            "mcc": r.metrics.mcc,
        }
        flat_results.append(row)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=flat_results[0].keys())
        writer.writeheader()
        writer.writerows(flat_results)

    logger.info(f"Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Allowlist diagnostic evaluator",
        epilog="Flags process nodes with executables or paths not seen in training set.",
    )
    parser.add_argument("dataset", nargs="?", help="Target dataset (e.g., CADETS_E3)")
    parser.add_argument(
        "--all-datasets", action="store_true", help="Run on all defined datasets"
    )
    parser.add_argument(
        "--output", default="outputs/allowlist_diagnostic.csv", help="CSV result path"
    )
    parser.add_argument(
        "--cmd-mode",
        choices=["executable", "full", "both"],
        default="both",
        help=(
            "Command allowlist feature: executable name, full command line, or both "
            "(default: both)."
        ),
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "DARPA",
        help="Directory containing dataset folders (default: PROJECT_ROOT/data/DARPA)",
    )
    parser.add_argument(
        "--ground-truth-dir",
        type=Path,
        default=PROJECT_ROOT
        / "ground_truth"
        / "reapr-ground-truth"
        / "darpa-tc-engagement3",
        help="Directory containing ground truth CSV files (default: PROJECT_ROOT/ground_truth/reapr-ground-truth/darpa-tc-engagement3)",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=PROJECT_ROOT / "configs" / "datasets",
        help="Directory containing dataset config YAML files (default: PROJECT_ROOT/configs/datasets)",
    )

    args = parser.parse_args()

    if not args.all_datasets and not args.dataset:
        parser.error("Specify a dataset or use --all-datasets")

    target_datasets = (
        get_all_datasets(args.config_dir) if args.all_datasets else [args.dataset]
    )

    modes = ["executable", "full"] if args.cmd_mode == "both" else [args.cmd_mode]
    results_by_mode: dict[str, list[DatasetStats]] = {mode: [] for mode in modes}

    for mode in modes:
        for ds_name in target_datasets:
            try:
                config = load_dataset_config(ds_name, args.config_dir)
                stats = analyze_dataset(
                    ds_name, config, args.data_dir, args.ground_truth_dir, cmd_mode=mode
                )
                if stats:
                    print_report(stats)
                    results_by_mode[mode].append(stats)
            except Exception as e:
                logger.error(f"Failed to analyze {ds_name} ({mode}): {e}")

    output_path = Path(args.output)
    for mode, results in results_by_mode.items():
        if not results:
            continue
        mode_output = output_path
        if args.cmd_mode == "both" and mode == "full":
            mode_output = output_path.with_name(
                f"{output_path.stem}_full_command{output_path.suffix}"
            )
        save_csv_results(results, str(mode_output))


if __name__ == "__main__":
    main()
