#!/usr/bin/env python3
"""This script implements the Binary Allowlist baseline described in the Theseus paper."""

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

    # Vocabulary counts
    vocab_size_train_cmd: int = 0
    vocab_size_train_path: int = 0
    novel_test_cmds: int = 0
    novel_test_paths: int = 0

    # Node counts
    n_test_process: int = 0
    n_attack: int = 0
    n_contaminated: int = 0
    n_flagged: int = 0

    # Metrics
    strict: EvaluationMetrics = field(default_factory=EvaluationMetrics)
    causal: EvaluationMetrics = field(default_factory=EvaluationMetrics)


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

    # Compute metrics with epsilon for numerical stability
    eps = 1e-12
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    fpr = fp / (fp + tn + eps)

    # Matthews Correlation Coefficient
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
    dataset_name: str, ground_truth_dir: Path
) -> tuple[set[str], set[str]]:
    """
    Parses ground truth CSV to retrieve Attack and Contaminated UUIDs.
    Returns: (attack_uuids, contaminated_uuids)
    """
    gt_path = ground_truth_dir / f"{dataset_name.lower()}_labels.csv"

    if not gt_path.exists():
        logger.warning(f"Ground truth file missing: {gt_path}")
        return set(), set()

    attack_uuids = set()
    contaminated_uuids = set()

    with open(gt_path, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or len(row) < 4:
                continue

            cleaned_row = [c.strip() for c in row]
            # Skip header or malformed rows
            if (
                cleaned_row[0].lower() == "attack_chain"
                or cleaned_row[1].lower() == "uuid"
            ):
                continue

            uuid = cleaned_row[1]
            label = cleaned_row[-1].lower()

            if not uuid:
                continue

            if label == "attack":
                attack_uuids.add(uuid)
            elif label == "contaminated":
                contaminated_uuids.add(uuid)

    return attack_uuids, contaminated_uuids


def load_process_metadata(
    dataset_name: str, data_dir: Path
) -> tuple[dict[str, tuple[str, str]], dict[str, str]]:
    """Loads process node table. Returns (index_id -> (path, cmd), uuid -> index_id)."""
    # Try Parquet first, fall back to CSV
    parquet_path = data_dir / dataset_name / "process_node_table.parquet"
    csv_path = data_dir / dataset_name / "process_node_table.csv"

    if parquet_path.exists():
        df = pl.read_parquet(
            parquet_path, columns=["node_uuid", "index_id", "path", "cmd"]
        )
    elif csv_path.exists():
        df = pl.read_csv(csv_path, columns=["node_uuid", "index_id", "path", "cmd"])
    else:
        logger.warning(f"Process table missing: {csv_path}")
        return {}, {}

    # Fill nulls and collect
    rows = df.with_columns(
        [pl.col("path").fill_null(""), pl.col("cmd").fill_null("")]
    ).rows(named=True)

    meta_map = {str(r["index_id"]): (r["path"], r["cmd"]) for r in rows}
    uuid_map = {r["node_uuid"]: str(r["index_id"]) for r in rows}

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

    # Convert timestamp (nanoseconds) to day string and filter
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
    dataset_name: str, config: dict, data_dir: Path, ground_truth_dir: Path
) -> DatasetStats | None:
    """Execute analysis pipeline for a single dataset."""
    logger.info(f"Analyzing {dataset_name}...")

    proc_meta, uuid_to_id = load_process_metadata(dataset_name, data_dir)
    if not proc_meta:
        return None

    try:
        train_nodes = get_active_nodes(
            dataset_name, config["train_days"], config["year_month"], data_dir
        )
        test_nodes_all = get_active_nodes(
            dataset_name, config["test_days"], config["year_month"], data_dir
        )
        # Validation nodes are not neeeded
    except FileNotFoundError as e:
        logger.error(str(e))
        return None

    # Filter strictly to process nodes (only process nodes are considered
    # malicious and present in the ground truth)
    train_procs = train_nodes & proc_meta.keys()
    test_procs = test_nodes_all & proc_meta.keys()

    train_cmds = {proc_meta[n][1] for n in train_procs if proc_meta[n][1]}
    train_paths = {proc_meta[n][0] for n in train_procs if proc_meta[n][0]}

    def get_proc_name(cmd_str):
        if not cmd_str:
            return ""
        # Split by space and take the first part (the executable)
        return cmd_str.split(" ")[0]

    train_cmds = {
        get_proc_name(proc_meta[n][1]) for n in train_procs if proc_meta[n][1]
    }

    # Detection Logic: Flag process if (cmd NOT in train_cmds) OR (path NOT in train_paths)
    flagged_nodes = set()

    # For reporting statistics only
    test_cmds = set()
    test_paths = set()

    for nid in test_procs:
        path, cmd = proc_meta[nid]

        cmd_token = get_proc_name(cmd)

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

    attack_uuids, contam_uuids = load_ground_truth(
        dataset_name.split("_")[0], ground_truth_dir
    )

    attack_ids = {uuid_to_id[u] for u in attack_uuids if u in uuid_to_id}
    contam_ids = {uuid_to_id[u] for u in contam_uuids if u in uuid_to_id}

    # Restrict ground truth to test set process nodes
    test_attacks = attack_ids & test_procs
    test_contams = contam_ids & test_procs
    test_malicious = test_attacks | test_contams

    # Exclude contaminated nodes for Strict Attak Chain evaluation
    strict_universe = test_procs - test_contams
    strict_flagged = flagged_nodes - test_contams
    strict_metrics = compute_classification_metrics(
        strict_flagged, test_attacks, strict_universe
    )

    # Treat both attack and contaminated nodes as Positive class
    causal_metrics = compute_classification_metrics(
        flagged_nodes, test_malicious, test_procs
    )

    return DatasetStats(
        dataset=dataset_name,
        vocab_size_train_cmd=len(train_cmds),
        vocab_size_train_path=len(train_paths),
        novel_test_cmds=novel_test_cmds,
        novel_test_paths=novel_test_paths,
        n_test_process=len(test_procs),
        n_attack=len(test_attacks),
        n_contaminated=len(test_contams),
        n_flagged=len(flagged_nodes),
        strict=strict_metrics,
        causal=causal_metrics,
    )


def print_report(stats: DatasetStats):
    """Print formatted analysis results."""
    m_strict = stats.strict
    m_causal = stats.causal

    print(f"\n{'=' * 60}")
    print(f"Dataset: {stats.dataset} Evaluation Report")
    print(f"{'=' * 60}")
    print("Stats:")
    print(f"  Process Nodes (Test): {stats.n_test_process}")
    print(
        f"  Training Vocab:       {stats.vocab_size_train_cmd} cmds, {stats.vocab_size_train_path} paths"
    )
    print(
        f"  Novel in Test:        {stats.novel_test_cmds} cmds, {stats.novel_test_paths} paths"
    )
    print(
        f"  Nodes Flagged:        {stats.n_flagged} ({stats.n_flagged / stats.n_test_process * 100:.1f}%)"
    )

    print("\nParadigm 1: Strict (Attack Chain)")
    print(f"  (Excludes {stats.n_contaminated} contaminated nodes)")
    print(f"  Precision: {m_strict.precision:.4f}")
    print(f"  Recall:    {m_strict.recall:.4f}")
    print(f"  F1 Score:  {m_strict.f1_score:.4f}")
    print(f"  FPR:       {m_strict.fpr:.4f}")
    print(f"  MCC:       {m_strict.mcc:.4f}")

    print("\nParadigm 2: Causal Scope (All Malicious)")
    print(f"  (Includes {stats.n_contaminated} contaminated nodes as Positive)")
    print(f"  Precision: {m_causal.precision:.4f}")
    print(f"  Recall:    {m_causal.recall:.4f}")
    print(f"  F1 Score:  {m_causal.f1_score:.4f}")
    print(f"  FPR:       {m_causal.fpr:.4f}")
    print(f"  MCC:       {m_causal.mcc:.4f}")
    print(f"{'=' * 60}\n")


def save_csv_results(results: list[DatasetStats], output_path: str):
    """Serialize results to CSV."""
    if not results:
        return

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # Flatten dataclass for CSV
    flat_results = []
    for r in results:
        row = {
            "dataset": r.dataset,
            "train_cmd_vocab": r.vocab_size_train_cmd,
            "train_path_vocab": r.vocab_size_train_path,
            "test_novel_cmds": r.novel_test_cmds,
            "test_novel_paths": r.novel_test_paths,
            "n_test_process": r.n_test_process,
            "n_attack": r.n_attack,
            "n_contaminated": r.n_contaminated,
            "n_flagged": r.n_flagged,
            # Strict
            "strict_precision": r.strict.precision,
            "strict_recall": r.strict.recall,
            "strict_f1": r.strict.f1_score,
            "strict_mcc": r.strict.mcc,
            # Causal
            "causal_precision": r.causal.precision,
            "causal_recall": r.causal.recall,
            "causal_f1": r.causal.f1_score,
            "causal_mcc": r.causal.mcc,
        }
        flat_results.append(row)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=flat_results[0].keys())
        writer.writeheader()
        writer.writerows(flat_results)

    logger.info(f"Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Naive Binary Allowlist baseline evaluator",
        epilog="Flags process nodes with executables or paths not seen in training set.",
    )
    parser.add_argument("dataset", nargs="?", help="Target dataset (e.g., CADETS_E3)")
    parser.add_argument(
        "--all-datasets", action="store_true", help="Run on all defined datasets"
    )
    parser.add_argument(
        "--output", default="outputs/vocab_analysis.csv", help="CSV result path"
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

    results = []
    for ds_name in target_datasets:
        try:
            config = load_dataset_config(ds_name, args.config_dir)
            stats = analyze_dataset(
                ds_name, config, args.data_dir, args.ground_truth_dir
            )
            if stats:
                print_report(stats)
                results.append(stats)
        except Exception as e:
            logger.error(f"Failed to analyze {ds_name}: {e}")

    if results:
        save_csv_results(results, args.output)


if __name__ == "__main__":
    main()
