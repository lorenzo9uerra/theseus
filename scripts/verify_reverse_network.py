#!/usr/bin/env python3
"""Analyze network flow structure to validate token ordering hypothesis.

Computes entropy and cardinality metrics for source/destination attributes
to determine if reversing token weights improves information extraction.
"""

import argparse
import sys
from pathlib import Path

import polars as pl
from scipy.stats import entropy

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.utils import read_node_table  # noqa: E402

DATASETS = ["CADETS_E3", "THEIA_E3", "FIVEDIRECTIONS_E3", "TRACE_E3"]
EPHEMERAL_PORT_THRESHOLD = 10000


def calculate_shannon_entropy(series: pl.Series) -> float:
    """Compute Shannon entropy in bits for a categorical series."""
    if series.len() == 0:
        return 0.0
    value_counts = series.value_counts()
    counts = value_counts["count"].to_numpy()
    probs = counts / counts.sum()
    return entropy(probs, base=2)


def analyze_netflow_structure(dataset_name: str):
    """Analyze entropy distribution across netflow attributes."""
    print(f"\n{'=' * 80}")
    print(f"Dataset: {dataset_name}")
    print("=" * 80)

    data_dir = PROJECT_ROOT / "data" / "DARPA" / dataset_name
    df = read_node_table(str(data_dir), "netflow_node_table")

    if df is None:
        print(f"Skipping {dataset_name}: netflow table not found")
        return

    df = df.drop_nulls(subset=["src_port", "dst_port", "src_addr", "dst_addr"])
    total_rows = len(df)
    print(f"Total flows: {total_rows:,}")

    columns_map = {
        "Source (Head)": [("src_addr", "IP"), ("src_port", "Port")],
        "Destination (Tail)": [("dst_addr", "IP"), ("dst_port", "Port")],
    }

    results = {}

    print(f"\n{'-' * 80}")
    print(f"{'Position':<22} {'Attr':<8} {'Unique':>12} {'%':>8} {'Entropy':>14}")
    print("-" * 80)

    for position, cols in columns_map.items():
        for col_name, attr_type in cols:
            series = df[col_name]
            n_unique = series.n_unique()
            pct_unique = 100 * n_unique / total_rows
            ent = calculate_shannon_entropy(series)

            print(
                f"{position:<22} {attr_type:<8} {n_unique:>12,} {pct_unique:>7.2f}% {ent:>13.4f}"
            )
            results[f"{position}_{attr_type}"] = ent

    src_ephemeral = df.filter(pl.col("src_port") > EPHEMERAL_PORT_THRESHOLD).height
    dst_ephemeral = df.filter(pl.col("dst_port") > EPHEMERAL_PORT_THRESHOLD).height

    print(f"\n{'-' * 80}")
    print(f"Ephemeral ports (>{EPHEMERAL_PORT_THRESHOLD})")
    print("-" * 80)
    print(f"Source:      {100 * src_ephemeral / total_rows:5.2f}%")
    print(f"Destination: {100 * dst_ephemeral / total_rows:5.2f}%")

    src_port_ent = results["Source (Head)_Port"]
    dst_port_ent = results["Destination (Tail)_Port"]

    print(f"\n{'-' * 80}")
    print("Hypothesis validation")
    print("-" * 80)

    if src_port_ent > dst_port_ent:
        ratio = src_port_ent / (dst_port_ent + 1e-9)
        print(f"CONFIRMED: Source port entropy {ratio:.1f}x higher than destination")
    else:
        print("REJECTED: Destination port entropy >= source port entropy")


def main():
    parser = argparse.ArgumentParser(description="Analyze netflow token variance")
    parser.add_argument("--dataset", type=str, help="Specific dataset to analyze")
    parser.add_argument("--all", action="store_true", help="Analyze all datasets")
    args = parser.parse_args()

    if args.dataset:
        target_datasets = [args.dataset]
    elif args.all:
        target_datasets = DATASETS
    else:
        parser.print_help()
        return

    for ds in target_datasets:
        try:
            analyze_netflow_structure(ds)
        except Exception as e:
            print(f"Error analyzing {ds}: {e}")


if __name__ == "__main__":
    main()
