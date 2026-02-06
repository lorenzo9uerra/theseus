#!/usr/bin/env python3
"""Analyze feature completeness and semantic richness of node attributes.

Computes two metrics per dataset:
- Completeness (%): filled attribute slots / total possible slots
- Entropy (bits): weighted average of corpus-level token entropy

The entropy is computed at the corpus level (bag of all tokens in a field),
then aggregated by weighting each field by its non-empty count. This measures
the average information density of feature vectors passed to the model.
"""

import argparse
import os
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.utils import read_node_table, tokenize_node_description  # noqa: E402

DATASETS = ["CADETS_E3", "THEIA_E3", "FIVEDIRECTIONS_E3", "TRACE_E3"]


@dataclass
class FieldStats:
    field_name: str
    total_count: int
    non_empty_count: int
    unique_values: int
    avg_tokens: float
    median_tokens: float
    max_tokens: int
    total_tokens: int
    unique_tokens: int
    corpus_entropy: float

    @property
    def completeness(self) -> float:
        if self.total_count == 0:
            return 0.0
        return 100 * self.non_empty_count / self.total_count


@dataclass
class NodeTypeStats:
    node_type: str
    total_nodes: int
    fields: list[FieldStats]

    @property
    def total_slots_possible(self) -> int:
        return sum(f.total_count for f in self.fields)

    @property
    def total_slots_filled(self) -> int:
        return sum(f.non_empty_count for f in self.fields)

    @property
    def completeness(self) -> float:
        if self.total_slots_possible == 0:
            return 0.0
        return 100 * self.total_slots_filled / self.total_slots_possible

    @property
    def type_entropy(self) -> float:
        """Unweighted average of field entropies."""
        if not self.fields:
            return 0.0
        return sum(f.corpus_entropy for f in self.fields) / len(self.fields)


@dataclass
class DatasetStats:
    """Aggregated statistics for an entire dataset."""

    dataset: str
    process: NodeTypeStats | None
    file: NodeTypeStats | None
    netflow: NodeTypeStats | None

    @property
    def node_types(self) -> list[NodeTypeStats]:
        return [nt for nt in [self.process, self.file, self.netflow] if nt]

    @property
    def total_nodes(self) -> int:
        return sum(nt.total_nodes for nt in self.node_types)

    @property
    def overall_completeness(self) -> float:
        """Filled slots / possible slots across all node types."""
        total_possible = sum(nt.total_slots_possible for nt in self.node_types)
        total_filled = sum(nt.total_slots_filled for nt in self.node_types)
        if total_possible == 0:
            return 0.0
        return 100 * total_filled / total_possible

    @property
    def overall_entropy(self) -> float:
        """
        Measures average bits per filled attribute slot passed to the model.
        """
        total_filled = 0
        weighted_sum = 0.0
        for nt in self.node_types:
            for f in nt.fields:
                weighted_sum += f.corpus_entropy * f.non_empty_count
                total_filled += f.non_empty_count
        if total_filled == 0:
            return 0.0
        return weighted_sum / total_filled


def corpus_token_entropy(token_counts: Counter) -> float:
    if not token_counts:
        return 0.0
    total = sum(token_counts.values())
    probs = np.array(list(token_counts.values())) / total
    return float(-np.sum(probs * np.log2(probs)))


def analyze_field(
    df: pl.DataFrame, field_name: str, tokenizer_type: str
) -> FieldStats | None:
    """Compute corpus-level token entropy for a single field."""
    if field_name not in df.columns:
        return None

    total_count = len(df)
    non_empty_mask = df[field_name].is_not_null() & (
        df[field_name].cast(pl.Utf8).str.strip_chars() != ""
    )
    non_empty = df.filter(non_empty_mask)[field_name].cast(pl.Utf8)
    n = len(non_empty)

    if n == 0:
        return FieldStats(
            field_name=field_name,
            total_count=total_count,
            non_empty_count=0,
            unique_values=0,
            avg_tokens=0.0,
            median_tokens=0.0,
            max_tokens=0,
            total_tokens=0,
            unique_tokens=0,
            corpus_entropy=0.0,
        )

    unique_values = non_empty.n_unique()
    all_strings = non_empty.to_list()

    token_counts: Counter = Counter()
    tokens_per_string = []

    for text in all_strings:
        tokens = [t[0] for t in tokenize_node_description(text, tokenizer_type)]
        tokens_per_string.append(len(tokens))
        token_counts.update(tokens)

    sorted_counts = sorted(tokens_per_string)

    return FieldStats(
        field_name=field_name,
        total_count=total_count,
        non_empty_count=n,
        unique_values=unique_values,
        avg_tokens=sum(tokens_per_string) / n,
        median_tokens=float(sorted_counts[n // 2]),
        max_tokens=max(tokens_per_string),
        total_tokens=sum(token_counts.values()),
        unique_tokens=len(token_counts),
        corpus_entropy=corpus_token_entropy(token_counts),
    )


def analyze_netflow(df: pl.DataFrame) -> FieldStats | None:
    """Analyze netflow using combined 'src_addr src_port dst_addr dst_port' format."""
    cols = ["src_addr", "src_port", "dst_addr", "dst_port"]
    if not all(c in df.columns for c in cols):
        return None

    total_count = len(df)
    combined = (
        df.select(cols)
        .with_columns([pl.col(c).cast(pl.Utf8).fill_null("") for c in cols])
        .with_columns(pl.concat_str(cols, separator=" ").alias("desc"))
    )

    non_empty_mask = combined["desc"].str.strip_chars() != ""
    non_empty = combined.filter(non_empty_mask)["desc"]
    n = len(non_empty)

    if n == 0:
        return FieldStats(
            field_name="netflow",
            total_count=total_count,
            non_empty_count=0,
            unique_values=0,
            avg_tokens=0.0,
            median_tokens=0.0,
            max_tokens=0,
            total_tokens=0,
            unique_tokens=0,
            corpus_entropy=0.0,
        )

    unique_values = non_empty.n_unique()
    all_strings = non_empty.to_list()

    token_counts: Counter = Counter()
    tokens_per_string = []

    for text in all_strings:
        tokens = [t[0] for t in tokenize_node_description(text, "netflow")]
        tokens_per_string.append(len(tokens))
        token_counts.update(tokens)

    sorted_counts = sorted(tokens_per_string)

    return FieldStats(
        field_name="netflow",
        total_count=total_count,
        non_empty_count=n,
        unique_values=unique_values,
        avg_tokens=sum(tokens_per_string) / n,
        median_tokens=float(sorted_counts[n // 2]),
        max_tokens=max(tokens_per_string),
        total_tokens=sum(token_counts.values()),
        unique_tokens=len(token_counts),
        corpus_entropy=corpus_token_entropy(token_counts),
    )


def analyze_node_type(
    dataset: str, node_type: str, table_name: str, fields: list[str]
) -> NodeTypeStats | None:
    """Load Parquet/CSV and analyze specified fields."""
    data_dir = os.path.join(PROJECT_ROOT, "data", "DARPA", dataset)
    df = read_node_table(data_dir, table_name)
    if df is None:
        return None

    if node_type == "netflow":
        stats = analyze_netflow(df)
        field_stats = [stats] if stats else []
    else:
        tokenizer_map = {
            "path": "file" if node_type == "file" else "process",
            "cmd": "process",
        }
        field_stats = [
            s
            for s in (
                analyze_field(df, f, tokenizer_map.get(f, node_type)) for f in fields
            )
            if s
        ]

    if not field_stats:
        return None

    return NodeTypeStats(node_type=node_type, total_nodes=len(df), fields=field_stats)


def analyze_dataset(dataset: str, verbose: bool = True) -> DatasetStats:
    """Analyze all node types in a dataset."""
    if verbose:
        print(f"\n{'=' * 80}")
        print(f"Dataset: {dataset}")
        print("=" * 80)

    process = analyze_node_type(
        dataset, "process", "process_node_table", ["path", "cmd"]
    )
    file = analyze_node_type(dataset, "file", "file_node_table", ["path"])
    netflow = analyze_node_type(dataset, "netflow", "netflow_node_table", [])

    if verbose:
        for nt in [process, file, netflow]:
            if not nt:
                continue
            print(f"\n  {nt.node_type.upper()} ({nt.total_nodes:,} nodes)")
            print(f"  Completeness: {nt.completeness:.1f}%")
            for f in nt.fields:
                print(f"    {f.field_name}:")
                print(
                    f"      Filled: {f.non_empty_count:,}/{f.total_count:,} ({f.completeness:.1f}%)"
                )
                print(f"      Unique values: {f.unique_values:,}")
                if f.non_empty_count > 0:
                    print(
                        f"      Tokens: avg={f.avg_tokens:.2f}, median={f.median_tokens:.0f}, max={f.max_tokens}"
                    )
                    print(
                        f"      Vocabulary: {f.unique_tokens:,} unique / {f.total_tokens:,} total"
                    )
                    print(f"      Corpus Entropy: {f.corpus_entropy:.2f} bits")

    return DatasetStats(dataset=dataset, process=process, file=file, netflow=netflow)


def print_summary(results: list[DatasetStats]):
    """Print aggregated tables for paper."""
    print("\n" + "=" * 90)
    print("DATASET FEATURE RICHNESS")
    print("=" * 90)
    print(f"{'Dataset':<20} {'Nodes':>12} {'Completeness':>16} {'Entropy':>16}")
    print("-" * 90)
    for r in results:
        print(
            f"{r.dataset:<20} {r.total_nodes:>12,} "
            f"{r.overall_completeness:>15.1f}% {r.overall_entropy:>14.2f} bits"
        )
    print("=" * 90)
    print("\nCompleteness = filled_slots / possible_slots")
    print("Entropy = sum(H_field * filled_count) / sum(filled_count)")

    print("\n" + "=" * 90)
    print("COMPLETENESS BY NODE TYPE")
    print("=" * 90)
    print(f"{'Dataset':<20} {'Process':>16} {'File':>16} {'Netflow':>16}")
    print("-" * 90)
    for r in results:
        p = f"{r.process.completeness:.1f}%" if r.process else "N/A"
        f = f"{r.file.completeness:.1f}%" if r.file else "N/A"
        n = f"{r.netflow.completeness:.1f}%" if r.netflow else "N/A"
        print(f"{r.dataset:<20} {p:>16} {f:>16} {n:>16}")
    print("=" * 90)

    print("\n" + "=" * 90)
    print("ENTROPY BY NODE TYPE (avg of field entropies)")
    print("=" * 90)
    print(f"{'Dataset':<20} {'Process':>16} {'File':>16} {'Netflow':>16}")
    print("-" * 90)
    for r in results:
        p = f"{r.process.type_entropy:.2f}" if r.process else "N/A"
        f = f"{r.file.type_entropy:.2f}" if r.file else "N/A"
        n = f"{r.netflow.type_entropy:.2f}" if r.netflow else "N/A"
        print(f"{r.dataset:<20} {p:>16} {f:>16} {n:>16}")
    print("=" * 90)

    print("\n" + "=" * 90)
    print("FIELD-LEVEL DETAILS")
    print("=" * 90)
    print(f"{'Dataset':<16} {'Field':<12} {'Filled':>12} {'Vocab':>10} {'Entropy':>12}")
    print("-" * 90)
    for r in results:
        for nt in r.node_types:
            for fld in nt.fields:
                if fld.non_empty_count > 0:
                    print(
                        f"{r.dataset:<16} {fld.field_name:<12} {fld.non_empty_count:>12,} "
                        f"{fld.unique_tokens:>10,} {fld.corpus_entropy:>10.2f} bits"
                    )
    print("=" * 90)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze feature richness of DARPA datasets"
    )
    parser.add_argument("dataset", nargs="?", help="Dataset name (e.g., THEIA_E3)")
    parser.add_argument("--all", action="store_true", help="Analyze all datasets")
    parser.add_argument(
        "--quiet", "-q", action="store_true", help="Only show summary tables"
    )
    args = parser.parse_args()

    if not args.dataset and not args.all:
        parser.print_help()
        sys.exit(1)

    datasets = DATASETS if args.all else [args.dataset]
    results = []

    for ds in datasets:
        try:
            results.append(analyze_dataset(ds, verbose=not args.quiet))
        except Exception as e:
            print(f"Error analyzing {ds}: {e}")

    if len(results) > 1:
        print_summary(results)


if __name__ == "__main__":
    main()
