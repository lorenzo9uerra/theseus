#!/usr/bin/env python3
"""Compute graph statistics for provenance graph snapshots.

Analyzes cached graph files and outputs statistics:
- Number of graphs per dataset
- Total nodes and edges
- Node type distribution (Process/File/Netflow)

Usage:
    python scripts/analyze_graph_statistics.py --seed 83811
    python scripts/analyze_graph_statistics.py --cache_dir ./cache --seed 83811
"""

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATASETS = ["CADETS_E3", "FIVEDIRECTIONS_E3", "THEIA_E3", "TRACE_E3"]
NODE_TYPE_NAMES = ["process", "file", "netflow"]


@dataclass
class DatasetStats:
    """Statistics for a single dataset."""

    name: str
    num_graphs: int
    total_nodes: int
    total_edges: int
    process_nodes: int
    file_nodes: int
    netflow_nodes: int

    @property
    def process_pct(self) -> float:
        return 100 * self.process_nodes / self.total_nodes if self.total_nodes else 0

    @property
    def file_pct(self) -> float:
        return 100 * self.file_nodes / self.total_nodes if self.total_nodes else 0

    @property
    def netflow_pct(self) -> float:
        return 100 * self.netflow_nodes / self.total_nodes if self.total_nodes else 0

    @property
    def display_name(self) -> str:
        mapping = {
            "CADETS_E3": "Cadets",
            "FIVEDIRECTIONS_E3": "Fivedirections",
            "THEIA_E3": "Theia (T3)",
            "TRACE_E3": "Trace",
        }
        return mapping.get(self.name, self.name)


def count_node_types(node_features: torch.Tensor) -> tuple[int, int, int]:
    """Extract node type counts from one-hot encoded features."""
    type_onehot = node_features[:, :3]
    type_indices = type_onehot.argmax(dim=1)

    process_count = (type_indices == 0).sum().item()
    file_count = (type_indices == 1).sum().item()
    netflow_count = (type_indices == 2).sum().item()

    return process_count, file_count, netflow_count


def analyze_dataset(
    dataset_name: str, cache_dir: str, seed: int
) -> DatasetStats | None:
    """Load cached graphs and compute statistics."""
    cache_pattern = f"graph_{dataset_name.lower()}{'' if 'THEIA' in dataset_name else '_fused_edge_node_deg'}_{seed}_cache.pt"
    cache_path = os.path.join(cache_dir, cache_pattern)

    if not os.path.exists(cache_path):
        cache_pattern = f"graph_{dataset_name.lower()}_{seed}_cache.pt"
        cache_path = os.path.join(cache_dir, cache_pattern)
        if not os.path.exists(cache_path):
            print(f"Cache not found for {dataset_name} (seed={seed})")
            return None

    print(f"Loading {cache_path}...")
    cache_data = torch.load(cache_path, map_location="cpu", weights_only=False)
    graphs = cache_data.get("graphs", {})

    total_nodes = 0
    total_edges = 0
    process_nodes = 0
    file_nodes = 0
    netflow_nodes = 0
    num_graphs = 0

    for split in ["train", "val", "test"]:
        split_graphs = graphs.get(split, [])
        for graph in split_graphs:
            num_graphs += 1
            total_nodes += graph.x.shape[0]
            total_edges += graph.edge_index.shape[1]

            p, f, n = count_node_types(graph.x)
            process_nodes += p
            file_nodes += f
            netflow_nodes += n

    return DatasetStats(
        name=dataset_name,
        num_graphs=num_graphs,
        total_nodes=total_nodes,
        total_edges=total_edges,
        process_nodes=process_nodes,
        file_nodes=file_nodes,
        netflow_nodes=netflow_nodes,
    )


def print_text_table(stats_list: list[DatasetStats]) -> None:
    """Print statistics in plain text format."""
    print("\n" + "=" * 80)
    print(
        f"{'Dataset':<20} {'Graphs':>8} {'Nodes':>12} {'Edges':>14} {'P/F/N Distribution'}"
    )
    print("=" * 80)

    for s in stats_list:
        dist = f"{s.process_pct:.0f}% / {s.file_pct:.0f}% / {s.netflow_pct:.0f}%"
        print(
            f"{s.display_name:<20} {s.num_graphs:>8,} {s.total_nodes:>12,} {s.total_edges:>14,} {dist}"
        )

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Analyze graph cache statistics")
    parser.add_argument(
        "--cache_dir", type=str, default="./cache", help="Cache directory"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Seed used in cache filename"
    )
    parser.add_argument(
        "--datasets", nargs="+", default=DATASETS, help="Datasets to analyze"
    )
    args = parser.parse_args()

    stats_list = []
    for dataset in args.datasets:
        stats = analyze_dataset(dataset, args.cache_dir, args.seed)
        if stats:
            stats_list.append(stats)

    if not stats_list:
        print("No datasets found.")
        sys.exit(1)

    print_text_table(stats_list)


if __name__ == "__main__":
    main()
