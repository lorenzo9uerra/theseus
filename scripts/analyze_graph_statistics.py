#!/usr/bin/env python3
"""Compute graph statistics for provenance graph snapshots.

Analyzes cached graph files and outputs statistics:
- Number of graph snapshots per dataset and split
- Total node and edge instances across snapshots
- Unique entities observed across snapshots
- Median node and edge count per snapshot
- Node type distribution (Process/File/Netflow)

Usage:
    python scripts/analyze_graph_statistics.py --seed 83811
    python scripts/analyze_graph_statistics.py --cache_dir ./cache --seed 83811
"""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import median

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATASETS = ["CADETS_E3", "FIVEDIRECTIONS_E3", "TRACE_E3", "THEIA_E3"]
NODE_TYPE_NAMES = ["process", "file", "netflow"]


@dataclass
class DatasetStats:
    """Statistics for a single dataset."""

    name: str
    num_graphs: int
    train_graphs: int
    val_graphs: int
    test_graphs: int
    total_nodes: int
    total_edges: int
    unique_nodes: int
    unique_process_nodes: int
    unique_file_nodes: int
    unique_netflow_nodes: int
    median_nodes: float
    median_edges: float
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

    process_count = int((type_indices == 0).sum().item())
    file_count = int((type_indices == 1).sum().item())
    netflow_count = int((type_indices == 2).sum().item())

    return process_count, file_count, netflow_count


def _median_or_zero(values: list[int]) -> float:
    return float(median(values)) if values else 0.0


def analyze_dataset(
    dataset_name: str, cache_dir: str, seed: int
) -> DatasetStats | None:
    """Load cached graphs and compute statistics."""
    dataset_key = dataset_name.lower()
    exact_paths = [
        Path(cache_dir) / f"graph_{dataset_key}_fused_edge_node_deg_{seed}_cache.pt",
        Path(cache_dir) / f"graph_{dataset_key}_{seed}_cache.pt",
    ]
    glob_paths = [
        f"graph_{dataset_key}_fused_edge_node_deg_*_{seed}_cache.pt",
        f"graph_{dataset_key}_*_{seed}_cache.pt",
    ]
    cache_paths = exact_paths + [
        path for pattern in glob_paths for path in sorted(Path(cache_dir).glob(pattern))
    ]
    cache_path = next((path for path in cache_paths if path.exists()), None)

    if cache_path is None:
        checked = ", ".join(str(path) for path in exact_paths + glob_paths)
        print(f"Cache not found for {dataset_name} (seed={seed}); checked {checked}")
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
    split_counts = {"train": 0, "val": 0, "test": 0}
    node_counts: list[int] = []
    edge_counts: list[int] = []
    unique_nodes: set[int] = set()
    unique_process_nodes: set[int] = set()
    unique_file_nodes: set[int] = set()
    unique_netflow_nodes: set[int] = set()

    for split in ["train", "val", "test"]:
        split_graphs = graphs.get(split, [])
        split_counts[split] = len(split_graphs)
        for graph in split_graphs:
            num_graphs += 1
            num_nodes = int(graph.x.shape[0])
            num_edges = int(graph.edge_index.shape[1])
            total_nodes += num_nodes
            total_edges += num_edges
            node_counts.append(num_nodes)
            edge_counts.append(num_edges)

            p, f, n = count_node_types(graph.x)
            process_nodes += p
            file_nodes += f
            netflow_nodes += n

            if not hasattr(graph, "original_n_id"):
                raise ValueError(
                    f"{cache_path} contains a graph without original_n_id; "
                    "cannot compute unique entity counts across snapshots"
                )

            node_ids = graph.original_n_id.cpu().tolist()
            type_indices = graph.x[:, :3].argmax(dim=1).cpu().tolist()
            for node_id, type_index in zip(node_ids, type_indices, strict=True):
                node_id = int(node_id)
                unique_nodes.add(node_id)
                if type_index == 0:
                    unique_process_nodes.add(node_id)
                elif type_index == 1:
                    unique_file_nodes.add(node_id)
                elif type_index == 2:
                    unique_netflow_nodes.add(node_id)

    return DatasetStats(
        name=dataset_name,
        num_graphs=num_graphs,
        train_graphs=split_counts["train"],
        val_graphs=split_counts["val"],
        test_graphs=split_counts["test"],
        total_nodes=total_nodes,
        total_edges=total_edges,
        unique_nodes=len(unique_nodes),
        unique_process_nodes=len(unique_process_nodes),
        unique_file_nodes=len(unique_file_nodes),
        unique_netflow_nodes=len(unique_netflow_nodes),
        median_nodes=_median_or_zero(node_counts),
        median_edges=_median_or_zero(edge_counts),
        process_nodes=process_nodes,
        file_nodes=file_nodes,
        netflow_nodes=netflow_nodes,
    )


def print_text_table(stats_list: list[DatasetStats]) -> None:
    """Print statistics in plain text format."""
    print("\n" + "=" * 132)
    print(
        f"{'Dataset':<16} {'Snapshots':>9} {'Train/Val/Test':>14} "
        f"{'Node Inst.':>12} {'Edge Inst.':>14} {'Unique Ent.':>12} "
        f"{'Median N/E':>17} {'P/F/N Distribution':>20}"
    )
    print("=" * 132)

    for s in stats_list:
        dist = f"{s.process_pct:.0f}% / {s.file_pct:.0f}% / {s.netflow_pct:.0f}%"
        split_counts = f"{s.train_graphs}/{s.val_graphs}/{s.test_graphs}"
        median_counts = f"{s.median_nodes:,.0f} / {s.median_edges:,.0f}"
        print(
            f"{s.display_name:<16} {s.num_graphs:>9,} {split_counts:>14} "
            f"{s.total_nodes:>12,} {s.total_edges:>14,} {s.unique_nodes:>12,} "
            f"{median_counts:>17} {dist:>20}"
        )

    print("=" * 132)

    print("\nUnique entity breakdown (P/F/N):")
    for s in stats_list:
        print(
            f"  {s.display_name:<16} "
            f"{s.unique_process_nodes:,} / {s.unique_file_nodes:,} / {s.unique_netflow_nodes:,}"
        )


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
