#!/usr/bin/env python3
"""
Benchmark Theseus pipeline with realistic dataset sizes.

Example usage: uv run scripts/benchmark.py CADETS_E3
"""

import gc
import os
import random
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import psutil
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ruff: noqa: E402
from tasks.build_graphs import create_graph_window
from tasks.training import initialize_model
from utils.constants.graph_events import DARPA_TC_EVENTS, NODE_TYPES
from utils.parsers import parse_args, parse_config
from utils.utils import create_one_hot, log

WINDOW_SIZE_MINUTES = 15


@dataclass
class DatasetStatistics:
    """Statistics for a specific dataset's graph structure."""

    avg_nodes: int
    avg_edges: int
    median_nodes: int
    median_edges: int
    max_nodes: int
    max_edges: int
    node_type_dist: dict[str, float]
    total_graphs: int


DATASET_STATISTICS: dict[str, DatasetStatistics] = {
    "CADETS_E3": DatasetStatistics(
        avg_nodes=690,
        avg_edges=8948,
        median_nodes=513,
        median_edges=7686,
        max_nodes=8295,
        max_edges=46572,
        node_type_dist={"process": 0.33, "file": 0.54, "netflow": 0.13},
        total_graphs=871,
    ),
    "THEIA_E3": DatasetStatistics(
        avg_nodes=4306,
        avg_edges=31677,
        median_nodes=4380,
        median_edges=15100,
        max_nodes=9888,
        max_edges=161360,
        node_type_dist={"process": 0.08, "file": 0.66, "netflow": 0.26},
        total_graphs=316,
    ),
    "FIVEDIRECTIONS_E3": DatasetStatistics(
        avg_nodes=4370,
        avg_edges=13843,
        median_nodes=5430,
        median_edges=11817,
        max_nodes=9972,
        max_edges=101358,
        node_type_dist={"process": 0.07, "file": 0.90, "netflow": 0.03},
        total_graphs=948,
    ),
    "TRACE_E3": DatasetStatistics(
        avg_nodes=5757,
        avg_edges=14384,
        median_nodes=5636,
        median_edges=7456,
        max_nodes=10000,
        max_edges=142902,
        node_type_dist={"process": 0.15, "file": 0.36, "netflow": 0.49},
        total_graphs=559,
    ),
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


class BenchmarkResult:
    def __init__(self, name: str, num_nodes: int = 0, num_edges: int = 0):
        self.name = name
        self.times_ms: list[float] = []
        self.memory_bytes: list[int] = []
        self.gpu_memory_bytes: list[int] = []
        self.num_nodes = num_nodes
        self.num_edges = num_edges

    def add_measurement(
        self, time_ms: float, memory_bytes: int, gpu_memory_bytes: int = 0
    ):
        self.times_ms.append(time_ms)
        self.memory_bytes.append(memory_bytes)
        self.gpu_memory_bytes.append(gpu_memory_bytes)

    def summary(self) -> dict:
        if not self.times_ms:
            return {}
        mean_time_ms = statistics.mean(self.times_ms)
        result = {
            "name": self.name,
            "samples": len(self.times_ms),
            "time_ms_mean": mean_time_ms,
            "time_ms_std": statistics.stdev(self.times_ms)
            if len(self.times_ms) > 1
            else 0,
            "time_ms_min": min(self.times_ms),
            "time_ms_max": max(self.times_ms),
            "memory_mb_mean": statistics.mean(self.memory_bytes) / (1024 * 1024),
            "memory_mb_max": max(self.memory_bytes) / (1024 * 1024),
            "gpu_memory_mb_mean": statistics.mean(self.gpu_memory_bytes) / (1024 * 1024)
            if self.gpu_memory_bytes
            else 0,
            "gpu_memory_mb_max": max(self.gpu_memory_bytes) / (1024 * 1024)
            if self.gpu_memory_bytes
            else 0,
        }
        if self.num_nodes > 0 and mean_time_ms > 0:
            result["nodes_per_second"] = (self.num_nodes / mean_time_ms) * 1000
        if self.num_edges > 0 and mean_time_ms > 0:
            result["edges_per_second"] = (self.num_edges / mean_time_ms) * 1000
        return result


def measure_execution(func, *args, device="cpu", **kwargs):
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    process = psutil.Process()
    start_time = time.perf_counter()
    result = func(*args, **kwargs)
    if device == "cuda":
        torch.cuda.synchronize()
    end_time = time.perf_counter()
    peak_memory = process.memory_info().rss

    time_ms = (end_time - start_time) * 1000
    gpu_memory = torch.cuda.max_memory_allocated() if device == "cuda" else 0
    return result, time_ms, peak_memory, gpu_memory


def generate_synthetic_edge_arrays(
    num_nodes: int, num_edges: int, event_types: list[str], base_timestamp: int
) -> tuple:
    """Generate synthetic edge data as numpy arrays matching create_graph_window input."""
    src_ids = np.random.randint(0, num_nodes, size=num_edges, dtype=np.int64)
    dst_ids = np.random.randint(0, num_nodes, size=num_edges, dtype=np.int64)

    mask = src_ids == dst_ids
    while mask.any():
        dst_ids[mask] = np.random.randint(0, num_nodes, size=mask.sum(), dtype=np.int64)
        mask = src_ids == dst_ids

    edge_times = base_timestamp + np.arange(num_edges, dtype=np.int64) * 1000000
    edge_ops = [random.choice(event_types) for _ in range(num_edges)]
    fused_counts = np.ones(num_edges, dtype=np.int32)

    return src_ids, dst_ids, edge_times, edge_ops, fused_counts


def benchmark_embedding_generation(
    node_ids: list[str], config, num_samples: int = 10
) -> BenchmarkResult:
    result = BenchmarkResult("embedding_generation")

    from utils.constants.graph_events import NODE_TYPES
    from utils.utils import create_one_hot

    ntype2onehot = create_one_hot(NODE_TYPES)

    for _ in range(num_samples):

        def generate_embeddings():
            embeddings = {}
            for node_id in node_ids:
                node_type = "process" if random.random() < 0.80 else "file"
                node_type_onehot = ntype2onehot[node_type]
                word2vec_embedding = torch.randn(config.embed_dim, dtype=torch.float32)
                embeddings[node_id] = torch.cat([node_type_onehot, word2vec_embedding])
            return embeddings

        _, time_ms, peak_mem, gpu_mem = measure_execution(
            generate_embeddings, device=config.device
        )
        result.add_measurement(time_ms, peak_mem, gpu_mem)

    return result


def benchmark_graph_construction(
    edge_arrays: tuple,
    day: int,
    config,
    node_embeddings: dict,
    edge_type_to_onehot: dict,
    all_malicious_nodes: set,
    num_samples: int = 10,
) -> BenchmarkResult:
    result = BenchmarkResult("graph_construction")

    for _ in range(num_samples):

        def construct_graph():
            graph, _ = create_graph_window(
                edge_arrays,
                day,
                config,
                node_embeddings,
                edge_type_to_onehot,
                all_malicious_nodes,
            )
            return graph

        _, time_ms, peak_mem, gpu_mem = measure_execution(
            construct_graph, device=config.device
        )
        result.add_measurement(time_ms, peak_mem, gpu_mem)

    return result


@torch.inference_mode()
def benchmark_inference(model, graph, config, num_samples: int = 10) -> BenchmarkResult:
    """Benchmark inference including graph-to-GPU transfer."""
    result = BenchmarkResult("inference")
    device = config.device

    graph_device = graph.to(device=device)
    _ = model(graph_device)
    if device == "cuda":
        torch.cuda.synchronize()
    del graph_device

    for _ in range(num_samples):
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        process = psutil.Process()
        start_time = time.perf_counter()

        graph_device = graph.to(device=device)
        outputs, encoded_target = model(graph_device)
        _ = model.loss(outputs, encoded_target, "none")

        if device == "cuda":
            torch.cuda.synchronize()

        end_time = time.perf_counter()
        peak_memory = process.memory_info().rss

        time_ms = (end_time - start_time) * 1000
        gpu_memory = torch.cuda.max_memory_allocated() if device == "cuda" else 0
        result.add_measurement(time_ms, peak_memory, gpu_memory)

        del graph_device

    return result


@torch.inference_mode()
def benchmark_end_to_end(
    num_nodes: int,
    num_edges: int,
    event_types: list[str],
    base_timestamp: int,
    day: int,
    config,
    edge_type_to_onehot: dict,
    all_malicious_nodes: set,
    model,
    num_samples: int = 10,
) -> BenchmarkResult:
    result = BenchmarkResult("end_to_end")
    device = config.device

    from utils.constants.graph_events import NODE_TYPES
    from utils.utils import create_one_hot

    ntype2onehot = create_one_hot(NODE_TYPES)
    node_ids = [str(i) for i in range(num_nodes)]

    for _ in range(num_samples):
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        process = psutil.Process()
        start_time = time.perf_counter()

        node_embeddings = {}
        for node_id in node_ids:
            node_type = "process" if random.random() < 0.80 else "file"
            node_type_onehot = ntype2onehot[node_type]
            word2vec_embedding = torch.randn(config.embed_dim, dtype=torch.float32)
            node_embeddings[node_id] = torch.cat([node_type_onehot, word2vec_embedding])

        edge_arrays = generate_synthetic_edge_arrays(
            num_nodes, num_edges, event_types, base_timestamp
        )
        graph, _ = create_graph_window(
            edge_arrays,
            day,
            config,
            node_embeddings,
            edge_type_to_onehot,
            all_malicious_nodes,
        )

        graph = graph.to(device=device)
        outputs, encoded_target = model(graph)
        _ = model.loss(outputs, encoded_target, "none")

        if device == "cuda":
            torch.cuda.synchronize()

        end_time = time.perf_counter()
        peak_memory = process.memory_info().rss

        time_ms = (end_time - start_time) * 1000
        gpu_memory = torch.cuda.max_memory_allocated() if device == "cuda" else 0
        result.add_measurement(time_ms, peak_memory, gpu_memory)

    return result


def print_benchmark_results(
    results: list[BenchmarkResult],
    graph_stats: dict | None = None,
    scenario: str = "Synthetic",
):
    print("\n" + "=" * 90)
    print(f"BENCHMARK RESULTS ({scenario})")
    print("=" * 90)

    if graph_stats:
        print("\nGraph Statistics:")
        print(f"  Nodes: {graph_stats.get('num_nodes', 'N/A'):,}")
        print(f"  Edges: {graph_stats.get('num_edges', 'N/A'):,}")
        print(f"  Process nodes: {graph_stats.get('num_process_nodes', 'N/A'):,}")

    print("\n" + "-" * 90)
    print(
        f"{'Stage':<22} {'Time (ms)':<18} {'CPU Mem (MB)':<15} "
        f"{'GPU Mem (MB)':<15} {'Throughput':<18}"
    )
    print(
        f"{'':22} {'mean +/- std':<18} {'mean / max':<15} "
        f"{'mean / max':<15} {'(nodes/sec)':<18}"
    )
    print("-" * 90)

    total_time_mean = 0
    for r in results:
        summary = r.summary()
        if not summary:
            continue

        time_str = f"{summary['time_ms_mean']:.2f} +/- {summary['time_ms_std']:.2f}"
        cpu_mem_str = (
            f"{summary['memory_mb_mean']:.1f} / {summary['memory_mb_max']:.1f}"
        )
        gpu_mem_str = (
            f"{summary['gpu_memory_mb_mean']:.1f} / {summary['gpu_memory_mb_max']:.1f}"
        )
        throughput_str = (
            f"{summary['nodes_per_second']:,.0f}"
            if "nodes_per_second" in summary
            else "N/A"
        )

        print(
            f"{summary['name']:<22} {time_str:<18} {cpu_mem_str:<15} "
            f"{gpu_mem_str:<15} {throughput_str:<18}"
        )

        if summary["name"] != "end_to_end":
            total_time_mean += summary["time_ms_mean"]

    print("-" * 90)
    print(f"{'Sum (steps only)':<22} {total_time_mean:.2f} ms")
    print("=" * 90)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Benchmark Theseus PIDS pipeline with realistic dataset sizes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "dataset",
        type=str,
        choices=["CADETS_E3", "THEIA_E3", "FIVEDIRECTIONS_E3", "TRACE_E3"],
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument(
        "--size", type=str, choices=["avg", "median", "max"], default="avg"
    )
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--num_nodes", type=int, default=None)
    parser.add_argument("--num_edges", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    config_arg_list = [args.dataset]
    if args.config:
        config_arg_list.extend(["--config", args.config])
    if args.cpu:
        config_arg_list.append("--cpu")

    config_args = parse_args(config_arg_list)
    config = parse_config(config_args)
    set_seed(args.seed)

    dataset_stats: DatasetStatistics = DATASET_STATISTICS.get(
        args.dataset, DATASET_STATISTICS["CADETS_E3"]
    )

    if args.num_nodes is not None and args.num_edges is not None:
        num_nodes = args.num_nodes
        num_edges = args.num_edges
        size_desc = "custom"
    elif args.size == "median":
        num_nodes = dataset_stats.median_nodes
        num_edges = dataset_stats.median_edges
        size_desc = "median"
    elif args.size == "max":
        num_nodes = dataset_stats.max_nodes
        num_edges = dataset_stats.max_edges
        size_desc = "max"
    else:
        num_nodes = dataset_stats.avg_nodes
        num_edges = dataset_stats.avg_edges
        size_desc = "average"

    node_type_dist = dataset_stats.node_type_dist

    log(f"Benchmarking Theseus pipeline for {args.dataset}")
    log(f"Device: {config.device}")
    log(f"Graph size: {size_desc} ({num_nodes:,} nodes, {num_edges:,} edges)")
    log(
        f"Node type distribution: process={node_type_dist['process']:.0%}, "
        f"file={node_type_dist['file']:.0%}, netflow={node_type_dist['netflow']:.0%}"
    )

    edge_type_to_onehot = create_one_hot(DARPA_TC_EVENTS)
    ntype2onehot = create_one_hot(NODE_TYPES)

    node_ids = [str(i) for i in range(num_nodes)]
    base_timestamp = 1539100800000000000

    node_embeddings = {}
    process_thresh = node_type_dist["process"]
    file_thresh = process_thresh + node_type_dist["file"]

    for node_id in node_ids:
        rand_val = random.random()
        if rand_val < process_thresh:
            node_type = "process"
        elif rand_val < file_thresh:
            node_type = "file"
        else:
            node_type = "netflow"

        node_type_onehot = ntype2onehot[node_type]
        word2vec_embedding = torch.randn(config.embed_dim, dtype=torch.float32)
        node_embeddings[node_id] = torch.cat([node_type_onehot, word2vec_embedding])

    edge_arrays = generate_synthetic_edge_arrays(
        num_nodes, num_edges, DARPA_TC_EVENTS, base_timestamp
    )

    all_malicious_nodes = set()
    day = config.dataset_info.test_days[0] if config.dataset_info.test_days else 1

    sample_graph, _ = create_graph_window(
        edge_arrays,
        day,
        config,
        node_embeddings,
        edge_type_to_onehot,
        all_malicious_nodes,
    )

    graph_stats = {
        "num_nodes": sample_graph.x.shape[0],
        "num_edges": sample_graph.edge_index.shape[1],
        "num_process_nodes": int((sample_graph.x[:, 0] == 1).sum().item()),
    }

    log(
        f"Constructed graph: {graph_stats['num_nodes']:,} nodes, "
        f"{graph_stats['num_edges']:,} edges"
    )

    model = initialize_model(config, [sample_graph], checkpoint_path=None)
    model.eval()

    log(f"Warming up ({args.warmup} iterations)...")
    sample_graph_device = sample_graph.to(config.device)
    for _ in range(args.warmup):
        with torch.inference_mode():
            outputs, encoded_target = model(sample_graph_device)
            _ = model.loss(outputs, encoded_target, "none")
            if config.device == "cuda":
                torch.cuda.synchronize()

    log("Running benchmarks...")

    embedding_result = benchmark_embedding_generation(
        node_ids, config, args.num_samples
    )
    embedding_result.num_nodes = len(node_ids)

    construction_result = benchmark_graph_construction(
        edge_arrays,
        day,
        config,
        node_embeddings,
        edge_type_to_onehot,
        all_malicious_nodes,
        args.num_samples,
    )
    construction_result.num_edges = num_edges

    inference_result = benchmark_inference(
        model, sample_graph, config, args.num_samples
    )
    inference_result.num_nodes = graph_stats["num_nodes"]

    e2e_result = benchmark_end_to_end(
        num_nodes,
        num_edges,
        DARPA_TC_EVENTS,
        base_timestamp,
        day,
        config,
        edge_type_to_onehot,
        all_malicious_nodes,
        model,
        args.num_samples,
    )
    e2e_result.num_nodes = graph_stats["num_nodes"]
    e2e_result.num_edges = num_edges

    print_benchmark_results(
        [embedding_result, construction_result, inference_result, e2e_result],
        graph_stats,
        scenario=f"{args.dataset} ({size_desc}: {num_nodes:,} nodes, {num_edges:,} edges)",
    )

    emb_summary = embedding_result.summary()
    const_summary = construction_result.summary()
    inf_summary = inference_result.summary()
    e2e_summary = e2e_result.summary()

    window_ms = WINDOW_SIZE_MINUTES * 60 * 1000
    e2e_latency_ms = e2e_summary["time_ms_mean"]
    speedup_vs_realtime = window_ms / e2e_latency_ms
    utilization_pct = (e2e_latency_ms / window_ms) * 100

    total_graphs = dataset_stats.total_graphs
    avg_edges = dataset_stats.avg_edges
    total_edges = avg_edges * total_graphs
    total_seconds = total_graphs * WINDOW_SIZE_MINUTES * 60
    events_per_sec = total_edges / total_seconds

    print("\n" + "=" * 78)
    print("RUNTIME ANALYSIS SUMMARY")
    print("=" * 78)
    print(f"Dataset:              {args.dataset}")
    print(
        f"Graph Size:           {size_desc} ({num_nodes:,} nodes, {num_edges:,} edges)"
    )
    print(f"Device:               {config.device}")
    print(f"Samples:              {args.num_samples}")
    print("-" * 78)
    print("PIPELINE LATENCY")
    print("-" * 78)
    print(
        f"  Embedding Gen:      {emb_summary['time_ms_mean']:>8.2f} +/- "
        f"{emb_summary['time_ms_std']:<6.2f} ms"
    )
    print(
        f"  Graph Construction: {const_summary['time_ms_mean']:>8.2f} +/- "
        f"{const_summary['time_ms_std']:<6.2f} ms"
    )
    print(
        f"  Model Inference:    {inf_summary['time_ms_mean']:>8.2f} +/- "
        f"{inf_summary['time_ms_std']:<6.2f} ms"
    )
    print("  " + "-" * 39)
    print(
        f"  End-to-End Total:   {e2e_latency_ms:>8.2f} +/- "
        f"{e2e_summary['time_ms_std']:<6.2f} ms"
    )
    print("-" * 78)
    print("REAL-TIME VIABILITY")
    print("-" * 78)
    print(
        f"  Time Window:        {WINDOW_SIZE_MINUTES} minutes "
        f"({window_ms / 1000:.0f} seconds)"
    )
    print(f"  Processing Time:    {e2e_latency_ms / 1000:.2f} seconds")
    print(f"  Speedup vs Window:  {speedup_vs_realtime:,.0f}x faster than real-time")
    print(f"  CPU Utilization:    {utilization_pct:.2f}% of window budget")
    print("-" * 78)
    print("THROUGHPUT")
    print("-" * 78)
    print(f"  Nodes/sec:          {inf_summary.get('nodes_per_second', 0):>12,.0f}")
    print(f"  Edges/sec:          {const_summary.get('edges_per_second', 0):>12,.0f}")
    print(f"  Dataset event rate: {events_per_sec:>12,.0f} events/sec (estimated)")
    print("-" * 78)
    print("RESOURCE USAGE")
    print("-" * 78)
    print(f"  GPU Memory (peak):  {e2e_summary['gpu_memory_mb_max']:.1f} MB")
    print(f"  CPU Memory (peak):  {e2e_summary['memory_mb_max']:.1f} MB")
    print("=" * 78)

    print("\n" + "-" * 78)
    if speedup_vs_realtime >= 100:
        print("VERDICT: System is REAL-TIME VIABLE")
        print(
            f"  Processing {WINDOW_SIZE_MINUTES}-min window in "
            f"{e2e_latency_ms / 1000:.2f}s ({speedup_vs_realtime:.0f}x faster)"
        )
        print(
            f"  Can process up to {speedup_vs_realtime:.0f} parallel streams "
            "on single GPU"
        )
    elif speedup_vs_realtime >= 10:
        print("VERDICT: System is REAL-TIME VIABLE with margin")
        print(
            f"  Processing {WINDOW_SIZE_MINUTES}-min window in "
            f"{e2e_latency_ms / 1000:.2f}s ({speedup_vs_realtime:.0f}x faster)"
        )
    else:
        print("WARNING: System may struggle with real-time processing")
        print(f"  Processing takes {utilization_pct:.1f}% of available window time")
    print("-" * 78)

    log("Benchmark complete!")


if __name__ == "__main__":
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    main()
