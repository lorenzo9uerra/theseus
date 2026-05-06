from collections import namedtuple
from datetime import UTC, datetime

import numpy as np
import polars as pl
import torch
from torch_geometric.data import Data

from tasks.build_graphs_support import (
    determine_split,
    embed_all_nodes,
    ensure_partitioned_event_parquet,
    fetch_node_metadata,
    filter_ground_truth,
    format_event_day_label,
    get_malicious_nodes,
    get_nodes_from_selected_days,
    get_selected_days,
    get_training_nodes_from_csv,
    load_graph_cache,
    relabel_graphs,
    save_graph_cache,
    train_word2vec,
)
from utils.constants.graph_events import get_dataset_event_types
from utils.ground_truth import get_ground_truth
from utils.utils import create_one_hot, datetime_to_ns_time_us, log, timed_execution

NANOSECONDS_PER_MINUTE = 60_000_000_000
NANOSECONDS_PER_DAY = 24 * 60 * NANOSECONDS_PER_MINUTE  # 24 hours * 60 minutes
EVENT_COLUMNS = [
    "src_node",
    "src_index_id",
    "operation",
    "dst_node",
    "dst_index_id",
    "event_uuid",
    "timestamp_rec",
]

EventRecord = namedtuple("EventRecord", EVENT_COLUMNS)


def _datetime_to_ns_time(date_str: str, tz_name: str) -> int:
    """Convert 'YYYY-MM-DD HH:MM:SS' to epoch ns using the configured timezone."""
    if tz_name.strip().upper() in {"UTC", "GMT"}:
        dt = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC)
        return int(dt.timestamp() * 1_000_000_000)
    return datetime_to_ns_time_us(date_str)


def _group_consecutive_operations(operation_list):
    if not operation_list:
        return []

    groups = []
    current_operation = operation_list[0]
    start_idx = 0

    for idx in range(1, len(operation_list)):
        if operation_list[idx] != current_operation:
            groups.append((start_idx, idx - 1, current_operation))
            current_operation = operation_list[idx]
            start_idx = idx

    groups.append((start_idx, len(operation_list) - 1, current_operation))
    return groups


def _split_graph_by_node_limit(graph, max_nodes):
    """Split graph into subgraphs with at most max_nodes nodes each (evenly distributed)."""
    num_nodes = graph.x.shape[0]

    if num_nodes <= max_nodes:
        return [graph]

    num_subgraphs = (num_nodes + max_nodes - 1) // max_nodes
    nodes_per_subgraph = num_nodes // num_subgraphs

    subgraphs = []
    for i in range(num_subgraphs):
        start_idx = i * nodes_per_subgraph
        end_idx = num_nodes if i == num_subgraphs - 1 else (i + 1) * nodes_per_subgraph
        node_indices = torch.arange(start_idx, end_idx)

        subgraph = graph.subgraph(node_indices)
        subgraphs.append(subgraph)

    return subgraphs


def create_graph_window(
    edge_arrays, day, config, node_embeddings, edge_type_to_onehot, malicious_nodes
):
    """Build a PyG Data object for a single time window with node/edge features and labels."""
    src_ids, dst_ids, edge_times, edge_ops, fused_counts = edge_arrays
    split = determine_split(day, config)
    num_raw_edges = len(src_ids)

    # Build node ID mappings
    unique_node_ids = np.unique(np.concatenate([src_ids, dst_ids]))
    num_nodes = len(unique_node_ids)
    max_node_id = unique_node_ids.max() + 1

    id_to_local = np.zeros(max_node_id, dtype=np.int64)
    id_to_local[unique_node_ids] = np.arange(num_nodes)

    # Node features
    first_embedding = next(iter(node_embeddings.values()))
    base_feature_dim = first_embedding.shape[0]
    feature_dim = base_feature_dim + 1

    node_features = torch.zeros((num_nodes, feature_dim), dtype=torch.float32)
    node_labels = torch.zeros(num_nodes, dtype=torch.long)

    # Compute per-window node degrees if enabled
    if config.use_node_degrees:
        degree_counts = np.zeros(num_nodes, dtype=np.int32)
        np.add.at(degree_counts, id_to_local[src_ids], 1)
        np.add.at(degree_counts, id_to_local[dst_ids], 1)

    for local_id, orig_id in enumerate(unique_node_ids):
        node_id_str = str(orig_id)
        if node_id_str in node_embeddings:
            node_features[local_id, :base_feature_dim] = node_embeddings[node_id_str]
            if config.use_node_degrees:
                node_features[local_id, -1] = np.log1p(degree_counts[local_id])
        node_labels[local_id] = 1 if orig_id in malicious_nodes else 0

    # Map edges to local node indices
    src_local = id_to_local[src_ids]
    dst_local = id_to_local[dst_ids]

    # Build edge tensors
    bidirectional = config.bidirectional_edges
    if bidirectional:
        num_edges = num_raw_edges * 2
        edge_index = torch.from_numpy(
            np.stack(
                [
                    np.concatenate([src_local, dst_local]),
                    np.concatenate([dst_local, src_local]),
                ],
                axis=0,
            )
        )
        timestamps = torch.from_numpy(np.tile(edge_times, 2))
    else:
        num_edges = num_raw_edges
        edge_index = torch.from_numpy(np.stack([src_local, dst_local], axis=0))
        timestamps = torch.from_numpy(edge_times)

    # Edge features with vectorized operation assignment
    edge_feature_dim = len(edge_type_to_onehot) + 1
    edge_features = torch.zeros((num_edges, edge_feature_dim), dtype=torch.float32)

    op_to_idx = {op: idx for idx, op in enumerate(edge_type_to_onehot.keys())}
    op_indices = np.array([op_to_idx.get(op, -1) for op in edge_ops], dtype=np.int32)
    valid_ops = op_indices >= 0

    if valid_ops.any():
        valid_idx = np.where(valid_ops)[0]
        valid_op_idx = op_indices[valid_ops]
        edge_features[valid_idx, valid_op_idx] = 1.0
        if bidirectional:
            edge_features[valid_idx + num_raw_edges, valid_op_idx] = 1.0

    # Fused edge count feature
    if config.use_fused_edge_count:
        fused_log = np.log1p(fused_counts.astype(np.float32))
        if bidirectional:
            edge_features[:num_raw_edges, -1] = torch.from_numpy(fused_log)
            edge_features[num_raw_edges:, -1] = torch.from_numpy(fused_log)
        else:
            edge_features[:, -1] = torch.from_numpy(fused_log)

    data = Data(
        x=node_features,
        edge_index=edge_index,
        t=timestamps,
        edge_attr=edge_features,
        y=node_labels,
    )
    data.original_n_id = torch.from_numpy(unique_node_ids.astype(np.int64))

    return data, split


def _process_events_to_graphs(
    day_events, day, config, node_embeddings, edge_type_to_onehot, all_malicious_nodes
):
    """Convert a day's events into temporal graph windows.

    Returns dict of {split: [graphs]} for the processed events.
    """
    result = {"train": [], "val": [], "test": []}

    if day_events.is_empty():
        return result

    if "event_day" in day_events.columns:
        day_events = day_events.drop("event_day")

    events_list = [
        EventRecord(
            src_node=row["src_node"],
            src_index_id=row["src_index_id"],
            operation=row["operation"],
            dst_node=row["dst_node"],
            dst_index_id=row["dst_index_id"],
            event_uuid=row["event_uuid"],
            timestamp_rec=row["timestamp_rec"],
        )
        for row in day_events.iter_rows(named=True)
    ]

    if not events_list:
        return result

    window_size_in_ns = config.window_size_minutes * NANOSECONDS_PER_MINUTE
    window_start_time = events_list[0].timestamp_rec
    window_events = []
    last_event = events_list[-1]

    for event in events_list:
        window_events.append(event)

        is_last_event = event is last_event
        time_exceeded = event.timestamp_rec >= window_start_time + window_size_in_ns

        if time_exceeded or is_last_event:
            if config.fuse_edge:
                edge_arrays = _build_fused_edge_arrays(window_events)
            else:
                edge_arrays = _build_edge_arrays(window_events)

            graph_window, split = create_graph_window(
                edge_arrays,
                day,
                config,
                node_embeddings,
                edge_type_to_onehot,
                all_malicious_nodes,
            )

            split_graphs = _split_graph_by_node_limit(graph_window, config.max_nodes)
            result[split].extend(split_graphs)

            window_start_time = event.timestamp_rec
            window_events.clear()

    return result


def _build_edge_arrays(window_events):
    """Build edge arrays directly without intermediate dicts."""
    n = len(window_events)
    src_ids = np.empty(n, dtype=np.int64)
    dst_ids = np.empty(n, dtype=np.int64)
    edge_times = np.empty(n, dtype=np.int64)
    edge_ops = []
    fused_counts = np.ones(n, dtype=np.int32)

    for i, ev in enumerate(window_events):
        src_ids[i] = ev.src_index_id
        dst_ids[i] = ev.dst_index_id
        edge_times[i] = ev.timestamp_rec
        edge_ops.append(ev.operation)

    return src_ids, dst_ids, edge_times, edge_ops, fused_counts


def _build_fused_edge_arrays(window_events):
    """Build fused edge arrays, grouping consecutive identical operations."""
    edge_info = {}
    for ev in window_events:
        edge_key = (ev.src_index_id, ev.dst_index_id)
        if edge_key not in edge_info:
            edge_info[edge_key] = []
        edge_info[edge_key].append((ev.timestamp_rec, ev.operation))

    src_list = []
    dst_list = []
    time_list = []
    op_list = []
    count_list = []

    for (src, dst), event_data in edge_info.items():
        sorted_events = sorted(event_data)
        operation_list = [evt[1] for evt in sorted_events]

        for start_idx, end_idx, op_type in _group_consecutive_operations(
            operation_list
        ):
            src_list.append(src)
            dst_list.append(dst)
            time_list.append(sorted_events[start_idx][0])
            op_list.append(op_type)
            count_list.append(end_idx - start_idx + 1)

    src_ids = np.array(src_list, dtype=np.int64)
    dst_ids = np.array(dst_list, dtype=np.int64)
    edge_times = np.array(time_list, dtype=np.int64)
    fused_counts = np.array(count_list, dtype=np.int32)

    sort_order = np.argsort(edge_times)
    return (
        src_ids[sort_order],
        dst_ids[sort_order],
        edge_times[sort_order],
        [op_list[i] for i in sort_order],
        fused_counts[sort_order],
    )


def _load_day_events(events_scan, day_label, start_ns, end_ns):
    return (
        events_scan.filter(pl.col("event_day") == day_label)
        .filter(
            (pl.col("timestamp_rec") >= start_ns) & (pl.col("timestamp_rec") < end_ns)
        )
        .sort(["timestamp_rec", "event_uuid"])
        .collect()
    )


def _get_event_count_estimate(events_scan, day_label):
    try:
        count = (
            events_scan.filter(pl.col("event_day") == day_label)
            .select(pl.count())
            .collect()
            .item()
        )
        return count
    except Exception:
        return 0


def _process_day_in_chunks(
    day,
    day_idx,
    total_days,
    day_label,
    start_ns,
    end_ns,
    node_embeddings,
    edge_type_to_onehot,
    all_malicious_nodes,
    config,
    events_scan,
):
    """Process high-volume day in hourly chunks to limit memory usage.

    Returns dict of {split: [graphs]} for the day.
    """
    log(f"High event volume detected - processing day {day} in 1-hour chunks")
    result = {"train": [], "val": [], "test": []}
    chunk_duration = 1 * 60 * NANOSECONDS_PER_MINUTE  # 1 hour

    for chunk_idx in range(24):  # 24 chunks x 1 hour = 24 hours
        chunk_start = start_ns + (chunk_idx * chunk_duration)
        chunk_end = min(chunk_start + chunk_duration, end_ns)

        log(f"  Chunk {chunk_idx + 1}/24 (hour {chunk_idx})")

        chunk_events = _load_day_events(events_scan, day_label, chunk_start, chunk_end)

        if chunk_events.is_empty():
            continue

        if "event_day" in chunk_events.columns:
            chunk_events = chunk_events.drop("event_day")

        chunk_graphs = _process_events_to_graphs(
            chunk_events,
            day,
            config,
            node_embeddings,
            edge_type_to_onehot,
            all_malicious_nodes,
        )
        for split in result:
            result[split].extend(chunk_graphs[split])

    total_windows = sum(len(g) for g in result.values())
    log(f"Completed day {day_idx + 1}/{total_days}: Created {total_windows} windows")
    return result


def gen_edge_fused_tw_single_day(
    day,
    day_idx,
    total_days,
    node_embeddings,
    edge_type_to_onehot,
    all_malicious_nodes,
    config,
    events_scan,
):
    """Process a single day into graph windows (auto-chunks high-volume days).

    Returns dict of {split: [graphs]} for the day.
    """
    log(f"Processing day {day_idx + 1}/{total_days}: {day}")
    day_label = format_event_day_label(config, day)
    tz_name = getattr(config.dataset_info, "timezone", "US/Eastern")
    start_ns = _datetime_to_ns_time(f"{day_label} 00:00:00", tz_name)
    end_ns = start_ns + NANOSECONDS_PER_DAY

    event_count = _get_event_count_estimate(events_scan, day_label)
    needs_chunking = event_count > 5_000_000

    if needs_chunking:
        return _process_day_in_chunks(
            day,
            day_idx,
            total_days,
            day_label,
            start_ns,
            end_ns,
            node_embeddings,
            edge_type_to_onehot,
            all_malicious_nodes,
            config,
            events_scan,
        )

    day_events = _load_day_events(events_scan, day_label, start_ns, end_ns)

    if day_events.is_empty():
        log(f"No events found for day {day}, skipping")
        return {"train": [], "val": [], "test": []}

    day_graphs = _process_events_to_graphs(
        day_events,
        day,
        config,
        node_embeddings,
        edge_type_to_onehot,
        all_malicious_nodes,
    )

    total_windows = sum(len(g) for g in day_graphs.values())
    log(f"Completed day {day_idx + 1}/{total_days}: Created {total_windows} windows")
    return day_graphs


def gen_edge_fused_tw(node_metadata, config, word2vec_models, all_malicious_nodes):
    """Build temporal graph windows with node/edge features for all selected days."""
    event_types = get_dataset_event_types(config.dataset)
    event_type_set = set(event_types)

    parquet_path = ensure_partitioned_event_parquet(config, event_type_set)
    events_scan = pl.scan_parquet(parquet_path).filter(
        pl.col("operation").is_in(event_type_set)
    )

    edge_type_to_onehot = create_one_hot(event_types)
    selected_nodes = get_nodes_from_selected_days(config, parquet_path)
    node_embeddings = embed_all_nodes(
        node_metadata, word2vec_models, config, selected_nodes_only=selected_nodes
    )

    graph_windows = {"train": [], "val": [], "test": []}

    days = get_selected_days(config)
    total_days = len(days)
    log(f"Processing {total_days} days...")

    for day_idx, day in enumerate(days):
        day_graphs = gen_edge_fused_tw_single_day(
            day=day,
            day_idx=day_idx,
            total_days=total_days,
            node_embeddings=node_embeddings,
            edge_type_to_onehot=edge_type_to_onehot,
            all_malicious_nodes=all_malicious_nodes,
            config=config,
            events_scan=events_scan,
        )

        for split in graph_windows:
            graph_windows[split].extend(day_graphs[split])

        total_so_far = sum(len(g) for g in graph_windows.values())
        log(f"Total windows created so far: {total_so_far}")

    log(
        f"Graph counts by split: train={len(graph_windows['train'])}, "
        f"val={len(graph_windows['val'])}, test={len(graph_windows['test'])}"
    )

    return graph_windows


@timed_execution
def build_graphs(config):
    """Build or load cached graphs with features and ground truth labels."""
    cache_dir = config.cache_dir
    dataset_name = config.dataset

    cached_graphs, cached_ground_truth = load_graph_cache(
        cache_dir, dataset_name, config
    )
    if cached_graphs and cached_ground_truth:
        filtered_ground_truth = filter_ground_truth(cached_ground_truth)
        all_malicious_nodes = get_malicious_nodes(filtered_ground_truth)
        log(
            f"Label scope: causal (attack + contaminated): {len(all_malicious_nodes)} malicious nodes"
        )
        relabel_graphs(cached_graphs, all_malicious_nodes)
        return cached_graphs, filtered_ground_truth

    log("Building graphs with embedded features and ground truth labels...")

    try:
        ground_truth = get_ground_truth(config)
    except Exception as exc:
        raise ValueError(f"Failed to load ground truth: {exc}") from exc

    filtered_ground_truth = filter_ground_truth(ground_truth)
    all_malicious_nodes = get_malicious_nodes(filtered_ground_truth)
    log(
        f"Label scope: causal (attack + contaminated): {len(all_malicious_nodes)} malicious nodes"
    )

    node_metadata = fetch_node_metadata(config)
    train_nodes = get_training_nodes_from_csv(config)
    word2vec_models = train_word2vec(node_metadata, train_nodes, config)

    graphs = gen_edge_fused_tw(
        node_metadata, config, word2vec_models, all_malicious_nodes
    )

    save_graph_cache(graphs, ground_truth, cache_dir, dataset_name, config)
    log("Graph building complete!")
    return graphs, filtered_ground_truth
