import contextlib
import fcntl
import os
import time
from collections import namedtuple

import numpy as np
import polars as pl
import torch
from gensim.models import Word2Vec
from torch_geometric.data import Data

from utils.constants.graph_events import DARPA_TC_EVENTS, NODE_TYPES
from utils.constants.token_weighting import (
    TOKEN_WEIGHTING_MODE_CACHE_SUFFIX,
    TOKEN_WEIGHTING_MODE_TO_REVERSED_NODE_TYPES,
    get_token_weighting_mode,
)
from utils.ground_truth import get_ground_truth
from utils.utils import (
    create_one_hot,
    datetime_to_ns_time_us,
    log,
    log_tqdm,
    read_node_table,
    timed_execution,
    tokenize_node_description,
)

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

# Attack classes to exclude from graphs (artifacts that affect training/evaluation)
# These nodes will be completely removed from train/val/test sets
EXCLUDED_ATTACK_CLASSES = {
    "wwtawwtal_bad_neighborhood"  # CADETS artifact
}

EventRecord = namedtuple("EventRecord", EVENT_COLUMNS)


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


def _determine_split(day, config):
    if day in config.dataset_info.train_days:
        return "train"
    elif day in config.dataset_info.val_days:
        return "val"
    elif day in config.dataset_info.test_days:
        return "test"
    return "train"


def _format_event_day_label(config, day):
    return f"{config.dataset_info.year_month}-{int(day):02d}"


def _get_selected_days(config):
    days = set()
    days.update(config.dataset_info.train_days)
    days.update(config.dataset_info.val_days)
    days.update(config.dataset_info.test_days)
    return sorted(days)


def _is_valid_parquet(path):
    """Check if a parquet file is valid and readable."""
    if not os.path.exists(path):
        return False
    try:
        _ = pl.scan_parquet(path).collect_schema()
        return True
    except Exception:
        return False


def _ensure_partitioned_event_parquet(config, event_type_set):
    """Create Parquet dataset with only selected days from config (with file locking)."""

    data_dir = os.path.join(config.data_dir, config.dataset)
    csv_path = os.path.join(data_dir, "event_table.csv")
    raw_parquet_path = os.path.join(data_dir, "event_table.parquet")
    parquet_path = os.path.join(data_dir, "event_table_preprocessed.parquet")
    lock_path = parquet_path + ".lock"

    if os.path.exists(parquet_path) and _is_valid_parquet(parquet_path):
        return parquet_path

    if os.path.exists(raw_parquet_path):
        source_path = raw_parquet_path
        scan_fn = pl.scan_parquet
    elif os.path.exists(csv_path):
        source_path = csv_path
        scan_fn = pl.scan_csv
    else:
        raise FileNotFoundError(
            f"Event table not found at {raw_parquet_path} or {csv_path}."
        )

    selected_days = _get_selected_days(config)
    selected_day_labels = [
        _format_event_day_label(config, day) for day in selected_days
    ]

    log(
        f"Materializing preprocessed Parquet for {len(selected_days)} selected days: {selected_days}"
    )

    # Use file locking to prevent race conditions
    os.makedirs(data_dir, exist_ok=True)
    lock_fd = None
    try:
        lock_fd = open(lock_path, "w")  # noqa: SIM115
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        if os.path.exists(parquet_path) and _is_valid_parquet(parquet_path):
            return parquet_path

        lock_fd.write(f"Creating parquet at {time.time()}")
        lock_fd.flush()

        event_scan = (
            scan_fn(source_path)
            .select(EVENT_COLUMNS)
            .filter(pl.col("operation").is_in(event_type_set))
            .with_columns(
                pl.col("timestamp_rec")
                .cast(pl.Int64)
                .cast(pl.Datetime("ns"))
                .dt.strftime("%Y-%m-%d")
                .alias("event_day")
            )
            .filter(pl.col("event_day").is_in(selected_day_labels))
        )

        event_scan.sink_parquet(parquet_path, compression="zstd", mkdir=True)
        log(f"Created preprocessed Parquet at {parquet_path}")
    except Exception as e:
        log(f"Error creating preprocessed parquet: {e}")
        raise
    finally:
        if lock_fd is not None:
            with contextlib.suppress(Exception):
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
                lock_fd.close()
        with contextlib.suppress(Exception):
            os.remove(lock_path)

    return parquet_path


def _get_node_table_path(data_dir, table_name):
    """Get path to node table, preferring Parquet over CSV."""
    parquet_path = os.path.join(data_dir, f"{table_name}.parquet")
    csv_path = os.path.join(data_dir, f"{table_name}.csv")
    if os.path.exists(parquet_path):
        return parquet_path
    elif os.path.exists(csv_path):
        return csv_path
    return None


def _read_node_table_batched(data_dir, table_name, columns=None, batch_size=500_000):
    """Read node table in batches, trying Parquet first then CSV."""
    parquet_path = os.path.join(data_dir, f"{table_name}.parquet")
    csv_path = os.path.join(data_dir, f"{table_name}.csv")
    if os.path.exists(parquet_path):
        lf = pl.scan_parquet(parquet_path)
        if columns:
            lf = lf.select(columns)
        return [(pl.read_parquet(parquet_path, columns=columns), parquet_path)]
    elif os.path.exists(csv_path):
        batch_reader = pl.read_csv_batched(
            csv_path, columns=columns, batch_size=batch_size
        )
        return (batch_reader, csv_path)
    return None


def fetch_node_metadata(config):
    """Load node metadata from Parquet/CSV files: {node_id: [node_type, description]}."""
    node_metadata = {}
    data_dir = os.path.join(config.data_dir, config.dataset)

    # Netflow nodes
    netflow_path = _get_node_table_path(data_dir, "netflow_node_table")
    if netflow_path:
        log(f"Loading netflow nodes from {netflow_path}...")
        df = read_node_table(
            data_dir,
            "netflow_node_table",
            columns=["index_id", "src_addr", "src_port", "dst_addr", "dst_port"],
        )
        for row in df.iter_rows(named=True):
            index_id = str(row["index_id"])
            node_metadata[index_id] = [
                "netflow",
                f"{row['src_addr']} {row['src_port']} {row['dst_addr']} {row['dst_port']}",
            ]
        log(f"Number of netflow nodes: {len(df)}")
        del df

    process_path = _get_node_table_path(data_dir, "process_node_table")
    if process_path:
        log(f"Loading process nodes from {process_path}...")

        process_count = 0
        result = _read_node_table_batched(
            data_dir,
            "process_node_table",
            columns=["index_id", "path", "cmd"],
            batch_size=500_000,
        )

        if isinstance(result, list):
            # Parquet: single batch
            for df, _ in result:
                for row in df.iter_rows(named=True):
                    index_id = str(row["index_id"])
                    path = row["path"] if row["path"] else ""
                    cmd = row["cmd"] if row["cmd"] else ""
                    parts = [p for p in [path, cmd] if p]
                    description = " ".join(parts) if parts else ""
                    node_metadata[index_id] = ["process", description]
                process_count += len(df)
                del df
        else:
            # CSV: batched reader
            batch_reader, _ = result
            for batch in iter(lambda: batch_reader.next_batches(1), None):
                if len(batch) == 0:
                    break
                df = batch[0]
                for row in df.iter_rows(named=True):
                    index_id = str(row["index_id"])
                    path = row["path"] if row["path"] else ""
                    cmd = row["cmd"] if row["cmd"] else ""
                    parts = [p for p in [path, cmd] if p]
                    description = " ".join(parts) if parts else ""
                    node_metadata[index_id] = ["process", description]
                process_count += len(df)
                if process_count % 5_000_000 == 0:
                    log(f"  Processed {process_count:,} process nodes...")
                del df

        log(f"Number of process nodes: {process_count}")

    file_path = _get_node_table_path(data_dir, "file_node_table")
    if file_path:
        log(f"Loading file nodes from {file_path}...")
        df = read_node_table(data_dir, "file_node_table", columns=["index_id", "path"])
        for row in df.iter_rows(named=True):
            index_id = str(row["index_id"])
            path = row["path"] if row["path"] else ""
            node_metadata[index_id] = ["file", path]
        log(f"Number of file nodes: {len(df)}")
        del df

    log(f"Total nodes loaded: {len(node_metadata)}")
    return node_metadata


def train_word2vec(node_metadata, train_nodes, config):
    """Train or load cached Word2Vec models on training set nodes.

    Returns dict of models: {'text', 'netflow'} if grouped, else {'path', 'command', 'netflow'}.
    """
    word2vec_dir = os.path.join(config.checkpoint_dir, "word2vec")
    os.makedirs(word2vec_dir, exist_ok=True)

    cache_components = [
        f"word2vec_{config.dataset}",
        f"win{config.window_size_minutes}",
        f"seed{config.seed}",
    ]

    if config.fuse_edge:
        cache_components.append("fused")
    if config.bidirectional_edges:
        cache_components.append("bidir")

    base_filename = "_".join(cache_components)

    for filename in (base_filename, f"{base_filename}_contam"):
        path_model_path = os.path.join(word2vec_dir, f"{filename}_path.model")
        command_model_path = os.path.join(word2vec_dir, f"{filename}_command.model")
        netflow_model_path = os.path.join(word2vec_dir, f"{filename}_netflow.model")
        if (
            os.path.exists(path_model_path)
            and os.path.exists(command_model_path)
            and os.path.exists(netflow_model_path)
            and not config.force_restart
        ):
            log(f"Loading cached Word2Vec models from {word2vec_dir}")
            return {
                "path": Word2Vec.load(path_model_path),
                "command": Word2Vec.load(command_model_path),
                "netflow": Word2Vec.load(netflow_model_path),
            }

    path_model_path = os.path.join(word2vec_dir, f"{base_filename}_path.model")
    command_model_path = os.path.join(word2vec_dir, f"{base_filename}_command.model")
    netflow_model_path = os.path.join(word2vec_dir, f"{base_filename}_netflow.model")

    log("Training three specialized Word2Vec models on training set nodes...")

    path_words = []
    command_words = []
    netflow_words = []
    seen_descriptions = set()

    # Pre-filter to training nodes to avoid repeated set membership checks
    train_node_metadata = {
        nid: node_metadata[nid] for nid in train_nodes if nid in node_metadata
    }

    for _, (node_type, node_description) in train_node_metadata.items():
        if node_description not in seen_descriptions:
            seen_descriptions.add(node_description)
            # tokenize_node_description now returns [(token, token_type), ...]
            typed_tokens = tokenize_node_description(node_description, node_type)

            # Separate tokens by type in single pass instead of 3 list comprehensions
            path_tokens = []
            command_tokens = []
            netflow_tokens = []
            for token, ttype in typed_tokens:
                if ttype == "path":
                    path_tokens.append(token)
                elif ttype == "command":
                    command_tokens.append(token)
                elif ttype == "netflow":
                    netflow_tokens.append(token)

            if path_tokens:
                path_words.append(path_tokens)
            if command_tokens:
                command_words.append(command_tokens)
            if netflow_tokens:
                netflow_words.append(netflow_tokens)

    log("Training Word2Vec models:")
    log(
        f"  Path model: {len(path_words)} sequences from {len(train_nodes)} training nodes"
    )
    log(
        f"  Command model: {len(command_words)} sequences from {len(train_nodes)} training nodes"
    )
    log(
        f"  Netflow model: {len(netflow_words)} sequences from {len(train_nodes)} training nodes"
    )

    models = {}

    if path_words:
        path_model = Word2Vec(
            path_words,
            alpha=config.word2vec.alpha,
            vector_size=config.word2vec.emb_dim,
            window=max(5, config.word2vec.window_size),
            min_count=config.word2vec.min_count,
            sg=config.word2vec.use_skip_gram,
            epochs=config.word2vec.epochs,
            compute_loss=config.word2vec.compute_loss,
            negative=config.word2vec.negative,
            seed=config.seed,
            workers=1,
        )
        path_model.save(path_model_path)
        log(f"Saved path Word2Vec model to {path_model_path}")
        models["path"] = path_model
    else:
        log("Warning: No path tokens found, creating empty model")
        models["path"] = None

    if command_words:
        command_model = Word2Vec(
            command_words,
            alpha=config.word2vec.alpha,
            vector_size=config.word2vec.emb_dim,
            window=config.word2vec.window_size,
            min_count=config.word2vec.min_count,
            sg=config.word2vec.use_skip_gram,
            epochs=config.word2vec.epochs,
            compute_loss=config.word2vec.compute_loss,
            negative=config.word2vec.negative,
            seed=config.seed,
            workers=1,
        )
        command_model.save(command_model_path)
        log(f"Saved command Word2Vec model to {command_model_path}")
        models["command"] = command_model
    else:
        log("Warning: No command tokens found, creating empty model")
        models["command"] = None

    if netflow_words:
        netflow_model = Word2Vec(
            netflow_words,
            alpha=config.word2vec.alpha,
            vector_size=config.word2vec.emb_dim,
            window=max(8, config.word2vec.window_size),
            min_count=config.word2vec.min_count,
            sg=config.word2vec.use_skip_gram,
            epochs=config.word2vec.epochs,
            compute_loss=config.word2vec.compute_loss,
            negative=config.word2vec.negative,
            seed=config.seed,
            workers=1,
        )
        netflow_model.save(netflow_model_path)
        log(f"Saved netflow Word2Vec model to {netflow_model_path}")
        models["netflow"] = netflow_model
    else:
        log("Warning: No netflow tokens found, creating empty model")
        models["netflow"] = None

    return models


def embed_all_nodes(node_metadata, word2vec_models, config, selected_nodes_only=None):
    """Create node feature vectors (type one-hot + Word2Vec embedding)."""
    if selected_nodes_only:
        log(f"Computing node features for {len(selected_nodes_only)} selected nodes...")
        nodes_to_embed = {
            nid: node_metadata[nid]
            for nid in selected_nodes_only
            if nid in node_metadata
        }
    else:
        log("Computing node features (type + Word2Vec embeddings) for all nodes...")
        nodes_to_embed = node_metadata

    ntype2onehot = create_one_hot(NODE_TYPES)

    decline_percentage = config.word2vec.decline_rate
    embedding_dim = config.word2vec.emb_dim
    zeros_np = np.zeros((embedding_dim,))
    zeros_torch = torch.zeros(embedding_dim, dtype=torch.float32)
    node_embeddings = {}
    token_weighting_mode = get_token_weighting_mode(config)
    reverse_weight_node_types = TOKEN_WEIGHTING_MODE_TO_REVERSED_NODE_TYPES[
        token_weighting_mode
    ]

    log(
        f"Token weighting mode: {token_weighting_mode} (reversed types: {sorted(reverse_weight_node_types)})"
    )

    for node_id, metadata in log_tqdm(
        nodes_to_embed.items(), desc="Embedding nodes", miniters=1000
    ):
        node_type, node_description = metadata[0], metadata[1]
        node_type_onehot = ntype2onehot[node_type]
        typed_tokens = tokenize_node_description(node_description, node_type)

        n = len(typed_tokens)
        if n == 0:
            node_embedding = zeros_torch
        else:
            d = -1 / n * decline_percentage / 100
            a_1 = 1 / n - 0.5 * (n - 1) * d
            weight_list = [a_1 + i * d for i in range(n)]
            if node_type in reverse_weight_node_types:
                # Reverse selected node types to emphasize trailing tokens
                weight_list.reverse()

            vectors_list = []
            for token, token_type in typed_tokens:
                model = word2vec_models.get(token_type)
                if model is not None and token in model.wv:
                    vectors_list.append(model.wv[token])
                else:
                    vectors_list.append(zeros_np)

            word_vectors = np.array(vectors_list)
            weights = np.array(weight_list)[:, np.newaxis]

            sentence_vector = np.mean(word_vectors * weights, axis=0)
            norm = np.linalg.norm(sentence_vector) + 1e-12
            node_embedding = torch.from_numpy(sentence_vector / norm).float()

        full_feature = torch.cat([node_type_onehot, node_embedding])
        node_embeddings[node_id] = full_feature

    log(f"Created in-memory features for {len(node_embeddings)} nodes")
    return node_embeddings


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
    edge_arrays, day, config, node_embeddings, edge_type_to_onehot, all_malicious_nodes
):
    """Build a PyG Data object for a single time window with node/edge features and labels."""
    src_ids, dst_ids, edge_times, edge_ops, fused_counts = edge_arrays
    split = _determine_split(day, config)
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
        node_labels[local_id] = 1 if orig_id in all_malicious_nodes else 0

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
    day_label = _format_event_day_label(config, day)
    start_ns = datetime_to_ns_time_us(f"{day_label} 00:00:00")
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
    event_type_set = set(DARPA_TC_EVENTS)

    parquet_path = _ensure_partitioned_event_parquet(config, event_type_set)
    events_scan = pl.scan_parquet(parquet_path).filter(
        pl.col("operation").is_in(event_type_set)
    )

    edge_type_to_onehot = create_one_hot(DARPA_TC_EVENTS)
    selected_nodes = get_nodes_from_selected_days(config, parquet_path)
    node_embeddings = embed_all_nodes(
        node_metadata, word2vec_models, config, selected_nodes_only=selected_nodes
    )

    graph_windows = {"train": [], "val": [], "test": []}

    days = _get_selected_days(config)
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


def get_nodes_from_selected_days(config, parquet_path):
    """Collect unique node IDs appearing in any of the train/val/test days."""
    log("Identifying nodes from selected days...")

    events_scan = pl.scan_parquet(parquet_path).select(
        ["event_day", "src_index_id", "dst_index_id"]
    )

    selected_days = _get_selected_days(config)
    all_nodes = set()

    for day in log_tqdm(
        selected_days, desc="Collecting nodes from selected days", logging=False
    ):
        day_label = _format_event_day_label(config, day)

        day_events = (
            events_scan.filter(pl.col("event_day") == day_label)
            .select(["src_index_id", "dst_index_id"])
            .collect()
        )

        assert isinstance(day_events, pl.DataFrame)

        if day_events.is_empty():
            continue

        all_nodes.update(str(x) for x in day_events["src_index_id"].to_list())
        all_nodes.update(str(x) for x in day_events["dst_index_id"].to_list())

    log(f"Found {len(all_nodes)} unique nodes in selected days")
    return all_nodes


def get_training_nodes_from_csv(config):
    """Collect unique node IDs appearing in training days only."""
    log("Identifying training set nodes from partitioned event dataset...")

    event_types = DARPA_TC_EVENTS
    event_type_set = set(event_types)
    parquet_path = _ensure_partitioned_event_parquet(config, event_type_set)

    events_scan = pl.scan_parquet(parquet_path).select(
        ["event_day", "src_index_id", "dst_index_id"]
    )

    train_nodes = set()

    for day in log_tqdm(
        config.dataset_info.train_days, desc="Collecting training nodes", logging=False
    ):
        day_label = _format_event_day_label(config, day)

        day_events = (
            events_scan.filter(pl.col("event_day") == day_label)
            .select(["src_index_id", "dst_index_id"])
            .collect()
        )

        assert isinstance(day_events, pl.DataFrame)

        if day_events.is_empty():
            continue

        train_nodes.update(str(x) for x in day_events["src_index_id"].to_list())
        train_nodes.update(str(x) for x in day_events["dst_index_id"].to_list())

    log(f"Found {len(train_nodes)} unique nodes in training set")
    return train_nodes


def get_cache_filename(dataset_name, config):
    """Generate cache filename reflecting configuration options."""
    components = [f"graph_{dataset_name.lower()}"]

    if config.use_fused_edge_count:
        components.append("fused_edge")
    if config.use_node_degrees:
        components.append("node_deg")

    token_weighting_mode = get_token_weighting_mode(config)
    components.append(TOKEN_WEIGHTING_MODE_CACHE_SUFFIX[token_weighting_mode])
    components.append(str(config.seed))

    return "_".join(components) + "_cache.pt"


def _save_graph_cache(graphs, ground_truth, cache_dir, dataset_name, config):
    os.makedirs(cache_dir, exist_ok=True)
    cache_filename = get_cache_filename(dataset_name, config)
    cache_path = os.path.join(cache_dir, cache_filename)
    torch.save({"graphs": graphs, "ground_truth": ground_truth}, cache_path)
    log(f"Saved graph cache with ground truth to {cache_path}")


def _filter_ground_truth(ground_truth):
    """Remove excluded attack classes from ground truth."""
    filtered = {}
    for attack_id, attack_metadata in ground_truth.items():
        if attack_id not in EXCLUDED_ATTACK_CLASSES:
            filtered[attack_id] = attack_metadata
        else:
            log(f"Filtering out attack class '{attack_id}' from ground truth")
    return filtered


def _get_malicious_nodes(ground_truth):
    """Extract malicious node IDs from ground truth (attack + contaminated)."""
    all_malicious_nodes = set()
    for attack_id, attack_metadata in ground_truth.items():
        if attack_id in EXCLUDED_ATTACK_CLASSES:
            continue
        all_malicious_nodes.update(attack_metadata.get("nids", []) or [])
        all_malicious_nodes.update(attack_metadata.get("contaminated_nids", []) or [])
    return all_malicious_nodes


def _relabel_graphs(graphs, all_malicious_nodes):
    """Relabel graph nodes based on the set of malicious node IDs (in-place)."""
    total_relabeled = 0
    for graph_list in graphs.values():
        for graph in graph_list:
            if hasattr(graph, "original_n_id"):
                original_ids = graph.original_n_id.numpy()
                new_labels = torch.tensor(
                    [1 if nid in all_malicious_nodes else 0 for nid in original_ids],
                    dtype=torch.long,
                )
                graph.y = new_labels
                total_relabeled += len(new_labels)
    log(f"Relabeled {total_relabeled} nodes across all graphs")


def load_graph_cache(cache_dir, dataset_name, config):
    """Load graphs and ground truth from cache (returns None, None if invalid)."""
    cache_filename = get_cache_filename(dataset_name, config)
    cache_path = os.path.join(cache_dir, cache_filename)

    if not os.path.exists(cache_path):
        return None, None

    log(f"Loading graph cache from {cache_path}")
    cache_data = torch.load(cache_path, map_location="cpu", weights_only=False)

    graphs = cache_data.get("graphs")
    ground_truth = cache_data.get("ground_truth")

    if not graphs or not ground_truth:
        raise ValueError("Cache file is missing required data.")

    # Validate ground truth is not stale (should have malicious nodes)
    try:
        total_malicious = 0
        for meta in ground_truth.values():
            if isinstance(meta, dict):
                total_malicious += len(meta.get("nids", set()))
                total_malicious += len(meta.get("contaminated_nids", set()))
    except Exception:
        total_malicious = 0

    if total_malicious == 0:
        log(
            f"Cached ground_truth appears empty (0 malicious nodes). Ignoring cache {cache_path} and rebuilding."
        )
        return None, None

    log(
        f"Loaded {sum(len(g) for g in graphs.values())} graphs with ground truth metadata"
    )
    return graphs, ground_truth


@timed_execution
def build_graphs(config):
    """Build or load cached graphs with features and ground truth labels."""
    cache_dir = config.cache_dir
    dataset_name = config.dataset

    cached_graphs, cached_ground_truth = load_graph_cache(
        cache_dir, dataset_name, config
    )
    if cached_graphs and cached_ground_truth:
        filtered_ground_truth = _filter_ground_truth(cached_ground_truth)
        all_malicious_nodes = _get_malicious_nodes(filtered_ground_truth)

        log(
            f"Label scope: causal (attack + contaminated): {len(all_malicious_nodes)} malicious nodes"
        )
        _relabel_graphs(cached_graphs, all_malicious_nodes)
        return cached_graphs, filtered_ground_truth

    log("Building graphs with embedded features and ground truth labels...")

    try:
        ground_truth = get_ground_truth(config)
    except Exception as exc:
        raise ValueError(f"Failed to load ground truth: {exc}") from exc

    filtered_ground_truth = _filter_ground_truth(ground_truth)
    all_malicious_nodes = _get_malicious_nodes(filtered_ground_truth)
    log(
        f"Label scope: causal (attack + contaminated): {len(all_malicious_nodes)} malicious nodes"
    )

    node_metadata = fetch_node_metadata(config)
    train_nodes = get_training_nodes_from_csv(config)
    word2vec_models = train_word2vec(node_metadata, train_nodes, config)

    graphs = gen_edge_fused_tw(
        node_metadata, config, word2vec_models, all_malicious_nodes
    )

    _save_graph_cache(graphs, ground_truth, cache_dir, dataset_name, config)
    log("Graph building complete!")
    return graphs, filtered_ground_truth
