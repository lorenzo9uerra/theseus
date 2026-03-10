import contextlib
import fcntl
import os
import time

import numpy as np
import polars as pl
import torch
from gensim.models import Word2Vec

from utils.constants.graph_events import DARPA_TC_EVENTS, NODE_TYPES
from utils.constants.token_weighting import (
    TOKEN_WEIGHTING_MODE_CACHE_SUFFIX,
    TOKEN_WEIGHTING_MODE_TO_REVERSED_NODE_TYPES,
    get_token_weighting_mode,
)
from utils.utils import (
    create_one_hot,
    log,
    log_tqdm,
    read_node_table,
    tokenize_node_description,
)

# Attack classes to exclude from graphs (artifacts that affect training/evaluation)
# These nodes will be completely removed from train/val/test sets
EXCLUDED_ATTACK_CLASSES = {
    "wwtawwtal_bad_neighborhood"  # CADETS artifact
}


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
            .select(
                [
                    "src_node",
                    "src_index_id",
                    "operation",
                    "dst_node",
                    "dst_index_id",
                    "event_uuid",
                    "timestamp_rec",
                ]
            )
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
        if columns:
            _ = pl.scan_parquet(parquet_path).select(columns)
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
    """Train or load cached Word2Vec models on training set nodes."""
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

    train_node_metadata = {
        nid: node_metadata[nid] for nid in train_nodes if nid in node_metadata
    }

    for _, (node_type, node_description) in train_node_metadata.items():
        if node_description in seen_descriptions:
            continue
        seen_descriptions.add(node_description)
        typed_tokens = tokenize_node_description(node_description, node_type)

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
    parquet_path = _ensure_partitioned_event_parquet(config, set(DARPA_TC_EVENTS))
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
