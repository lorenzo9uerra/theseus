"""
Parquet/CSV parser with daily splits.
Parses DARPA TC datasets into daily DGL graphs for MAGIC training/evaluation.
Uses Polars for fast parsing with lazy evaluation and filter pushdown.
Supports both Parquet (preferred) and CSV formats.
"""

import argparse
import gc
import json
import os
import pickle as pkl
from datetime import datetime

import dgl
import networkx as nx
import polars as pl
import torch.nn.functional as F

# Dataset split configurations (train/val/test days)
DATASET_SPLITS = {
    "cadets": {
        "train_days": [2, 3, 4, 5, 7, 8, 9],
        "val_days": [6, 10],
        "test_days": [11, 12, 13],
    },
    "fivedirections": {
        "train_days": [2, 3, 4, 5, 6, 7],
        "val_days": [8, 9],
        "test_days": [10, 11, 12, 13],
    },
    "theia": {"train_days": [11], "val_days": [10], "test_days": [12, 13]},
    "trace": {"train_days": [9, 11], "val_days": [10], "test_days": [12, 13]},
}

NANOSECONDS_PER_DAY = 24 * 60 * 60 * 1_000_000_000


def _read_table(dir_path, table_name, columns=None):
    """Read a table from Parquet (preferred) or CSV fallback."""
    parquet_path = os.path.join(dir_path, f"{table_name}.parquet")
    csv_path = os.path.join(dir_path, f"{table_name}.csv")

    if os.path.exists(parquet_path):
        return pl.read_parquet(parquet_path, columns=columns)
    elif os.path.exists(csv_path):
        if columns:
            return pl.read_csv(csv_path, columns=columns)
        return pl.read_csv(csv_path)
    return None


def _scan_table(dir_path, table_name):
    """Lazy scan a table from Parquet (preferred) or CSV fallback."""
    parquet_path = os.path.join(dir_path, f"{table_name}.parquet")
    csv_path = os.path.join(dir_path, f"{table_name}.csv")

    if os.path.exists(parquet_path):
        return pl.scan_parquet(parquet_path)
    elif os.path.exists(csv_path):
        return pl.scan_csv(csv_path)
    return None


def load_all_node_tables(data_dir):
    """
    Load all node tables and cache in memory.
    Returns hash_id -> (uuid, type, name).
    Uses Parquet if available, falls back to CSV.
    """
    hash_to_node = {}

    # Process nodes - only load needed columns
    print("Loading process nodes...")
    df = _read_table(
        data_dir, "process_node_table", columns=["hash_id", "node_uuid", "cmd", "path"]
    )
    if df is not None:
        # Prefer cmd, fallback to path
        if "cmd" in df.columns:
            names = df["cmd"].fill_null("").to_list()
        elif "path" in df.columns:
            names = df["path"].fill_null("").to_list()
        else:
            names = [""] * len(df)

        for hash_id, node_uuid, name in zip(
            df["hash_id"].to_list(), df["node_uuid"].to_list(), names
        ):
            hash_to_node[hash_id] = (node_uuid, "SUBJECT_PROCESS", name)
        print(f"  Loaded {len(df)} process nodes")
        del df

    # File nodes
    print("Loading file nodes...")
    df = _read_table(
        data_dir, "file_node_table", columns=["hash_id", "node_uuid", "path"]
    )
    if df is not None:
        names = (
            df["path"].fill_null("").to_list()
            if "path" in df.columns
            else [""] * len(df)
        )

        for hash_id, node_uuid, name in zip(
            df["hash_id"].to_list(), df["node_uuid"].to_list(), names
        ):
            hash_to_node[hash_id] = (node_uuid, "FILE_OBJECT", name)
        print(f"  Loaded {len(df)} file nodes")
        del df

    # Netflow nodes
    print("Loading netflow nodes...")
    df = _read_table(
        data_dir,
        "netflow_node_table",
        columns=["hash_id", "node_uuid", "src_addr", "src_port"],
    )
    if df is not None:
        if "src_addr" in df.columns and "src_port" in df.columns:
            names = (
                df["src_addr"].fill_null("")
                + ":"
                + df["src_port"].cast(pl.Utf8).fill_null("")
            ).to_list()
        else:
            names = [""] * len(df)

        for hash_id, node_uuid, name in zip(
            df["hash_id"].to_list(), df["node_uuid"].to_list(), names
        ):
            hash_to_node[hash_id] = (node_uuid, "NETFLOW_OBJECT", name)
        print(f"  Loaded {len(df)} netflow nodes")
        del df

    gc.collect()
    print(f"Total nodes cached: {len(hash_to_node)}")
    return hash_to_node


def get_day_boundaries(day_number):
    """Get start/end timestamps (ns) for a given day in April 2018."""
    start_dt = datetime(2018, 4, day_number, 0, 0, 0)
    start_ns = int(start_dt.timestamp() * 1_000_000_000)
    end_ns = start_ns + NANOSECONDS_PER_DAY
    return start_ns, end_ns


def build_graph_for_day(
    data_dir, day_number, hash_to_node_cache, node_type_dict, edge_type_dict
):
    """
    Build graph for a day using lazy scan with filter pushdown.
    Parquet format enables row group skipping based on timestamp statistics.
    """
    start_ns, end_ns = get_day_boundaries(day_number)

    # Lazy scan with predicate pushdown - Parquet can skip entire row groups
    lazy_df = _scan_table(data_dir, "event_table")
    if lazy_df is None:
        print(f"  WARNING: No event table found in {data_dir}")
        return nx.DiGraph(), node_type_dict, edge_type_dict

    print(f"  Building graph for day {day_number}...")

    # Filter and select only needed columns, use streaming for memory efficiency
    df = (
        lazy_df.filter(
            (pl.col("timestamp_rec").cast(pl.Int64) >= start_ns)
            & (pl.col("timestamp_rec").cast(pl.Int64) < end_ns)
        )
        .select(["src_node", "dst_node", "operation"])
        .collect(engine="streaming")
    )

    events_in_day = len(df)
    print(f"    Filtered {events_in_day} events")

    if events_in_day == 0:
        return nx.DiGraph(), node_type_dict, edge_type_dict

    # Pre-allocate type counters
    node_type_cnt = max(node_type_dict.values()) + 1 if node_type_dict else 0
    edge_type_cnt = max(edge_type_dict.values()) + 1 if edge_type_dict else 0

    # Build graph using direct iteration (avoids creating 3 separate lists)
    g = nx.DiGraph()
    hash_to_idx = {}
    node_idx = 0
    edge_count = 0

    # Process in chunks to reduce memory pressure for large days
    chunk_size = 500_000
    n_rows = len(df)

    for chunk_start in range(0, n_rows, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n_rows)
        chunk = df.slice(chunk_start, chunk_end - chunk_start)

        src_nodes = chunk["src_node"].to_list()
        dst_nodes = chunk["dst_node"].to_list()
        operations = chunk["operation"].to_list()

        for src_hash, dst_hash, operation in zip(src_nodes, dst_nodes, operations):
            if src_hash not in hash_to_node_cache or dst_hash not in hash_to_node_cache:
                continue

            src_uuid, src_type, src_name = hash_to_node_cache[src_hash]
            dst_uuid, dst_type, dst_name = hash_to_node_cache[dst_hash]

            # Map types to IDs (lazy initialization)
            if src_type not in node_type_dict:
                node_type_dict[src_type] = node_type_cnt
                node_type_cnt += 1
            if dst_type not in node_type_dict:
                node_type_dict[dst_type] = node_type_cnt
                node_type_cnt += 1
            if operation not in edge_type_dict:
                edge_type_dict[operation] = edge_type_cnt
                edge_type_cnt += 1

            src_type_id = node_type_dict[src_type]
            dst_type_id = node_type_dict[dst_type]
            edge_type_id = edge_type_dict[operation]

            # Add nodes
            if src_hash not in hash_to_idx:
                hash_to_idx[src_hash] = node_idx
                g.add_node(node_idx, type=src_type_id, uuid=src_uuid, name=src_name)
                node_idx += 1
            if dst_hash not in hash_to_idx:
                hash_to_idx[dst_hash] = node_idx
                g.add_node(node_idx, type=dst_type_id, uuid=dst_uuid, name=dst_name)
                node_idx += 1

            # Add edge (deduplicated)
            src_idx = hash_to_idx[src_hash]
            dst_idx = hash_to_idx[dst_hash]
            if not g.has_edge(src_idx, dst_idx):
                g.add_edge(src_idx, dst_idx, type=edge_type_id)
                edge_count += 1

        del chunk, src_nodes, dst_nodes, operations

    del df
    gc.collect()

    print(f"  Day {day_number}: {g.number_of_nodes()} nodes, {edge_count} edges")
    return g, node_type_dict, edge_type_dict


def load_groundtruth(groundtruth_file, label_filter="attack"):
    """
    Load ground truth from REAPR CSV format.
    Returns attack/contaminated UUID sets and UUID->attack_chain mappings for ADP metric.
    """
    attack_uuids = set()
    contaminated_uuids = set()
    uuid_to_attack_chain_attack_only = {}
    uuid_to_attack_chain_all = {}

    excluded_attack_chains = {"wwtawwtal_bad_neighborhood"}  # CADETS artifact

    if not os.path.exists(groundtruth_file):
        print(f"Warning: Ground truth file not found: {groundtruth_file}")
        return (
            attack_uuids,
            contaminated_uuids,
            uuid_to_attack_chain_attack_only,
            uuid_to_attack_chain_all,
        )

    df = pl.read_csv(groundtruth_file)
    excluded_count = 0

    for row in df.iter_rows(named=True):
        uuid = row["uuid"].strip()
        label = row["label"].strip()
        attack_chain = (
            row.get("attack_chain", "").strip() if row.get("attack_chain") else ""
        )

        if attack_chain in excluded_attack_chains:
            excluded_count += 1
            continue

        if label == "attack":
            attack_uuids.add(uuid)
            if attack_chain:
                uuid_to_attack_chain_attack_only[uuid] = attack_chain
                uuid_to_attack_chain_all[uuid] = attack_chain
        elif label == "contaminated":
            contaminated_uuids.add(uuid)
            if attack_chain:
                uuid_to_attack_chain_all[uuid] = attack_chain

    print(
        f"Ground truth: {len(attack_uuids)} attack, {len(contaminated_uuids)} contaminated"
    )
    print(
        f"  Attack chains: {len(set(uuid_to_attack_chain_attack_only.values()))} (L2), {len(set(uuid_to_attack_chain_all.values()))} (L1)"
    )
    if excluded_count > 0:
        print(f"  Excluded {excluded_count} from: {excluded_attack_chains}")

    return (
        attack_uuids,
        contaminated_uuids,
        uuid_to_attack_chain_attack_only,
        uuid_to_attack_chain_all,
    )


def process_single_day(
    data_dir,
    day_number,
    attack_uuids,
    contaminated_uuids,
    uuid_to_attack_chain_attack_only,
    uuid_to_attack_chain_all,
    node_type_dict,
    edge_type_dict,
    output_dir,
    split_name,
    split_idx,
    hash_to_node_cache,
):
    """Process a single day: build graph, extract labels, save to disk."""
    print(f"\nProcessing day {day_number} ({split_name}{split_idx})...")

    g, node_type_dict, edge_type_dict = build_graph_for_day(
        data_dir, day_number, hash_to_node_cache, node_type_dict, edge_type_dict
    )

    if g.number_of_nodes() == 0:
        print(f"  WARNING: Day {day_number} has no nodes - skipping!")
        del g
        gc.collect()
        return 0, [], [], {}, {}, {}, {}, node_type_dict, edge_type_dict

    # Find malicious nodes (only SUBJECT_PROCESS nodes)
    attack_indices = []
    contaminated_indices = []
    node_names = {}
    node_to_attack_chain_attack_only = {}
    node_to_attack_chain_all = {}

    process_type_id = node_type_dict.get("SUBJECT_PROCESS")

    for node_idx in g.nodes():
        node_uuid = g.nodes[node_idx].get("uuid", "")
        node_name = g.nodes[node_idx].get("name", "")
        node_type_id = g.nodes[node_idx].get("type")

        # Skip non-process nodes for malicious labeling
        if process_type_id is not None and node_type_id != process_type_id:
            continue

        if node_uuid in attack_uuids:
            attack_indices.append(node_idx)
            node_names[node_idx] = node_name if node_name else node_uuid
            if node_uuid in uuid_to_attack_chain_attack_only:
                node_to_attack_chain_attack_only[node_idx] = (
                    uuid_to_attack_chain_attack_only[node_uuid]
                )
            if node_uuid in uuid_to_attack_chain_all:
                node_to_attack_chain_all[node_idx] = uuid_to_attack_chain_all[node_uuid]
        elif node_uuid in contaminated_uuids:
            contaminated_indices.append(node_idx)
            node_names[node_idx] = node_name if node_name else node_uuid
            if node_uuid in uuid_to_attack_chain_all:
                node_to_attack_chain_all[node_idx] = uuid_to_attack_chain_all[node_uuid]

    print(
        f"  Malicious nodes: {len(attack_indices)} attack, {len(contaminated_indices)} contaminated"
    )
    if node_to_attack_chain_attack_only or node_to_attack_chain_all:
        print(
            f"  Attack chains (attack-only): {len(set(node_to_attack_chain_attack_only.values()))}, (all): {len(set(node_to_attack_chain_all.values()))}"
        )

    # Extract idx->UUID mapping for entity-level aggregation
    idx_to_uuid = {
        node_idx: g.nodes[node_idx].get("uuid", "")
        for node_idx in g.nodes()
        if g.nodes[node_idx].get("uuid", "")
    }

    # Convert to DGL and save
    dgl_g = dgl.from_networkx(g, node_attrs=["type"], edge_attrs=["type"])
    n_nodes = dgl_g.number_of_nodes()

    del g
    gc.collect()

    graph_pkl = os.path.join(output_dir, f"{split_name}{split_idx}.pkl")
    with open(graph_pkl, "wb") as f:
        pkl.dump(dgl_g, f)
    print(f"  Saved {graph_pkl}")

    del dgl_g
    gc.collect()

    return (
        n_nodes,
        attack_indices,
        contaminated_indices,
        node_names,
        node_to_attack_chain_attack_only,
        node_to_attack_chain_all,
        idx_to_uuid,
        node_type_dict,
        edge_type_dict,
    )


def add_onehot_features_to_graphs(
    output_dir, n_train, n_val, n_test, node_feature_dim, edge_feature_dim
):
    """Add one-hot features to all graphs in a second pass."""
    print("\nAdding one-hot features to graphs...")

    for split_name, n_graphs in [("train", n_train), ("val", n_val), ("test", n_test)]:
        for i in range(n_graphs):
            graph_pkl = os.path.join(output_dir, f"{split_name}{i}.pkl")
            print(f"  Processing {graph_pkl}...")

            with open(graph_pkl, "rb") as f:
                g = pkl.load(f)

            g.ndata["attr"] = F.one_hot(
                g.ndata["type"].view(-1).long(), num_classes=node_feature_dim
            ).float()
            g.edata["attr"] = F.one_hot(
                g.edata["type"].view(-1).long(), num_classes=edge_feature_dim
            ).float()

            with open(graph_pkl, "wb") as f:
                pkl.dump(g, f)

            del g
            gc.collect()

    print("  Done.")


def process_dataset(
    dataset, data_dir, ground_truth, label_filter="attack", output_dir=None
):
    """
    Process a dataset by splitting into daily graphs.
    Supports both Parquet and CSV input formats (Parquet preferred for speed).
    """
    if dataset not in DATASET_SPLITS:
        raise ValueError(
            f"Unknown dataset: {dataset}. Available: {list(DATASET_SPLITS.keys())}"
        )

    split_config = DATASET_SPLITS[dataset]

    print(f"\n{'=' * 60}")
    print(f"Processing dataset: {dataset}")
    print(f"Data directory: {data_dir}")
    print(f"Ground truth: {ground_truth}")
    print(f"Train days: {split_config['train_days']}")
    print(f"Val days: {split_config['val_days']}")
    print(f"Test days: {split_config['test_days']}")
    print(f"{'=' * 60}\n")

    # Load ground truth
    (
        attack_uuids,
        contaminated_uuids,
        uuid_to_attack_chain_attack_only,
        uuid_to_attack_chain_all,
    ) = load_groundtruth(ground_truth, label_filter=label_filter)

    # Load and cache all node tables once
    print("\nLoading and caching all node tables...")
    hash_to_node_cache = load_all_node_tables(data_dir)

    if output_dir is None:
        # Default: <MAGIC_ROOT>/data/<dataset>/
        _magic_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_dir = os.path.join(_magic_root, "data", dataset)
    os.makedirs(output_dir, exist_ok=True)

    node_type_dict = {}
    edge_type_dict = {}

    # Initialize tracking for val/test splits
    val_info = {
        "attack": [],
        "contaminated": [],
        "names": {},
        "node_to_attack_chain_attack_only": {},
        "node_to_attack_chain_all": {},
        "idx_to_uuid": {},
        "n_nodes": 0,
    }
    test_info = {
        "attack": [],
        "contaminated": [],
        "names": {},
        "node_to_attack_chain_attack_only": {},
        "node_to_attack_chain_all": {},
        "idx_to_uuid": {},
        "n_nodes": 0,
    }

    n_train = n_val = n_test = 0

    # Process training days
    print("\n" + "=" * 40)
    print("PROCESSING TRAINING DAYS")
    print("=" * 40)
    train_idx = 0
    for day in split_config["train_days"]:
        n_nodes, _, _, _, _, _, _, node_type_dict, edge_type_dict = process_single_day(
            data_dir,
            day,
            attack_uuids,
            contaminated_uuids,
            uuid_to_attack_chain_attack_only,
            uuid_to_attack_chain_all,
            node_type_dict,
            edge_type_dict,
            output_dir,
            "train",
            train_idx,
            hash_to_node_cache,
        )
        if n_nodes > 0:
            train_idx += 1
    n_train = train_idx

    # Process validation days
    print("\n" + "=" * 40)
    print("PROCESSING VALIDATION DAYS")
    print("=" * 40)
    node_offset = 0
    val_idx = 0
    for day in split_config["val_days"]:
        (
            n_nodes,
            attack_idx,
            contam_idx,
            names,
            node_to_attack_chain_attack_only,
            node_to_attack_chain_all,
            idx_to_uuid,
            node_type_dict,
            edge_type_dict,
        ) = process_single_day(
            data_dir,
            day,
            attack_uuids,
            contaminated_uuids,
            uuid_to_attack_chain_attack_only,
            uuid_to_attack_chain_all,
            node_type_dict,
            edge_type_dict,
            output_dir,
            "val",
            val_idx,
            hash_to_node_cache,
        )
        if n_nodes > 0:
            val_info["attack"].extend([idx + node_offset for idx in attack_idx])
            val_info["contaminated"].extend([idx + node_offset for idx in contam_idx])
            val_info["names"].update(
                {idx + node_offset: name for idx, name in names.items()}
            )
            val_info["node_to_attack_chain_attack_only"].update(
                {
                    idx + node_offset: chain
                    for idx, chain in node_to_attack_chain_attack_only.items()
                }
            )
            val_info["node_to_attack_chain_all"].update(
                {
                    idx + node_offset: chain
                    for idx, chain in node_to_attack_chain_all.items()
                }
            )
            val_info["idx_to_uuid"].update(
                {idx + node_offset: uuid for idx, uuid in idx_to_uuid.items()}
            )
            node_offset += n_nodes
            val_idx += 1
    val_info["n_nodes"] = node_offset
    n_val = val_idx

    # Process test days
    print("\n" + "=" * 40)
    print("PROCESSING TEST DAYS")
    print("=" * 40)
    node_offset = 0
    test_idx = 0
    for day in split_config["test_days"]:
        (
            n_nodes,
            attack_idx,
            contam_idx,
            names,
            node_to_attack_chain_attack_only,
            node_to_attack_chain_all,
            idx_to_uuid,
            node_type_dict,
            edge_type_dict,
        ) = process_single_day(
            data_dir,
            day,
            attack_uuids,
            contaminated_uuids,
            uuid_to_attack_chain_attack_only,
            uuid_to_attack_chain_all,
            node_type_dict,
            edge_type_dict,
            output_dir,
            "test",
            test_idx,
            hash_to_node_cache,
        )
        if n_nodes > 0:
            test_info["attack"].extend([idx + node_offset for idx in attack_idx])
            test_info["contaminated"].extend([idx + node_offset for idx in contam_idx])
            test_info["names"].update(
                {idx + node_offset: name for idx, name in names.items()}
            )
            test_info["node_to_attack_chain_attack_only"].update(
                {
                    idx + node_offset: chain
                    for idx, chain in node_to_attack_chain_attack_only.items()
                }
            )
            test_info["node_to_attack_chain_all"].update(
                {
                    idx + node_offset: chain
                    for idx, chain in node_to_attack_chain_all.items()
                }
            )
            test_info["idx_to_uuid"].update(
                {idx + node_offset: uuid for idx, uuid in idx_to_uuid.items()}
            )
            node_offset += n_nodes
            test_idx += 1
    test_info["n_nodes"] = node_offset
    n_test = test_idx

    # Add one-hot features
    node_feature_dim = max(node_type_dict.values()) + 1 if node_type_dict else 1
    edge_feature_dim = max(edge_type_dict.values()) + 1 if edge_type_dict else 1
    print(f"\nFeature dimensions: nodes={node_feature_dim}, edges={edge_feature_dim}")

    add_onehot_features_to_graphs(
        output_dir, n_train, n_val, n_test, node_feature_dim, edge_feature_dim
    )

    # Free cache
    del hash_to_node_cache
    gc.collect()

    # Save ground truth
    ground_truth_data = {
        "val": {
            "attack": val_info["attack"],
            "contaminated": val_info["contaminated"],
            "names": val_info["names"],
            "node_to_attack_chain_attack_only": val_info[
                "node_to_attack_chain_attack_only"
            ],
            "node_to_attack_chain_all": val_info["node_to_attack_chain_all"],
            "idx_to_uuid": val_info["idx_to_uuid"],
        },
        "test": {
            "attack": test_info["attack"],
            "contaminated": test_info["contaminated"],
            "names": test_info["names"],
            "node_to_attack_chain_attack_only": test_info[
                "node_to_attack_chain_attack_only"
            ],
            "node_to_attack_chain_all": test_info["node_to_attack_chain_all"],
            "idx_to_uuid": test_info["idx_to_uuid"],
        },
    }

    ground_truth_pkl = os.path.join(output_dir, "ground_truth.pkl")
    with open(ground_truth_pkl, "wb") as f:
        pkl.dump(ground_truth_data, f)
    print(f"Saved ground truth to {ground_truth_pkl}")

    # Save metadata
    metadata = {
        "node_feature_dim": node_feature_dim,
        "edge_feature_dim": edge_feature_dim,
        "node_type_dict": node_type_dict,
        "two_level_labels": {
            "val": {
                "attack": val_info["attack"],
                "contaminated": val_info["contaminated"],
                "node_to_attack_chain_attack_only": {
                    str(k): v
                    for k, v in val_info["node_to_attack_chain_attack_only"].items()
                },
                "node_to_attack_chain_all": {
                    str(k): v for k, v in val_info["node_to_attack_chain_all"].items()
                },
                "idx_to_uuid": {str(k): v for k, v in val_info["idx_to_uuid"].items()},
                "node_to_attack_chain": {
                    str(k): v for k, v in val_info["node_to_attack_chain_all"].items()
                },
            },
            "test": {
                "attack": test_info["attack"],
                "contaminated": test_info["contaminated"],
                "node_to_attack_chain_attack_only": {
                    str(k): v
                    for k, v in test_info["node_to_attack_chain_attack_only"].items()
                },
                "node_to_attack_chain_all": {
                    str(k): v for k, v in test_info["node_to_attack_chain_all"].items()
                },
                "idx_to_uuid": {str(k): v for k, v in test_info["idx_to_uuid"].items()},
                "node_to_attack_chain": {
                    str(k): v for k, v in test_info["node_to_attack_chain_all"].items()
                },
            },
        },
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
    }
    metadata_json = os.path.join(output_dir, "metadata.json")
    with open(metadata_json, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {metadata_json}")

    print(f"\n{'=' * 60}")
    print(f"Dataset {dataset} processing complete!")
    print(
        f"  Val - Attack: {len(val_info['attack'])}, Contaminated: {len(val_info['contaminated'])}"
    )
    print(f"        Unique entities: {len(set(val_info['idx_to_uuid'].values()))}")
    print(
        f"  Test - Attack: {len(test_info['attack'])}, Contaminated: {len(test_info['contaminated'])}"
    )
    print(f"         Unique entities: {len(set(test_info['idx_to_uuid'].values()))}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parse DARPA TC datasets into daily DGL graphs (supports Parquet/CSV)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["cadets", "fivedirections", "theia", "trace"],
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing Parquet or CSV tables",
    )
    parser.add_argument(
        "--ground_truth", type=str, required=True, help="Path to REAPR ground truth CSV"
    )
    parser.add_argument(
        "--label_filter",
        type=str,
        default="attack",
        choices=["attack", "contaminated", "both"],
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for processed graphs (default: <MAGIC_ROOT>/data/<dataset>/)",
    )
    args = parser.parse_args()

    process_dataset(
        args.dataset,
        args.data_dir,
        args.ground_truth,
        args.label_filter,
        args.output_dir,
    )
