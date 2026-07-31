import csv
import glob
import os.path
from collections import defaultdict
from typing import Optional

import polars as pl

from pidsmaker.utils.utils import (
    datetime_to_ns_time_US,
    datetime_to_ns_time_with_tz,
    log,
    read_table,
    scan_table,
)


def _safe_int(value):
    try:
        if value in (None, ""):
            return None
        string_value = str(value).strip()
        if string_value.endswith(".0"):
            string_value = string_value[:-2]
        return int(string_value)
    except (TypeError, ValueError):
        return None


def _normalize_windows_path(path: str) -> str:
    normalized = (path or "").strip().lower().replace("\\", "/")
    while "//" in normalized:
        normalized = normalized.replace("//", "/")
    return normalized


def _is_atlasv2_dataset(cfg) -> bool:
    return str(getattr(cfg.dataset, "name", "")).lower().startswith("atlasv2_")


def _normalize_atlas_attack_id(attack_id: str) -> str:
    attack_id = (attack_id or "").strip().lower()
    if attack_id and not attack_id.startswith("atlasv2/"):
        attack_id = f"atlasv2/{attack_id}"
    return attack_id


def _infer_atlas_host_from_dataset_name(dataset_name: str) -> Optional[str]:
    dataset_name = (dataset_name or "").lower()
    if dataset_name.endswith("_h1"):
        return "h1"
    if dataset_name.endswith("_h2"):
        return "h2"
    return None


def _datetime_to_ns_time_for_dataset(cfg, date_str: str) -> int:
    timezone = getattr(cfg.dataset, "timezone", "US/Eastern")
    return datetime_to_ns_time_with_tz(date_str, timezone)


def parse_atlasv2_ground_truth(cfg):
    reapr_gt_path = getattr(cfg.dataset, "reapr_ground_truth_path", None)
    if reapr_gt_path is None:
        return None, None, None, None, None, None, None

    gt_file = os.path.join(cfg._ground_truth_dir, reapr_gt_path)
    label_dir = os.environ.get("ATLASV2_LABEL_DIR", os.path.dirname(gt_file))
    legacy_file = os.path.join(label_dir, "atlasv2_labels.csv")
    revised_files = sorted(glob.glob(os.path.join(label_dir, "*.labels")))
    label_files = revised_files or (
        [legacy_file] if os.path.exists(legacy_file) else []
    )
    if not label_files:
        log(f"ATLASv2 ground truth not found under: {label_dir}")
        return None, None, None, None, None, None, None

    process_df = read_table(
        cfg.dataset.csv_dir,
        "process_node_table",
        columns=["node_uuid", "index_id", "path", "pid", "attack"],
    )
    if process_df is None:
        log(f"process_node_table missing for dataset {cfg.dataset.name}")
        return None, None, None, None, None, None, None

    dataset_host = _infer_atlas_host_from_dataset_name(cfg.dataset.name)
    if dataset_host is None:
        raise ValueError(f"Could not infer ATLASv2 host from dataset name '{cfg.dataset.name}'")

    attack_chains = defaultdict(
        lambda: {"attack_nids": set(), "contaminated_nids": set(), "all_nids": set(), "paths": {}}
    )
    all_attack_nids = set()
    all_contaminated_nids = set()
    all_nids = set()
    all_paths = {}
    uuid_to_node_id = {}
    excluded_nids = set()

    by_attack_pid_path = defaultdict(set)
    by_host_pid_path = defaultdict(set)
    by_process_uuid = defaultdict(set)
    node_id_to_path = {}

    for row in process_df.iter_rows(named=True):
        node_id = row["index_id"]
        pid = _safe_int(row.get("pid"))
        path = _normalize_windows_path(row.get("path", ""))
        attack_id = (row.get("attack") or "").strip()
        node_uuid = (row.get("node_uuid") or "").strip()

        if node_uuid and node_id is not None:
            uuid_to_node_id[node_uuid] = str(node_id)

        if node_id is None:
            continue

        node_id = int(node_id)
        node_id_to_path[node_id] = row.get("path", "")
        process_uuid = node_uuid.split("|")[-1].strip().lower()
        if process_uuid:
            by_process_uuid[process_uuid].add(node_id)

        if attack_id and pid is not None and path:
            by_attack_pid_path[(attack_id, pid, path)].add(node_id)
        if pid is not None and path:
            by_host_pid_path[(dataset_host, pid, path)].add(node_id)

    unmatched_rows = 0
    for label_file in label_files:
        with open(label_file, "r") as f:
            reader = csv.DictReader(f, skipinitialspace=True)
            for raw_row in reader:
                if not raw_row:
                    continue

                row = {
                    (key or "").strip().lower(): (value or "").strip()
                    for key, value in raw_row.items()
                }
                attack_id = _normalize_atlas_attack_id(row.get("attack", ""))
                label = row.get("label", "").lower()
                process_uuid = (row.get("process_uuid", "") or "").strip().lower()
                pid = _safe_int(row.get("process_id"))
                path = _normalize_windows_path(row.get("process_name", ""))

                if not attack_id.startswith(f"atlasv2/{dataset_host}-"):
                    continue
                if label not in {"attack", "contaminated"}:
                    continue

                if process_uuid:
                    matches = by_process_uuid.get(process_uuid)
                    if matches and len(matches) > 1:
                        raise ValueError(
                            f"ATLASv2 process UUID '{process_uuid}' maps to multiple nodes"
                        )
                elif pid is not None and path:
                    matches = by_attack_pid_path.get((attack_id, pid, path))
                    if not matches:
                        matches = by_host_pid_path.get((dataset_host, pid, path))
                else:
                    matches = None

                if not matches:
                    unmatched_rows += 1
                    continue

                attack_chain = attack_chains[attack_id]
                for node_id in matches:
                    attack_chain["all_nids"].add(node_id)
                    attack_chain["paths"][node_id] = node_id_to_path.get(node_id, "")
                    all_nids.add(node_id)
                    all_paths[node_id] = node_id_to_path.get(node_id, "")

                    if label == "attack":
                        attack_chain["attack_nids"].add(node_id)
                        all_attack_nids.add(node_id)
                    else:
                        attack_chain["contaminated_nids"].add(node_id)
                        all_contaminated_nids.add(node_id)

    log(
        f"ATLASv2 ground truth parsed for {cfg.dataset.name}: "
        f"{len(attack_chains)} attack chains, "
        f"{len(all_attack_nids)} attack nodes, "
        f"{len(all_contaminated_nids)} contaminated nodes, "
        f"{unmatched_rows} unmatched label rows; "
        f"source={'revised UUID labels' if revised_files else 'legacy CSV'}"
    )

    return (
        dict(attack_chains),
        all_attack_nids,
        all_contaminated_nids,
        all_nids,
        all_paths,
        uuid_to_node_id,
        excluded_nids,
    )


def load_node_tables(cfg):
    """
    Load node tables from Parquet or CSV files using Polars.
    Node tables are small enough to load into memory.
    Returns DataFrames for netflow, subject, and file node tables.
    """
    data_dir = cfg.dataset.csv_dir
    if not data_dir:
        raise ValueError(
            f"csv_dir is not configured for dataset {cfg.dataset.name}. "
            "Please set dataset.csv_dir in the config file."
        )

    netflow_df = read_table(data_dir, "netflow_node_table")
    subject_df = read_table(data_dir, "process_node_table")
    file_df = read_table(data_dir, "file_node_table")

    if netflow_df is None:
        netflow_df = pl.DataFrame()
    if subject_df is None:
        subject_df = pl.DataFrame()
    if file_df is None:
        file_df = pl.DataFrame()

    return netflow_df, subject_df, file_df


def get_event_table_path(cfg):
    """Get the path to the event table file (Parquet preferred, CSV fallback)."""
    data_dir = cfg.dataset.csv_dir
    if not data_dir:
        raise ValueError(
            f"csv_dir is not configured for dataset {cfg.dataset.name}. "
            "Please set dataset.csv_dir in the config file."
        )

    parquet_path = os.path.join(data_dir, "event_table.parquet")
    csv_path = os.path.join(data_dir, "event_table.csv")

    if os.path.exists(parquet_path):
        return parquet_path, "parquet"
    return csv_path, "csv"


def stream_events_for_time_range(cfg, start_time, end_time):
    """
    Stream events for a given time range using Polars lazy evaluation.
    Supports both Parquet and CSV formats.
    This avoids loading the entire event table into memory.

    Yields rows as tuples.
    """
    data_dir = cfg.dataset.csv_dir
    lazy_df = scan_table(data_dir, "event_table")

    if lazy_df is None:
        return

    # Get column names
    schema = lazy_df.collect_schema()
    cols = list(schema.names())
    timestamp_col = cols[6]  # timestamp_rec column

    # Filter lazily, then collect in streaming mode
    filtered = lazy_df.filter(
        (pl.col(timestamp_col) >= start_time) & (pl.col(timestamp_col) <= end_time)
    )

    # Collect with streaming to process in batches
    result_df = filtered.collect(streaming=True)

    # Yield rows
    for row in result_df.iter_rows():
        yield row


def get_uuid2nids_from_csv(cfg):
    """
    Build uuid2nids and nid2uuid mappings from CSV node tables.
    """
    netflow_df, subject_df, file_df = load_node_tables(cfg)

    uuid2nids = {}
    nid2uuid = {}

    # Process each node table
    # Columns: node_uuid, hash_id, ..., index_id (last column)
    for df in [file_df, netflow_df, subject_df]:
        cols = df.columns
        uuid_col = cols[0]  # node_uuid
        index_col = cols[-1]  # index_id

        for row in df.iter_rows(named=True):
            node_uuid = row[uuid_col]
            index_id = row[index_col]
            if node_uuid and index_id is not None:
                uuid2nids[node_uuid] = index_id
                nid2uuid[index_id] = node_uuid

    return uuid2nids, nid2uuid


def parse_reapr_ground_truth(cfg):
    """
    Parse REAPr-style ground truth CSV file directly.

    The CSV has columns: attack_chain, uuid, process_name, label
    - attack_chain: groups nodes by attack (e.g., 'theia_3.3_fail_allstate')
    - uuid: node UUID
    - process_name: process path/name
    - label: 'attack' or 'contaminated'

    Attack chains listed in cfg.dataset.excluded_attack_chains will be completely
    excluded from the evaluation (their nodes are treated as neither positive nor
    negative - they are simply removed from the evaluation set).

    Returns:
        attack_chains: dict mapping attack_chain_name -> {
            'attack_nids': set of node IDs labeled as 'attack',
            'contaminated_nids': set of node IDs labeled as 'contaminated',
            'all_nids': set of all node IDs,
            'paths': dict mapping node_id -> process_name
        }
        all_attack_nids: set of all attack node IDs across all chains
        all_contaminated_nids: set of all contaminated node IDs across all chains
        all_nids: set of all ground truth node IDs
        all_paths: dict mapping node_id -> process_name
        uuid_to_node_id: dict mapping UUID -> node_id string
        excluded_nids: set of node IDs from excluded attack chains (to be masked in eval)
    """
    if _is_atlasv2_dataset(cfg):
        return parse_atlasv2_ground_truth(cfg)

    uuid2nids, _ = get_uuid2nids_from_csv(cfg)

    reapr_gt_path = getattr(cfg.dataset, "reapr_ground_truth_path", None)
    if reapr_gt_path is None:
        return None, None, None, None, None, None, None

    gt_file = os.path.join(cfg._ground_truth_dir, reapr_gt_path)
    if not os.path.exists(gt_file):
        log(f"REAPr ground truth file not found: {gt_file}")
        return None, None, None, None, None, None, None

    # Get list of attack chains to exclude from evaluation
    excluded_attack_chains = set(getattr(cfg.dataset, "excluded_attack_chains", []))
    if excluded_attack_chains:
        log(f"Excluding attack chains from evaluation: {excluded_attack_chains}")

    attack_chains = defaultdict(
        lambda: {"attack_nids": set(), "contaminated_nids": set(), "all_nids": set(), "paths": {}}
    )

    all_attack_nids = set()
    all_contaminated_nids = set()
    all_nids = set()
    all_paths = {}
    uuid_to_node_id = {}
    excluded_nids = set()  # Nodes from excluded attack chains

    log(f"Parsing REAPr ground truth from {gt_file}")

    with open(gt_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            attack_chain = row["attack_chain"].strip()
            node_uuid = row["uuid"].strip()
            process_name = row.get("process_name", "").strip()
            label = row["label"].strip().lower()

            if node_uuid not in uuid2nids:
                # Node not found in CSV tables, skip
                continue

            node_id = int(uuid2nids[node_uuid])
            uuid_to_node_id[node_uuid] = str(node_id)

            # Check if this attack chain should be excluded from evaluation
            if attack_chain in excluded_attack_chains:
                excluded_nids.add(node_id)
                # Still track in attack_chains for reference, but don't add to global sets
                attack_chains[attack_chain]["all_nids"].add(node_id)
                attack_chains[attack_chain]["paths"][node_id] = process_name
                if label == "attack":
                    attack_chains[attack_chain]["attack_nids"].add(node_id)
                elif label == "contaminated":
                    attack_chains[attack_chain]["contaminated_nids"].add(node_id)
                continue

            # Add to attack chain
            attack_chains[attack_chain]["all_nids"].add(node_id)
            attack_chains[attack_chain]["paths"][node_id] = process_name

            # Add to global sets
            all_nids.add(node_id)
            all_paths[node_id] = process_name

            if label == "attack":
                attack_chains[attack_chain]["attack_nids"].add(node_id)
                all_attack_nids.add(node_id)
            elif label == "contaminated":
                attack_chains[attack_chain]["contaminated_nids"].add(node_id)
                all_contaminated_nids.add(node_id)

    log(
        f"REAPr ground truth parsed: {len(attack_chains)} attack chains, "
        f"{len(all_attack_nids)} attack nodes, "
        f"{len(all_contaminated_nids)} contaminated nodes, "
        f"{len(all_nids)} total nodes"
    )
    if excluded_nids:
        log(
            f"Excluded {len(excluded_nids)} nodes from {len(excluded_attack_chains)} attack chain(s)"
        )

    return (
        dict(attack_chains),
        all_attack_nids,
        all_contaminated_nids,
        all_nids,
        all_paths,
        uuid_to_node_id,
        excluded_nids,
    )


def get_ground_truth(cfg):
    """
    Get ground truth node IDs and paths.

    If reapr_ground_truth_path is configured, parses the REAPr CSV directly.
    Otherwise falls back to legacy per-attack CSV files.

    Returns:
        ground_truth_nids: set of all ground truth node IDs
        ground_truth_paths: dict mapping node_id -> path/label string
        uuid_to_node_id: dict mapping UUID -> node_id string
    """
    # Try REAPr ground truth first
    result = parse_reapr_ground_truth(cfg)
    if result[0] is not None:
        (
            attack_chains,
            all_attack_nids,
            all_contaminated_nids,
            all_nids,
            all_paths,
            uuid_to_node_id,
            excluded_nids,
        ) = result
        return all_nids, all_paths, uuid_to_node_id

    # Fall back to legacy parsing from separate CSV files
    uuid2nids, nid2uuid = get_uuid2nids_from_csv(cfg)

    ground_truth_nids, ground_truth_paths = [], {}
    uuid_to_node_id = {}
    for file in cfg.dataset.ground_truth_relative_path:
        with open(os.path.join(cfg._ground_truth_dir, file), "r") as f:
            reader = csv.reader(f)
            for row in reader:
                node_uuid, node_labels, _ = row[0], row[1], row[2]
                node_id = uuid2nids[node_uuid]
                ground_truth_nids.append(int(node_id))
                ground_truth_paths[int(node_id)] = node_labels
                uuid_to_node_id[node_uuid] = str(node_id)

    mimicry_edge_num = cfg.preprocessing.build_graphs.mimicry_edge_num
    if mimicry_edge_num is not None and mimicry_edge_num > 0:
        num_GPs = len(ground_truth_nids)
        for file in cfg.dataset.ground_truth_relative_path:
            file_name = file.split("/")[-1]
            with open(
                os.path.join(cfg.preprocessing.build_graphs._mimicry_dir, file_name), "r"
            ) as f:
                reader = csv.reader(f)
                for row in reader:
                    node_uuid, node_labels, _ = row[0], row[1], row[2]
                    node_id = uuid2nids[node_uuid]
                    ground_truth_nids.append(int(node_id))
                    ground_truth_paths[int(node_id)] = node_labels
                    uuid_to_node_id[node_uuid] = str(node_id)
        num_mimicry_GPs = len(ground_truth_nids) - num_GPs
        log(f"{num_mimicry_GPs} mimicry ground truth nodes loaded")

    return set(ground_truth_nids), ground_truth_paths, uuid_to_node_id


def get_ground_truth_with_labels(cfg):
    """
    Load ground truth from REAPr-style CSV files that distinguish between
    'attack' (attack chain) and 'contaminated' (downstream effect) nodes.

    Returns:
        attack_nids: set of node IDs labeled as 'attack' (strict malicious)
        contaminated_nids: set of node IDs labeled as 'contaminated'
        all_positive_nids: set of all positive node IDs (attack + contaminated)
        ground_truth_paths: dict mapping node_id -> path/label string
        uuid_to_node_id: dict mapping UUID -> node_id string
        excluded_nids: set of node IDs from excluded attack chains (to be masked in eval)
    """
    # Use shared parsing function
    result = parse_reapr_ground_truth(cfg)
    if result[0] is not None:
        (
            attack_chains,
            all_attack_nids,
            all_contaminated_nids,
            all_nids,
            all_paths,
            uuid_to_node_id,
            excluded_nids,
        ) = result
        return (
            all_attack_nids,
            all_contaminated_nids,
            all_nids,
            all_paths,
            uuid_to_node_id,
            excluded_nids,
        )

    # Fallback: return empty sets if no REAPr ground truth configured
    log("No REAPr ground truth configured, using legacy ground truth")
    legacy_nids, legacy_paths, legacy_uuid2nid = get_ground_truth(cfg)
    return legacy_nids, set(), legacy_nids, legacy_paths, legacy_uuid2nid, set()


def get_attack_chains(cfg):
    """
    Get ground truth organized by attack chain.

    Returns:
        attack_chains: dict mapping attack_chain_name -> {
            'attack_nids': set of node IDs labeled as 'attack',
            'contaminated_nids': set of node IDs labeled as 'contaminated',
            'all_nids': set of all node IDs,
            'paths': dict mapping node_id -> process_name
        }
        Returns None if no REAPr ground truth is configured.
    """
    result = parse_reapr_ground_truth(cfg)
    if result[0] is not None:
        return result[0]  # Return attack_chains dict
    return None


def get_GP_of_each_attack(cfg):
    """
    Get ground truth node IDs grouped by attack with time ranges.

    If REAPr ground truth is configured AND attack_to_time_window is provided,
    uses REAPr for node IDs and config for time ranges.

    Returns:
        attack_to_nids: dict mapping attack_index -> {
            'nids': set of node IDs,
            'time_range': [start_ns, end_ns],
            'attack_chain': attack chain name (if REAPr)
        }
    """
    # Try REAPr approach first
    attack_chains = get_attack_chains(cfg)

    if attack_chains is not None:
        # Check if we have attack_to_time_window mapping for REAPr attacks
        attack_chain_to_time_window = getattr(cfg.dataset, "attack_chain_to_time_window", None)

        if attack_chain_to_time_window is not None:
            # Use REAPr attack chains
            # Format: Simple list of attack chain names
            attack_to_nids = {}
            for i, chain_name in enumerate(attack_chain_to_time_window):
                if chain_name in attack_chains:
                    attack_to_nids[i] = {
                        "nids": attack_chains[chain_name]["attack_nids"],
                        "contaminated_nids": attack_chains[chain_name]["contaminated_nids"],
                        "all_nids": attack_chains[chain_name]["all_nids"],
                        "attack_chain": chain_name,
                    }
            if attack_to_nids:
                log(
                    f"Using REAPr attack chains: {[v['attack_chain'] for v in attack_to_nids.values()]}"
                )
                return attack_to_nids

        # Fall back: use legacy attack_to_time_window but get nids from REAPr
        if hasattr(cfg.dataset, "attack_to_time_window") and cfg.dataset.attack_to_time_window:
            log("WARNING: Using legacy attack_to_time_window with REAPr ground truth.")
            log("         This combines all attack chains into each time window.")
            log("         Consider using attack_chain_to_time_window for proper separation.")

            # Combine all REAPr nids as fallback (not ideal but maintains backward compatibility)
            all_attack_nids = set()
            all_contaminated_nids = set()
            for chain_data in attack_chains.values():
                all_attack_nids |= chain_data["attack_nids"]
                all_contaminated_nids |= chain_data["contaminated_nids"]
            all_reapr_nids = all_attack_nids | all_contaminated_nids

            attack_to_nids = {}
            for i, attack_tuple in enumerate(cfg.dataset.attack_to_time_window):
                attack_to_nids[i] = {
                    "nids": all_attack_nids,
                    "contaminated_nids": all_contaminated_nids,
                    "all_nids": all_reapr_nids,
                    "time_range": [
                        _datetime_to_ns_time_for_dataset(cfg, attack_tuple[1]),
                        _datetime_to_ns_time_for_dataset(cfg, attack_tuple[2]),
                    ],
                }
            return attack_to_nids

    # Legacy fallback: use separate CSV files
    uuid2nids, _ = get_uuid2nids_from_csv(cfg)

    attack_to_nids = {}

    for i, (path, attack_to_time_window) in enumerate(
        zip(cfg.dataset.ground_truth_relative_path, cfg.dataset.attack_to_time_window)
    ):
        attack_to_nids[i] = {}
        attack_to_nids[i]["nids"] = set()
        attack_to_nids[i]["time_range"] = [
            _datetime_to_ns_time_for_dataset(cfg, tw)
            for tw in [attack_to_time_window[1], attack_to_time_window[2]]
        ]

        with open(os.path.join(cfg._ground_truth_dir, path), "r") as f:
            reader = csv.reader(f)
            for row in reader:
                node_uuid, node_labels, _ = row[0], row[1], row[2]
                node_id = uuid2nids[node_uuid]
                attack_to_nids[i]["nids"].add(int(node_id))

        mimicry_edge_num = cfg.preprocessing.build_graphs.mimicry_edge_num
        if mimicry_edge_num is not None and mimicry_edge_num > 0:
            num_mimicry_GPs = 0
            with open(
                os.path.join(cfg.preprocessing.build_graphs._mimicry_dir, path.split("/")[-1]), "r"
            ) as f:
                reader = csv.reader(f)
                for row in reader:
                    num_mimicry_GPs += 1
                    node_uuid, node_labels, _ = row[0], row[1], row[2]
                    node_id = uuid2nids[node_uuid]
                    attack_to_nids[i]["nids"].add(int(node_id))
            log(f"{num_mimicry_GPs} mimicry ground truth nodes loaded")
    return attack_to_nids


def get_t2malicious_node(cfg) -> dict[list]:
    """
    Get mapping from timestamp to malicious nodes that had events at that time.
    """
    uuid2nids, nid2uuid = get_uuid2nids_from_csv(cfg)

    t_to_node = defaultdict(list)

    attack_nids, _, _, _, _, _ = get_ground_truth_with_labels(cfg)
    ground_truth_nids = {str(nid) for nid in attack_nids}

    # We still need time windows from config
    if not hasattr(cfg.dataset, "attack_to_time_window") or not cfg.dataset.attack_to_time_window:
        log("No attack_to_time_window configured, cannot get t2malicious_node")
        return t_to_node

    for attack_tuple in cfg.dataset.attack_to_time_window:
        start_time = _datetime_to_ns_time_for_dataset(cfg, attack_tuple[1])
        end_time = _datetime_to_ns_time_for_dataset(cfg, attack_tuple[2])

        # Stream events for this time range (memory efficient)
        for row in stream_events_for_time_range(cfg, start_time, end_time):
            src_id = str(row[1])
            dst_id = str(row[4])
            t = row[6]
            if src_id in ground_truth_nids:
                t_to_node[int(t)].append(nid2uuid[int(src_id)])
            if dst_id in ground_truth_nids:
                t_to_node[int(t)].append(nid2uuid[int(dst_id)])

    return t_to_node


def get_attack_to_mal_edges(cfg) -> dict[list]:
    """
    Get malicious edges grouped by attack, based on time windows and ground truth nodes.
    """
    uuid2nids, nid2uuid = get_uuid2nids_from_csv(cfg)

    malicious_edge_selection = cfg.detection.evaluation.edge_evaluation.malicious_edge_selection

    attack_nids, _, _, _, _, _ = get_ground_truth_with_labels(cfg)
    ground_truth_nids = {str(nid) for nid in attack_nids}

    attack_to_mal_edges = defaultdict(set)

    # We still need time windows from config
    if not hasattr(cfg.dataset, "attack_to_time_window") or not cfg.dataset.attack_to_time_window:
        log("No attack_to_time_window configured, cannot get attack_to_mal_edges")
        return attack_to_mal_edges

    for i, attack_tuple in enumerate(cfg.dataset.attack_to_time_window):
        start_time = _datetime_to_ns_time_for_dataset(cfg, attack_tuple[1])
        end_time = _datetime_to_ns_time_for_dataset(cfg, attack_tuple[2])

        # Stream events for this time range (memory efficient)
        for row in stream_events_for_time_range(cfg, start_time, end_time):
            src_idx_id = str(row[1])
            ope = row[2]
            dst_idx_id = str(row[4])
            event_uuid = row[5]
            timestamp_rec = row[6]

            condition = None
            if malicious_edge_selection == "src_node":
                condition = src_idx_id in ground_truth_nids
            elif malicious_edge_selection == "dst_node":
                condition = dst_idx_id in ground_truth_nids
            elif malicious_edge_selection == "both_nodes":
                condition = src_idx_id in ground_truth_nids and dst_idx_id in ground_truth_nids
            elif malicious_edge_selection == "either_node":
                condition = src_idx_id in ground_truth_nids or dst_idx_id in ground_truth_nids
            else:
                raise ValueError(
                    "`malicious_edge_selection` must be one of 'src_node', 'dst_node', 'both_nodes', 'either_node"
                )

            if condition:
                attack_to_mal_edges[i].add((src_idx_id, dst_idx_id, timestamp_rec, ope))

    return attack_to_mal_edges

    return attack_to_mal_edges


def get_ground_truth_edges(cfg) -> set:
    attack_to_mal_edges = get_attack_to_mal_edges(cfg)

    malicious_edges = set()
    for attack, edges_set in attack_to_mal_edges.items():
        malicious_edges |= edges_set

    return malicious_edges
