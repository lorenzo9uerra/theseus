import os
from collections import defaultdict
from datetime import datetime, timedelta

import networkx as nx
import polars as pl
import torch

from pidsmaker import mimicry
from pidsmaker.config.pipeline import get_darpa_tc_node_feats_from_cfg, get_days_from_cfg
from pidsmaker.utils.dataset_utils import get_rel2id
from pidsmaker.utils.utils import (
    datetime_to_ns_time_with_tz,
    get_split_to_files,
    get_table_path,
    log,
    log_start,
    log_tqdm,
    ns_time_to_datetime_US,
    read_table,
    scan_table,
    stringtomd5,
)


def _datetime_to_ns_time_for_dataset(cfg, date_str):
    timezone = getattr(cfg.dataset, "timezone", "US/Eastern")
    return datetime_to_ns_time_with_tz(date_str, timezone)


def load_node_tables(cfg):
    """
    Load node tables from Parquet or CSV files using Polars.
    Node tables are small enough to load into memory.
    Returns DataFrames for netflow, subject, and file tables.
    """
    data_dir = cfg.dataset.csv_dir
    if not data_dir:
        raise ValueError(
            f"csv_dir is not configured for dataset {cfg.dataset.name}. "
            "Please set dataset.csv_dir in the config file."
        )

    log(f"Loading node tables from {data_dir}...")

    netflow_df = read_table(data_dir, "netflow_node_table")
    subject_df = read_table(data_dir, "process_node_table")
    file_df = read_table(data_dir, "file_node_table")

    if netflow_df is None:
        netflow_df = pl.DataFrame()
    if subject_df is None:
        subject_df = pl.DataFrame()
    if file_df is None:
        file_df = pl.DataFrame()

    log(
        f"Loaded {len(netflow_df)} netflow nodes, {len(subject_df)} subject nodes, "
        f"{len(file_df)} file nodes"
    )

    return netflow_df, subject_df, file_df


def get_event_table_path(cfg):
    """Get the path to the event table file (Parquet preferred, CSV fallback)."""
    data_dir = cfg.dataset.csv_dir
    if not data_dir:
        raise ValueError(
            f"csv_dir is not configured for dataset {cfg.dataset.name}. "
            "Please set dataset.csv_dir in the config file."
        )
    path, fmt = get_table_path(data_dir, "event_table")
    return path


def stream_events_for_time_range(cfg, start_ns_timestamp, end_ns_timestamp, chunk_minutes=60):
    """
    Stream events for a given time range using chunked processing.
    Supports both Parquet and CSV formats.
    Instead of loading and sorting all events for a day, we process hour by hour.

    Yields tuples of (src_node, src_index_id, operation, dst_node, dst_index_id,
                      event_uuid, timestamp_rec, _id)
    """
    data_dir = cfg.dataset.csv_dir
    lazy_df = scan_table(data_dir, "event_table")

    if lazy_df is None:
        log(f"Warning: No event table found in {data_dir}")
        return

    # Get column names
    schema = lazy_df.collect_schema()
    cols = list(schema.names())
    timestamp_col = cols[6]  # timestamp_rec column

    # Process in chunks (e.g., 1 hour at a time) to avoid sorting massive amounts
    chunk_ns = chunk_minutes * 60 * 1_000_000_000  # Convert minutes to nanoseconds
    current_start = start_ns_timestamp

    while current_start < end_ns_timestamp:
        current_end = min(current_start + chunk_ns, end_ns_timestamp)

        # Filter for this chunk and sort
        filtered = (
            lazy_df.filter(
                (pl.col(timestamp_col) >= current_start) & (pl.col(timestamp_col) < current_end)
            ).sort([timestamp_col, cols[5]])  # Sort by timestamp, event_uuid
        )

        # Collect this chunk (much smaller than full day)
        try:
            chunk_df = filtered.collect(streaming=True)
        except Exception as e:
            log(f"Warning: Error processing chunk {current_start}-{current_end}: {e}")
            current_start = current_end
            continue

        # Yield rows from this chunk
        for row in chunk_df.iter_rows():
            yield row

        # Free memory
        del chunk_df

        current_start = current_end


def compute_indexid2msg(cfg):
    """
    Returns a dict or LazyNodeLoader that maps {
        node => [node type, feature msg],
    }
    Uses Parquet/CSV files loaded via Polars with memory-efficient processing.

    For very large datasets (like TRACE with 40M+ nodes), we use a lazy loading
    approach that only loads node metadata when needed during graph construction.
    """
    data_dir = cfg.dataset.csv_dir
    if not data_dir:
        raise ValueError(
            f"csv_dir is not configured for dataset {cfg.dataset.name}. "
            "Please set dataset.csv_dir in the config file."
        )

    use_hashed_label = cfg.preprocessing.build_graphs.use_hashed_label
    node_label_features = get_darpa_tc_node_feats_from_cfg(cfg)

    def get_label_str(attrs_dict, node_type):
        """Build label string from node attributes."""
        parts = []
        for label_used in node_label_features[node_type]:
            val = attrs_dict.get(label_used, "")
            parts.append(str(val) if val is not None else "")
        label_str = " ".join(parts)
        if use_hashed_label:
            label_str = stringtomd5(label_str)
        return label_str

    log(f"Loading node tables from {data_dir}...")

    # Get counts using lazy evaluation (very fast, doesn't load data)
    netflow_lf = scan_table(data_dir, "netflow_node_table")
    file_lf = scan_table(data_dir, "file_node_table")
    subject_lf = scan_table(data_dir, "process_node_table")

    netflow_count = netflow_lf.select(pl.len()).collect().item() if netflow_lf is not None else 0
    file_count = file_lf.select(pl.len()).collect().item() if file_lf is not None else 0
    subject_count = subject_lf.select(pl.len()).collect().item() if subject_lf is not None else 0
    total_nodes = netflow_count + file_count + subject_count

    log(
        f"Dataset has {netflow_count:,} netflow, {file_count:,} file, {subject_count:,} subject nodes"
    )
    log(f"Total: {total_nodes:,} nodes")

    # For very large datasets, use lazy loading approach
    LARGE_DATASET_THRESHOLD = 10_000_000  # 10M nodes

    if total_nodes > LARGE_DATASET_THRESHOLD:
        log(f"Large dataset detected ({total_nodes:,} nodes). Using lazy loading strategy.")
        return LazyNodeLoader(cfg, node_label_features, use_hashed_label)

    # For smaller datasets, load everything into memory
    log("Loading all node metadata into memory...")
    indexid2msg = {}

    # Process NETFLOW nodes
    if netflow_count > 0:
        log(f"Processing {netflow_count:,} netflow nodes...")
        netflow_df = read_table(data_dir, "netflow_node_table")
        netflow_cols = netflow_df.columns

        for row in netflow_df.iter_rows(named=True):
            index_id = str(row.get(netflow_cols[-1], ""))
            attrs = {
                "local_ip": str(row.get(netflow_cols[2], "") or ""),
                "local_port": str(row.get(netflow_cols[3], "") or ""),
                "remote_ip": str(row.get(netflow_cols[4], "") or ""),
                "remote_port": str(row.get(netflow_cols[5], "") or ""),
            }
            label_str = get_label_str(attrs, "netflow")
            indexid2msg[index_id] = ["netflow", label_str]
        del netflow_df

    # Process FILE nodes
    if file_count > 0:
        log(f"Processing {file_count:,} file nodes...")
        file_df = read_table(data_dir, "file_node_table")
        file_cols = file_df.columns

        for row in file_df.iter_rows(named=True):
            index_id = str(row.get(file_cols[-1], ""))
            attrs = {"path": str(row.get(file_cols[2], "") or "")}
            label_str = get_label_str(attrs, "file")
            indexid2msg[index_id] = ["file", label_str]
        del file_df

    # Process SUBJECT nodes
    if subject_count > 0:
        log(f"Processing {subject_count:,} subject nodes...")
        subject_df = read_table(data_dir, "process_node_table")
        subject_cols = subject_df.columns

        for row in subject_df.iter_rows(named=True):
            index_id = str(row.get(subject_cols[-1], ""))
            attrs = {
                "path": str(row.get(subject_cols[2], "") or ""),
                "cmd_line": str(row.get(subject_cols[3], "") or ""),
            }
            label_str = get_label_str(attrs, "subject")
            indexid2msg[index_id] = ["subject", label_str]
        del subject_df

    log(f"Loaded {len(indexid2msg):,} nodes into memory")
    return indexid2msg


class LazyNodeLoader:
    """Lazy loader for node metadata that loads on-demand from Parquet/CSV."""

    def __init__(self, cfg, node_label_features, use_hashed_label):
        self.cfg = cfg
        self.data_dir = cfg.dataset.csv_dir
        self.node_label_features = node_label_features
        self.use_hashed_label = use_hashed_label

        # Get column names from available tables
        netflow_lf = scan_table(self.data_dir, "netflow_node_table")
        file_lf = scan_table(self.data_dir, "file_node_table")
        subject_lf = scan_table(self.data_dir, "process_node_table")

        self.netflow_cols = netflow_lf.collect_schema().names() if netflow_lf is not None else []
        self.file_cols = file_lf.collect_schema().names() if file_lf is not None else []
        self.subject_cols = subject_lf.collect_schema().names() if subject_lf is not None else []

        # Cache for loaded nodes
        self._cache = {}
        self._cache_max_size = 2_000_000  # Cache up to 2M nodes

        log(f"LazyNodeLoader initialized (no upfront indexing)")

    def _get_label_str(self, attrs_dict, node_type):
        """Build label string from node attributes."""
        parts = []
        for label_used in self.node_label_features[node_type]:
            val = attrs_dict.get(label_used, "")
            parts.append(str(val) if val is not None else "")
        label_str = " ".join(parts)
        if self.use_hashed_label:
            label_str = stringtomd5(label_str)
        return label_str

    def __contains__(self, key):
        """Check if a node exists - always return True to avoid expensive lookups."""
        return True

    def __getitem__(self, key):
        """Get node info [node_type, label_str] for a given index_id."""
        key = str(key)

        # Check cache first
        if key in self._cache:
            return self._cache[key]

        # Load from CSV - try each file
        result = self._load_single_node(key)

        if result is not None:
            # Cache the result
            if len(self._cache) < self._cache_max_size:
                self._cache[key] = result
            return result

        raise KeyError(f"Node {key} not found in any table")

    def _load_single_node(self, index_id):
        """Load a single node from Parquet/CSV files."""
        int_id = int(index_id)

        # Try subject first (most common)
        if self.subject_cols:
            index_col = self.subject_cols[-1]
            subject_lf = scan_table(self.data_dir, "process_node_table")
            if subject_lf is not None:
                row = subject_lf.filter(pl.col(index_col) == int_id).collect()

                if len(row) > 0:
                    row_dict = row.to_dicts()[0]
                    attrs = {
                        "path": str(row_dict.get(self.subject_cols[2], "") or ""),
                        "cmd_line": str(row_dict.get(self.subject_cols[3], "") or ""),
                    }
                    return ["subject", self._get_label_str(attrs, "subject")]

        # Try netflow
        if self.netflow_cols:
            index_col = self.netflow_cols[-1]
            netflow_lf = scan_table(self.data_dir, "netflow_node_table")
            if netflow_lf is not None:
                row = netflow_lf.filter(pl.col(index_col) == int_id).collect()

                if len(row) > 0:
                    row_dict = row.to_dicts()[0]
                    attrs = {
                        "local_ip": str(row_dict.get(self.netflow_cols[2], "") or ""),
                        "local_port": str(row_dict.get(self.netflow_cols[3], "") or ""),
                        "remote_ip": str(row_dict.get(self.netflow_cols[4], "") or ""),
                        "remote_port": str(row_dict.get(self.netflow_cols[5], "") or ""),
                    }
                    return ["netflow", self._get_label_str(attrs, "netflow")]

        # Try file
        if self.file_cols:
            index_col = self.file_cols[-1]
            file_lf = scan_table(self.data_dir, "file_node_table")
            if file_lf is not None:
                row = file_lf.filter(pl.col(index_col) == int_id).collect()

                if len(row) > 0:
                    row_dict = row.to_dicts()[0]
                    attrs = {"path": str(row_dict.get(self.file_cols[2], "") or "")}
                    return ["file", self._get_label_str(attrs, "file")]

        return None

    def batch_load(self, index_ids):
        """
        Efficiently load multiple nodes at once.
        This is the preferred method during graph construction.

        Returns a dict mapping index_id -> [node_type, label_str]
        """
        index_ids = list(set(str(i) for i in index_ids))

        # Check cache first, collect uncached
        results = {}
        uncached_ids = []

        for idx in index_ids:
            if idx in self._cache:
                results[idx] = self._cache[idx]
            else:
                uncached_ids.append(idx)

        if not uncached_ids:
            return results

        # Convert to integers for querying
        int_ids = [int(i) for i in uncached_ids]
        int_id_set = set(int_ids)

        # Batch load from each table using scan_table helper
        # Subject nodes (usually largest)
        if self.subject_cols and int_id_set:
            index_col = self.subject_cols[-1]
            subject_lf = scan_table(self.data_dir, "process_node_table")
            if subject_lf is not None:
                subject_results = subject_lf.filter(
                    pl.col(index_col).is_in(list(int_id_set))
                ).collect()

                for row_dict in subject_results.to_dicts():
                    idx = str(row_dict[index_col])
                    attrs = {
                        "path": str(row_dict.get(self.subject_cols[2], "") or ""),
                        "cmd_line": str(row_dict.get(self.subject_cols[3], "") or ""),
                    }
                    results[idx] = ["subject", self._get_label_str(attrs, "subject")]
                    int_id_set.discard(int(idx))

        # Netflow nodes
        if self.netflow_cols and int_id_set:
            index_col = self.netflow_cols[-1]
            netflow_lf = scan_table(self.data_dir, "netflow_node_table")
            if netflow_lf is not None:
                netflow_results = netflow_lf.filter(
                    pl.col(index_col).is_in(list(int_id_set))
                ).collect()

                for row_dict in netflow_results.to_dicts():
                    idx = str(row_dict[index_col])
                    attrs = {
                        "local_ip": str(row_dict.get(self.netflow_cols[2], "") or ""),
                        "local_port": str(row_dict.get(self.netflow_cols[3], "") or ""),
                        "remote_ip": str(row_dict.get(self.netflow_cols[4], "") or ""),
                        "remote_port": str(row_dict.get(self.netflow_cols[5], "") or ""),
                    }
                    results[idx] = ["netflow", self._get_label_str(attrs, "netflow")]
                    int_id_set.discard(int(idx))

        # File nodes
        if self.file_cols and int_id_set:
            index_col = self.file_cols[-1]
            file_lf = scan_table(self.data_dir, "file_node_table")
            if file_lf is not None:
                file_results = file_lf.filter(pl.col(index_col).is_in(list(int_id_set))).collect()

                for row_dict in file_results.to_dicts():
                    idx = str(row_dict[index_col])
                    attrs = {"path": str(row_dict.get(self.file_cols[2], "") or "")}
                    results[idx] = ["file", self._get_label_str(attrs, "file")]

        # Update cache with new results
        for idx in uncached_ids:
            if idx in results and len(self._cache) < self._cache_max_size:
                self._cache[idx] = results[idx]

        return results

    def get(self, key, default=None):
        """Dict-like get method."""
        try:
            return self[key]
        except KeyError:
            return default


def save_indexid2msg(indexid2msg, split2nodes, cfg):
    """
    The saving must occur after the graph construction, because some edge types
    are not considered and this results in some nodes that are not used in the pipeline.
    These nodes must be removed before storing to disk to avoid future errors.

    For LazyNodeLoader, we batch load only the nodes that are actually used.
    """
    all_nodes = set().union(*split2nodes.values()) if split2nodes else set()

    out_dir = cfg.preprocessing.build_graphs._dicts_dir
    os.makedirs(out_dir, exist_ok=True)

    if isinstance(indexid2msg, LazyNodeLoader):
        # For lazy loader, batch load only the used nodes
        log(f"Batch loading {len(all_nodes):,} used nodes for saving...")

        # Process in chunks to avoid memory issues
        all_nodes_list = list(all_nodes)
        chunk_size = 500_000
        final_dict = {}

        for i in range(0, len(all_nodes_list), chunk_size):
            chunk = all_nodes_list[i : i + chunk_size]
            loaded = indexid2msg.batch_load(chunk)
            final_dict.update(loaded)
            log(
                f"  Loaded {min(i + chunk_size, len(all_nodes_list)):,}/{len(all_nodes_list):,} nodes"
            )

        log("Saving indexid2msg to disk...")
        torch.save(final_dict, os.path.join(out_dir, "indexid2msg.pkl"))
    else:
        # Standard dict - filter to used nodes
        filtered = {k: v for k, v in indexid2msg.items() if k in all_nodes}
        log("Saving indexid2msg to disk...")
        torch.save(filtered, os.path.join(out_dir, "indexid2msg.pkl"))


def compute_and_save_split2nodes(cfg):
    """
    Returns a dict that maps {
        "train" => nodes in train,
        "test" => nodes in test,
        "val" => nodes in val,
    }
    """
    split_to_files = get_split_to_files(cfg, cfg.preprocessing.build_graphs._graphs_dir)
    split2nodes = defaultdict(set)

    for split, files in split_to_files.items():
        graph_list = [torch.load(path) for path in files]
        for G in log_tqdm(graph_list, desc=f"Check nodes in {split} set"):
            for node in G.nodes():
                split2nodes[split].add(node)
    split2nodes = dict(split2nodes)

    out_dir = cfg.preprocessing.build_graphs._dicts_dir
    os.makedirs(out_dir, exist_ok=True)
    log("Saving split2nodes to disk...")
    torch.save(split2nodes, os.path.join(out_dir, "split2nodes.pkl"))

    return split2nodes


def generate_timestamps(start_time, end_time, interval_minutes):
    start = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
    end = datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S")

    timestamps = []
    current_time = start
    while current_time <= end:
        timestamps.append(current_time.strftime("%Y-%m-%d %H:%M:%S"))
        current_time += timedelta(minutes=interval_minutes)
    timestamps.append(end)
    return timestamps


def gen_edge_fused_tw(indexid2msg, cfg):
    """
    Generate time-window graphs from events using true streaming.
    Events are processed in batches without loading all events for a day into memory.
    """
    rel2id = get_rel2id(cfg)
    include_edge_type = set(rel2id.keys())

    mimicry_edge_num = cfg.preprocessing.build_graphs.mimicry_edge_num
    if mimicry_edge_num is not None and mimicry_edge_num > 0:
        attack_mimicry_events = mimicry.gen_mimicry_edges(cfg)
    else:
        attack_mimicry_events = defaultdict(list)

    # In test mode, we ensure to get 1 TW in each set
    days = get_days_from_cfg(cfg)
    window_size_in_ns = cfg.preprocessing.build_graphs.time_window_size * 60_000_000_000
    BATCH = 1024

    log("Building graphs...")
    for day in days:
        date_start = cfg.dataset.year_month + "-" + str(day) + " 00:00:00"
        date_stop = cfg.dataset.year_month + "-" + str(day + 1) + " 00:00:00"

        start_ns_timestamp = _datetime_to_ns_time_for_dataset(cfg, date_start)
        end_ns_timestamp = _datetime_to_ns_time_for_dataset(cfg, date_stop)

        # Check for mimicry events in this time range
        # Unused for Theseus, but kept for consistency
        attack_index = 0
        mimicry_events = []
        for attack_tuple in cfg.dataset.attack_to_time_window:
            attack = attack_tuple[0]
            attack_start_time = _datetime_to_ns_time_for_dataset(cfg, attack_tuple[1])
            attack_end_time = _datetime_to_ns_time_for_dataset(cfg, attack_tuple[2])

            if mimicry_edge_num > 0 and (
                attack_start_time >= start_ns_timestamp and attack_end_time <= end_ns_timestamp
            ):
                log(
                    f"Insert mimicry events into attack {attack_index} when building graphs from {date_start} to {date_stop}"
                )
                mimicry_events.extend(attack_mimicry_events[attack_index])
            attack_index += 1

        # Process events in TRUE streaming fashion - never load all events at once
        temp_list = []
        window_start_time = None
        events_in_window = 0
        total_events_processed = 0

        log(f"Processing day {day}...")

        for row in stream_events_for_time_range(cfg, start_ns_timestamp, end_ns_timestamp):
            src_node = row[0]
            src_index_id = str(row[1])
            operation = row[2]
            dst_node = row[3]
            dst_index_id = str(row[4])
            event_uuid = row[5]
            timestamp_rec = row[6]
            _id = row[7] if len(row) > 7 else None

            if operation not in include_edge_type:
                continue

            total_events_processed += 1

            # Initialize window start time with first event
            if window_start_time is None:
                window_start_time = timestamp_rec

            event_tuple = (
                src_node,
                src_index_id,
                operation,
                dst_node,
                dst_index_id,
                event_uuid,
                timestamp_rec,
                _id,
            )
            temp_list.append(event_tuple)
            events_in_window += 1

            # Check if we should save a time window
            should_save = (
                timestamp_rec > window_start_time + window_size_in_ns and events_in_window >= BATCH
            )

            if should_save:
                # Save current window
                _save_time_window_graph(
                    temp_list, indexid2msg, cfg, day, window_start_time, timestamp_rec
                )

                # Reset for next window
                window_start_time = timestamp_rec
                temp_list = []
                events_in_window = 0

            # Progress logging every 1M events
            if total_events_processed % 1_000_000 == 0:
                log(f"  Processed {total_events_processed:,} events...")

        # Add mimicry events to the last batch
        for mim_event in mimicry_events:
            (
                src_node,
                src_index_id,
                operation,
                dst_node,
                dst_index_id,
                event_uuid,
                timestamp_rec,
                _id,
            ) = mim_event
            if operation in include_edge_type:
                event_tuple = (
                    src_node,
                    str(src_index_id),
                    operation,
                    dst_node,
                    str(dst_index_id),
                    event_uuid,
                    timestamp_rec,
                    _id,
                )
                temp_list.append(event_tuple)

        # Save remaining events as final window
        if temp_list:
            final_time = temp_list[-1][6] if temp_list else end_ns_timestamp
            _save_time_window_graph(
                temp_list,
                indexid2msg,
                cfg,
                day,
                window_start_time or start_ns_timestamp,
                final_time,
            )

        log(f"  Day {day}: processed {total_events_processed:,} events total")

        # In test mode we already reduced the split lists to the first available day per split
        # in `get_days_from_cfg`, so we should keep iterating here to materialize all requested
        # smoke-test splits rather than stopping after the very first day overall.


def _save_time_window_graph(events_list, indexid2msg, cfg, day, start_time, end_time):
    """Helper function to save a time window graph."""
    if not events_list:
        return

    time_interval = ns_time_to_datetime_US(start_time) + "~" + ns_time_to_datetime_US(end_time)

    # Collect all unique node IDs first for batch loading
    all_node_ids = set()
    for event in events_list:
        src_index_id = event[1]
        dst_index_id = event[4]
        all_node_ids.add(src_index_id)
        all_node_ids.add(dst_index_id)

    # Batch load node metadata (efficient for LazyNodeLoader)
    if hasattr(indexid2msg, "batch_load"):
        node_metadata = indexid2msg.batch_load(all_node_ids)
    else:
        # Regular dict - just reference it directly
        node_metadata = {nid: indexid2msg[nid] for nid in all_node_ids if nid in indexid2msg}

    node_info = {}
    edge_list = []

    if cfg.preprocessing.build_graphs.fuse_edge:
        edge_info = {}
        for event in events_list:
            src_index_id = event[1]
            dst_index_id = event[4]
            operation = event[2]
            event_uuid = event[5]
            timestamp_rec = event[6]

            if src_index_id not in node_info and src_index_id in node_metadata:
                node_type, label = node_metadata[src_index_id]
                node_info[src_index_id] = {
                    "label": label,
                    "node_type": node_type,
                }
            if dst_index_id not in node_info and dst_index_id in node_metadata:
                node_type, label = node_metadata[dst_index_id]
                node_info[dst_index_id] = {
                    "label": label,
                    "node_type": node_type,
                }

            if (src_index_id, dst_index_id) not in edge_info:
                edge_info[(src_index_id, dst_index_id)] = []

            edge_info[(src_index_id, dst_index_id)].append((timestamp_rec, operation, event_uuid))

        for (src, dst), data in edge_info.items():
            sorted_data = sorted(data, key=lambda x: x[0])
            operation_list = [entry[1] for entry in sorted_data]

            indices = []
            current_type = None
            current_start_index = None

            for idx, item in enumerate(operation_list):
                if item == current_type:
                    continue
                else:
                    if current_type is not None and current_start_index is not None:
                        indices.append(current_start_index)
                    current_type = item
                    current_start_index = idx

            if current_type is not None and current_start_index is not None:
                indices.append(current_start_index)

            for k in indices:
                edge_list.append(
                    {
                        "src": src,
                        "dst": dst,
                        "time": sorted_data[k][0],
                        "label": sorted_data[k][1],
                        "event_uuid": sorted_data[k][2],
                    }
                )
    else:
        for event in events_list:
            src_index_id = event[1]
            dst_index_id = event[4]
            operation = event[2]
            event_uuid = event[5]
            timestamp_rec = event[6]

            if src_index_id not in node_info and src_index_id in node_metadata:
                node_type, label = node_metadata[src_index_id]
                node_info[src_index_id] = {
                    "label": label,
                    "node_type": node_type,
                }
            if dst_index_id not in node_info and dst_index_id in node_metadata:
                node_type, label = node_metadata[dst_index_id]
                node_info[dst_index_id] = {
                    "label": label,
                    "node_type": node_type,
                }

            edge_list.append(
                {
                    "src": src_index_id,
                    "dst": dst_index_id,
                    "time": timestamp_rec,
                    "label": operation,
                    "event_uuid": event_uuid,
                }
            )

    graph = nx.MultiDiGraph()

    for node, info in node_info.items():
        graph.add_node(node, node_type=info["node_type"], label=info["label"])

    for i, edge in enumerate(edge_list):
        graph.add_edge(
            edge["src"],
            edge["dst"],
            event_uuid=edge["event_uuid"],
            time=edge["time"],
            label=edge["label"],
            y=0,
        )

        # For unit tests, we only want few edges
        NUM_TEST_EDGES = 2000
        if cfg._test_mode and i >= NUM_TEST_EDGES:
            break

    date_dir = f"{cfg.preprocessing.build_graphs._graphs_dir}/graph_{day}/"
    os.makedirs(date_dir, exist_ok=True)
    graph_name = f"{date_dir}/{time_interval}"

    torch.save(graph, graph_name)


def main(cfg):
    log_start(__file__)

    # Compute indexid2msg using the node tables (loaded into memory - they're small)
    indexid2msg = compute_indexid2msg(cfg)

    # Build graphs from events (streamed from CSV - memory efficient)
    gen_edge_fused_tw(indexid2msg=indexid2msg, cfg=cfg)

    split2nodes = compute_and_save_split2nodes(cfg)
    save_indexid2msg(indexid2msg, split2nodes, cfg)
