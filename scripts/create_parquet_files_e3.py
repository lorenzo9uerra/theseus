#!/usr/bin/env python3
"""Convert DARPA TC E3 JSON logs to Parquet node and event tables."""

import argparse
import hashlib
import json
import os

import polars as pl
from tqdm import tqdm

# Edge types requiring direction reversal (object -> process)
EDGE_REVERSED = {
    "EVENT_EXECUTE",
    "EVENT_LSEEK",
    "EVENT_MMAP",
    "EVENT_OPEN",
    "EVENT_ACCEPT",
    "EVENT_READ",
    "EVENT_RECVFROM",
    "EVENT_RECVMSG",
    "EVENT_READ_SOCKET_PARAMS",
    "EVENT_CHECK_FILE_ATTRIBUTES",
    "READ",
}

# Edge types excluded from graph construction
EXCLUDE_EDGE_TYPE = {
    "EVENT_FCNTL",
    "EVENT_OTHER",
    "EVENT_ADD_OBJECT_ATTRIBUTE",
    "EVENT_FLOWS_TO",
}


def _get_union_value(field):
    """Extract value from Avro union representation."""
    if field is None:
        return None
    if isinstance(field, dict):
        for value in field.values():
            if value is None:
                return None
            if isinstance(value, dict):
                return _get_union_value(value)
            return value
    return field


def _hash_uuid(uuid: str) -> str:
    return hashlib.sha256(uuid.encode("utf-8")).hexdigest()


def store_netflow(file_path: str, output_path: str, index_id: int, filelist: list):
    """Parse NetFlowObject records and store to Parquet."""
    parquet_file = os.path.join(output_path, "netflow_node_table.parquet")
    if os.path.exists(parquet_file):
        print(f"Loading existing netflow from {parquet_file}")
        df = pl.read_parquet(parquet_file)
        uuid2hash = dict(zip(df["node_uuid"], df["hash_id"], strict=False))
        next_id = df.select(pl.col("index_id").max().fill_null(0) + 1).item()
        return next_id, uuid2hash

    print("Processing netflow data")
    netobj2hash = {}
    count = 0

    for file in tqdm(filelist):
        with open(os.path.join(file_path, file)) as f:
            for line in f:
                if '{"datum":{"com.bbn.tc.schema.avro.cdm18.NetFlowObject"' not in line:
                    continue
                try:
                    record = json.loads(line)
                    netflow = record["datum"][
                        "com.bbn.tc.schema.avro.cdm18.NetFlowObject"
                    ]
                    nodeid = netflow["uuid"]
                    srcaddr = _get_union_value(netflow.get("localAddress"))
                    srcport = _get_union_value(netflow.get("localPort"))
                    dstaddr = _get_union_value(netflow.get("remoteAddress"))
                    dstport = _get_union_value(netflow.get("remotePort"))

                    if not all([srcaddr, srcport, dstaddr, dstport]):
                        continue

                    node_values = [srcaddr, srcport, dstaddr, dstport]

                    # Merge with existing if present
                    if nodeid in netobj2hash:
                        existing = netobj2hash[nodeid][1:]
                        for idx, val in enumerate(node_values):
                            if val is None and existing[idx] is not None:
                                node_values[idx] = existing[idx]

                    netobj2hash[nodeid] = [_hash_uuid(nodeid)] + node_values
                    count += 1
                except Exception:
                    pass

    rows = []
    uuid2hash = {}
    for uuid, data in netobj2hash.items():
        if len(uuid) != 64:
            rows.append(
                {
                    "node_uuid": uuid,
                    "hash_id": data[0],
                    "src_addr": str(data[1]) if data[1] is not None else None,
                    "src_port": str(data[2]) if data[2] is not None else None,
                    "dst_addr": str(data[3]) if data[3] is not None else None,
                    "dst_port": str(data[4]) if data[4] is not None else None,
                    "index_id": index_id,
                }
            )
            uuid2hash[uuid] = data[0]
            index_id += 1

    pl.DataFrame(rows).write_parquet(parquet_file, compression="zstd")
    print(f"Netflow: {count} records -> {len(rows)} nodes saved")
    return index_id, uuid2hash


def store_process(file_path: str, output_path: str, index_id: int, filelist: list):
    """Parse Subject records and store to Parquet."""
    parquet_file = os.path.join(output_path, "process_node_table.parquet")
    if os.path.exists(parquet_file):
        print(f"Loading existing process from {parquet_file}")
        df = pl.read_parquet(parquet_file)
        uuid2hash = dict(zip(df["node_uuid"], df["hash_id"], strict=False))
        next_id = df.select(pl.col("index_id").max().fill_null(0) + 1).item()
        return next_id, uuid2hash

    print("Processing process data")
    process_obj2hash = {}
    event_paths = {}
    event_execs = {}
    count = 0

    for file in tqdm(filelist):
        with open(os.path.join(file_path, file)) as f:
            for line in f:
                if '{"datum":{"com.bbn.tc.schema.avro.cdm18.Subject"' in line:
                    try:
                        record = json.loads(line)
                        process = record["datum"][
                            "com.bbn.tc.schema.avro.cdm18.Subject"
                        ]
                        process_uuid = process["uuid"]
                        path = _get_union_value(process.get("path"))
                        cmd = _get_union_value(process.get("cmdLine"))

                        # Check properties.map for dataset-specific fields
                        properties = process.get("properties")
                        if properties and isinstance(properties, dict):
                            props_map = properties.get("map")
                            if props_map and isinstance(props_map, dict):
                                if not path:
                                    path = props_map.get("path")
                                if not cmd:
                                    cmd = props_map.get("name")

                        # Merge with existing if present
                        if process_uuid in process_obj2hash:
                            existing_path, existing_cmd = process_obj2hash[process_uuid]
                            if path is None and existing_path is not None:
                                path = existing_path
                            if cmd is None and existing_cmd is not None:
                                cmd = existing_cmd

                        process_obj2hash[process_uuid] = [path, cmd]
                        count += 1
                    except Exception:
                        pass

                elif '{"datum":{"com.bbn.tc.schema.avro.cdm18.Event"' in line:
                    if '"predicateObjectPath":null' in line and '"exec"' not in line:
                        continue
                    try:
                        record = json.loads(line)
                        event = record["datum"]["com.bbn.tc.schema.avro.cdm18.Event"]

                        # Extract path for predicateObject
                        pred_obj = event.get("predicateObject")
                        path = _get_union_value(event.get("predicateObjectPath"))
                        if pred_obj and path:
                            uuid = pred_obj.get("com.bbn.tc.schema.avro.cdm18.UUID")
                            if uuid:
                                event_paths[uuid] = path

                        # Extract exec for subject
                        process_node = event.get("process")
                        properties = event.get("properties")
                        if (
                            process_node
                            and properties
                            and isinstance(properties, dict)
                            and "map" in properties
                        ):
                            props_map = properties["map"]
                            if props_map and "exec" in props_map:
                                subj_uuid = process_node.get(
                                    "com.bbn.tc.schema.avro.cdm18.UUID"
                                )
                                if subj_uuid:
                                    event_execs[subj_uuid] = props_map["exec"]
                    except Exception:
                        pass

    # Enrich process objects with event metadata
    for uuid, path in event_paths.items():
        if uuid in process_obj2hash and process_obj2hash[uuid][0] is None:
            process_obj2hash[uuid][0] = path

    for uuid, cmd in event_execs.items():
        if uuid in process_obj2hash and process_obj2hash[uuid][1] is None:
            process_obj2hash[uuid][1] = cmd

    rows = []
    uuid2hash = {}
    for uuid, data in process_obj2hash.items():
        if len(uuid) != 64:
            rows.append(
                {
                    "node_uuid": uuid,
                    "hash_id": _hash_uuid(uuid),
                    "path": data[0],
                    "cmd": data[1],
                    "index_id": index_id,
                }
            )
            uuid2hash[uuid] = _hash_uuid(uuid)
            index_id += 1

    pl.DataFrame(rows).write_parquet(parquet_file, compression="zstd")
    print(f"Process: {count} records -> {len(rows)} nodes saved")
    return index_id, uuid2hash


def store_file(file_path: str, output_path: str, index_id: int, filelist: list):
    """Parse FileObject records and store to Parquet."""
    parquet_file = os.path.join(output_path, "file_node_table.parquet")
    if os.path.exists(parquet_file):
        print(f"Loading existing file from {parquet_file}")
        df = pl.read_parquet(parquet_file)
        uuid2hash = dict(zip(df["node_uuid"], df["hash_id"], strict=False))
        next_id = df.select(pl.col("index_id").max().fill_null(0) + 1).item()
        return next_id, uuid2hash

    print("Processing file data")
    file_obj2hash = {}
    event_paths = {}
    count = 0

    for file in tqdm(filelist):
        with open(os.path.join(file_path, file)) as f:
            for line in f:
                if '{"datum":{"com.bbn.tc.schema.avro.cdm18.FileObject"' in line:
                    try:
                        record = json.loads(line)
                        file_object = record["datum"][
                            "com.bbn.tc.schema.avro.cdm18.FileObject"
                        ]
                        object_uuid = file_object["uuid"]
                        filename = _get_union_value(file_object.get("filename"))

                        # Check baseObject.properties.map for filename
                        if not filename:
                            base_object = file_object.get("baseObject")
                            if base_object and isinstance(base_object, dict):
                                properties = base_object.get("properties")
                                if properties and isinstance(properties, dict):
                                    props_map = properties.get("map")
                                    if props_map and isinstance(props_map, dict):
                                        filename = props_map.get("filename")

                        # Merge with existing if present
                        if (
                            object_uuid in file_obj2hash
                            and file_obj2hash[object_uuid] is None
                        ):
                            file_obj2hash[object_uuid] = filename

                        file_obj2hash[object_uuid] = filename
                        count += 1
                    except Exception:
                        pass

                elif '{"datum":{"com.bbn.tc.schema.avro.cdm18.Event"' in line:
                    if '"predicateObjectPath":null' in line:
                        continue
                    try:
                        record = json.loads(line)
                        event = record["datum"]["com.bbn.tc.schema.avro.cdm18.Event"]
                        pred_obj = event.get("predicateObject")
                        if not pred_obj:
                            continue
                        uuid = pred_obj.get("com.bbn.tc.schema.avro.cdm18.UUID")
                        path = _get_union_value(event.get("predicateObjectPath"))
                        if uuid and path:
                            event_paths[uuid] = path
                    except Exception:
                        pass

    # Enrich file objects with event paths
    for uuid, path in event_paths.items():
        if uuid in file_obj2hash and file_obj2hash[uuid] is None:
            file_obj2hash[uuid] = path

    rows = []
    uuid2hash = {}
    for uuid, path in file_obj2hash.items():
        if len(uuid) != 64:
            rows.append(
                {
                    "node_uuid": uuid,
                    "hash_id": _hash_uuid(uuid),
                    "path": path,
                    "index_id": index_id,
                }
            )
            uuid2hash[uuid] = _hash_uuid(uuid)
            index_id += 1

    pl.DataFrame(rows).write_parquet(parquet_file, compression="zstd")
    print(f"File: {count} records -> {len(rows)} nodes saved")
    return index_id, uuid2hash


def store_event(
    file_path: str,
    output_path: str,
    process_uuid2hash: dict,
    file_uuid2hash: dict,
    net_uuid2hash: dict,
    filelist: list,
):
    """Parse Event records and store to Parquet."""
    parquet_file = os.path.join(output_path, "event_table.parquet")
    if os.path.exists(parquet_file):
        print(f"Event table already exists: {parquet_file}")
        return

    print("Processing events")

    # Build hash -> index mapping from node tables
    nodeid2msg = {}
    for table in ["process_node_table", "file_node_table", "netflow_node_table"]:
        p_path = os.path.join(output_path, f"{table}.parquet")
        if os.path.exists(p_path):
            df = pl.read_parquet(p_path, columns=["hash_id", "index_id"])
            nodeid2msg.update(dict(zip(df["hash_id"], df["index_id"], strict=False)))

    events = []

    for file in tqdm(filelist):
        with open(os.path.join(file_path, file)) as f:
            for line in f:
                if '{"datum":{"com.bbn.tc.schema.avro.cdm18.Event"' not in line:
                    continue
                try:
                    record = json.loads(line)
                    event = record["datum"]["com.bbn.tc.schema.avro.cdm18.Event"]
                    relation_type = event.get("type")
                    if relation_type in EXCLUDE_EDGE_TYPE:
                        continue

                    process_field = event.get("process") or {}
                    predicate_field = event.get("predicateObject") or {}
                    process_uuid = process_field.get(
                        "com.bbn.tc.schema.avro.cdm18.UUID"
                    )
                    predicate_uuid = predicate_field.get(
                        "com.bbn.tc.schema.avro.cdm18.UUID"
                    )

                    if not process_uuid or not predicate_uuid:
                        continue
                    if process_uuid not in process_uuid2hash:
                        continue
                    if (
                        predicate_uuid not in process_uuid2hash
                        and predicate_uuid not in file_uuid2hash
                        and predicate_uuid not in net_uuid2hash
                    ):
                        continue

                    event_uuid = event.get("uuid")
                    timestamp = event.get("timestampNanos")
                    if timestamp is None:
                        continue
                    timestamp = int(timestamp)

                    # Resolve node hashes
                    process_id = process_uuid2hash[process_uuid]
                    if predicate_uuid in file_uuid2hash:
                        object_id = file_uuid2hash[predicate_uuid]
                    elif predicate_uuid in net_uuid2hash:
                        object_id = net_uuid2hash[predicate_uuid]
                    else:
                        object_id = process_uuid2hash[predicate_uuid]

                    process_index = nodeid2msg.get(process_id)
                    object_index = nodeid2msg.get(object_id)
                    if process_index is None or object_index is None:
                        continue

                    # Apply edge direction
                    if relation_type in EDGE_REVERSED:
                        src_id, src_idx = object_id, object_index
                        dst_id, dst_idx = process_id, process_index
                    else:
                        src_id, src_idx = process_id, process_index
                        dst_id, dst_idx = object_id, object_index

                    events.append(
                        {
                            "src_node": src_id,
                            "src_index_id": src_idx,
                            "operation": relation_type,
                            "dst_node": dst_id,
                            "dst_index_id": dst_idx,
                            "event_uuid": event_uuid,
                            "timestamp_rec": timestamp,
                        }
                    )
                except Exception:
                    continue

    if events:
        pl.DataFrame(events).write_parquet(parquet_file, compression="zstd")
        print(f"Events: {len(events)} saved")


def main():
    parser = argparse.ArgumentParser(description="Convert DARPA TC E3 JSON to Parquet")
    parser.add_argument(
        "--raw_dir", type=str, required=True, help="Input JSON directory"
    )
    parser.add_argument(
        "--out_dir", type=str, default="./data/DARPA/", help="Output directory"
    )
    args = parser.parse_args()

    file_list = sorted(f for f in os.listdir(args.raw_dir) if "json" in f)
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Input: {args.raw_dir} | Output: {args.out_dir}")

    index_id = 0
    index_id, net_uuid2hash = store_netflow(
        args.raw_dir, args.out_dir, index_id, file_list
    )
    index_id, process_uuid2hash = store_process(
        args.raw_dir, args.out_dir, index_id, file_list
    )
    index_id, file_uuid2hash = store_file(
        args.raw_dir, args.out_dir, index_id, file_list
    )
    store_event(
        args.raw_dir,
        args.out_dir,
        process_uuid2hash,
        file_uuid2hash,
        net_uuid2hash,
        file_list,
    )

    print("All data exported to Parquet successfully!")


if __name__ == "__main__":
    main()
