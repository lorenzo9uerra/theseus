#!/usr/bin/env python3
"""Convert DARPA TC E3 JSON logs to CSV node and event tables."""

import argparse
import csv
import hashlib
import json
import os

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
    """Parse NetFlowObject records and store to CSV."""
    csv_file = os.path.join(output_path, "netflow_node_table.csv")

    if os.path.exists(csv_file):
        print(f"Loading existing netflow from {csv_file}")
        uuid2hash = {}
        max_index = index_id - 1

        with open(csv_file, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                uuid2hash[row["node_uuid"]] = row["hash_id"]
                max_index = max(max_index, int(row["index_id"]))
        if uuid2hash:
            print(f"Loaded {len(uuid2hash)} netflow nodes")
            return max_index + 1, uuid2hash

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
            values = [
                "" if data[idx] is None else str(data[idx]) for idx in range(1, 5)
            ]
            rows.append([uuid, data[0]] + values + [index_id])
            uuid2hash[uuid] = data[0]
            index_id += 1

    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "node_uuid",
                "hash_id",
                "src_addr",
                "src_port",
                "dst_addr",
                "dst_port",
                "index_id",
            ]
        )
        writer.writerows(rows)

    print(f"Netflow: {count} records -> {len(rows)} nodes saved to {csv_file}")
    return index_id, uuid2hash


def store_process(file_path: str, output_path: str, index_id: int, filelist: list):
    """Parse Subject records and store to CSV."""
    csv_file = os.path.join(output_path, "process_node_table.csv")

    if os.path.exists(csv_file):
        print(f"Loading existing process from {csv_file}")
        uuid2hash = {}
        max_index = index_id - 1

        with open(csv_file, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                uuid2hash[row["node_uuid"]] = row["hash_id"]
                max_index = max(max_index, int(row["index_id"]))

        print(f"Loaded {len(uuid2hash)} process nodes")
        return max_index + 1, uuid2hash

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

                        path = path or ""
                        cmd = cmd or ""

                        # Merge with existing if present
                        if process_uuid in process_obj2hash:
                            existing_path, existing_cmd = process_obj2hash[process_uuid]
                            if not path and existing_path:
                                path = existing_path
                            if not cmd and existing_cmd:
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
        if uuid in process_obj2hash and not process_obj2hash[uuid][0]:
            process_obj2hash[uuid][0] = path

    for uuid, cmd in event_execs.items():
        if uuid in process_obj2hash and not process_obj2hash[uuid][1]:
            process_obj2hash[uuid][1] = cmd

    rows = []
    uuid2hash = {}
    for uuid, data in process_obj2hash.items():
        if len(uuid) != 64:
            rows.append([uuid, _hash_uuid(uuid)] + data + [index_id])
            uuid2hash[uuid] = _hash_uuid(uuid)
            index_id += 1

    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["node_uuid", "hash_id", "path", "cmd", "index_id"])
        writer.writerows(rows)

    print(f"Process: {count} records -> {len(rows)} nodes saved to {csv_file}")
    return index_id, uuid2hash


def store_file(file_path: str, output_path: str, index_id: int, filelist: list):
    """Parse FileObject records and store to CSV."""
    csv_file = os.path.join(output_path, "file_node_table.csv")

    if os.path.exists(csv_file):
        print(f"Loading existing file from {csv_file}")
        uuid2hash = {}
        max_index = index_id - 1

        with open(csv_file, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                uuid2hash[row["node_uuid"]] = row["hash_id"]
                max_index = max(max_index, int(row["index_id"]))

        print(f"Loaded {len(uuid2hash)} file nodes")
        return max_index + 1, uuid2hash

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

                        filename = filename or ""

                        # Merge with existing if present
                        if object_uuid in file_obj2hash and not filename:
                            filename = file_obj2hash[object_uuid]

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
        if uuid in file_obj2hash and not file_obj2hash[uuid]:
            file_obj2hash[uuid] = path

    rows = []
    uuid2hash = {}
    for uuid, path in file_obj2hash.items():
        if len(uuid) != 64:
            rows.append([uuid, _hash_uuid(uuid), path, index_id])
            uuid2hash[uuid] = _hash_uuid(uuid)
            index_id += 1

    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["node_uuid", "hash_id", "path", "index_id"])
        writer.writerows(rows)

    print(f"File: {count} records -> {len(rows)} nodes saved to {csv_file}")
    return index_id, uuid2hash


def store_event(
    file_path: str,
    output_path: str,
    process_uuid2hash: dict,
    file_uuid2hash: dict,
    net_uuid2hash: dict,
    filelist: list,
):
    """Parse Event records and store to CSV."""
    csv_file = os.path.join(output_path, "event_table.csv")

    if os.path.exists(csv_file) and os.path.getsize(csv_file) > 0:
        print(f"Event table already exists: {csv_file}")
        return

    print("Processing events")

    # Build hash -> index mapping from node tables
    nodeid2msg = {}
    for table in ["process_node_table", "file_node_table", "netflow_node_table"]:
        table_path = os.path.join(output_path, f"{table}.csv")
        if os.path.exists(table_path):
            with open(table_path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    nodeid2msg[row["hash_id"]] = row["index_id"]

    total_events = 0
    buffer = []
    flush_every = 50000

    with open(csv_file, "w", newline="") as f_out:
        writer = csv.writer(f_out)
        writer.writerow(
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

        def flush_buffer():
            nonlocal buffer, total_events
            if buffer:
                writer.writerows(buffer)
                total_events += len(buffer)
                buffer.clear()

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
                            row = [
                                object_id,
                                object_index,
                                relation_type,
                                process_id,
                                process_index,
                                event_uuid,
                                timestamp,
                            ]
                        else:
                            row = [
                                process_id,
                                process_index,
                                relation_type,
                                object_id,
                                object_index,
                                event_uuid,
                                timestamp,
                            ]

                        buffer.append(row)
                        if len(buffer) >= flush_every:
                            flush_buffer()
                    except Exception:
                        continue

        flush_buffer()

    print(f"Events: {total_events} saved to {csv_file}")


def main():
    parser = argparse.ArgumentParser(description="Convert DARPA TC E3 JSON to CSV")
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

    print("All data exported to CSV successfully!")


if __name__ == "__main__":
    main()
