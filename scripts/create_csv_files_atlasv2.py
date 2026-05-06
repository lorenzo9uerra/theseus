#!/usr/bin/env python3
"""Convert ATLASv2 Carbon Black EDR JSONL logs into Theseus node/event tables."""

import argparse
import calendar
import csv
import hashlib
import json
from pathlib import Path

import polars as pl
from tqdm import tqdm

EVENT_COLUMNS = [
    "src_node",
    "src_index_id",
    "operation",
    "dst_node",
    "dst_index_id",
    "event_uuid",
    "timestamp_rec",
]

PROCESS_COLUMNS = ["node_uuid", "hash_id", "path", "cmd", "pid", "attack", "index_id"]
FILE_COLUMNS = ["node_uuid", "hash_id", "path", "index_id"]
NETFLOW_COLUMNS = [
    "node_uuid",
    "hash_id",
    "src_addr",
    "src_port",
    "dst_addr",
    "dst_port",
    "index_id",
]

FILE_WRITE_ACTIONS = {
    "ACTION_FILE_OPEN_WRITE",
    "ACTION_FILE_CREATE",
    "ACTION_FILE_OPEN_DELETE",
    "ACTION_FILE_DELETE",
    "ACTION_FILE_RENAME",
    "ACTION_FILE_TRUNCATE",
    "ACTION_FILE_WRITE",
    "ACTION_FILE_LAST_WRITE",
    "ACTION_FILE_OPEN_SET_ATTRIBUTES",
    "ACTION_FILE_OPEN_SET_SECURITY",
}

FILE_READ_ACTIONS = {"ACTION_FILE_OPEN_READ"}

REG_WRITE_ACTIONS = {
    "ACTION_WRITE_VALUE",
    "ACTION_CREATE_KEY",
    "ACTION_OPEN_KEY_WRITE",
    "ACTION_DELETE_KEY",
    "ACTION_DELETE_VALUE",
    "ACTION_OPEN_KEY_DELETE",
}

REG_READ_ACTIONS = {"ACTION_OPEN_KEY_READ"}


def _require_polars():
    if pl is None:  # pragma: no cover
        raise ModuleNotFoundError(
            "polars is required for Parquet output. Install polars or use --output_format csv."
        )


def _hash_uuid(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _safe_int(value) -> int | None:
    if value is None:
        return None
    try:
        if isinstance(value, float):
            return int(value)
        return int(value)
    except Exception:
        return None


def _parse_cbc_timestamp_to_ns(ts: str | None) -> int | None:
    """Parse CBC timestamps like '2022-07-19 16:40:46.7444377 +0000 UTC' to epoch ns."""
    if not ts:
        return None

    parts = ts.split()
    if len(parts) < 3:
        return None

    date_str, time_str, offset_str = parts[0], parts[1], parts[2]

    try:
        year_s, month_s, day_s = date_str.split("-", 2)
        year, month, day = int(year_s), int(month_s), int(day_s)

        if "." in time_str:
            hms_str, frac_str = time_str.split(".", 1)
        else:
            hms_str, frac_str = time_str, ""

        hour_s, minute_s, second_s = hms_str.split(":", 2)
        hour, minute, second = int(hour_s), int(minute_s), int(second_s)

        frac_digits = "".join(ch for ch in frac_str if ch.isdigit())
        if frac_digits:
            if len(frac_digits) > 9:
                frac_digits = frac_digits[:9]
            ns_fraction = int(frac_digits) * (10 ** (9 - len(frac_digits)))
        else:
            ns_fraction = 0

        # Offset like +0000 / -0500
        offset_seconds = 0
        if (
            offset_str
            and len(offset_str) == 5
            and offset_str[0] in "+-"
            and offset_str[1:].isdigit()
        ):
            sign = 1 if offset_str[0] == "+" else -1
            off_h = int(offset_str[1:3])
            off_m = int(offset_str[3:5])
            offset_seconds = sign * (off_h * 3600 + off_m * 60)

        epoch_seconds = calendar.timegm((year, month, day, hour, minute, second))
        epoch_seconds -= offset_seconds
        return epoch_seconds * 1_000_000_000 + ns_fraction
    except Exception:
        return None


def _split_action_tokens(action: str | None) -> set[str]:
    if not action:
        return set()
    return {t.strip() for t in action.split("|") if t.strip()}


def _get_or_create_process(
    nodes: dict,
    next_index: int,
    device_name: str,
    guid: str,
    path: str | None,
    cmd: str | None,
    *,
    pid: int | None,
    attack_id: str | None,
):
    node_uuid = f"{device_name}|{guid}"
    existing = nodes.get(node_uuid)
    if existing is None:
        nodes[node_uuid] = {
            "node_uuid": node_uuid,
            "hash_id": _hash_uuid(node_uuid),
            "path": path or "",
            "cmd": cmd or "",
            "pid": pid,
            "attack": attack_id or "",
            "index_id": next_index,
        }
        return nodes[node_uuid], next_index + 1

    if path and not existing["path"]:
        existing["path"] = path
    if cmd and not existing["cmd"]:
        existing["cmd"] = cmd
    if pid is not None and existing.get("pid") is None:
        existing["pid"] = pid
    if attack_id and not existing.get("attack"):
        existing["attack"] = attack_id
    return existing, next_index


def _get_or_create_file(nodes: dict, next_index: int, device_name: str, path: str):
    node_uuid = f"{device_name}|{path}"
    existing = nodes.get(node_uuid)
    if existing is None:
        nodes[node_uuid] = {
            "node_uuid": node_uuid,
            "hash_id": _hash_uuid(node_uuid),
            "path": path,
            "index_id": next_index,
        }
        return nodes[node_uuid], next_index + 1
    return existing, next_index


def _get_or_create_registry(nodes: dict, next_index: int, device_name: str, key: str):
    # Store registry keys in the file node table, but namespace them to avoid collisions with paths.
    path = f"REG|{key}"
    return _get_or_create_file(nodes, next_index, device_name, path)


def _get_or_create_netflow(
    nodes: dict,
    next_index: int,
    device_name: str,
    community_id: str | None,
    local_ip: str | None,
    local_port,
    remote_ip: str | None,
    remote_port,
):
    src_addr = (local_ip or "").strip()
    dst_addr = (remote_ip or "").strip()
    src_port_i = _safe_int(local_port)
    dst_port_i = _safe_int(remote_port)

    if not src_addr or not dst_addr or src_port_i is None or dst_port_i is None:
        return None, next_index

    # Prefer CBC's flow hash if present; otherwise build a stable UUID from the 4-tuple.
    if community_id:
        node_uuid = f"{device_name}|{community_id}"
    else:
        node_uuid = f"{device_name}|{src_addr}:{src_port_i}->{dst_addr}:{dst_port_i}"

    existing = nodes.get(node_uuid)
    if existing is None:
        nodes[node_uuid] = {
            "node_uuid": node_uuid,
            "hash_id": _hash_uuid(node_uuid),
            "src_addr": src_addr,
            "src_port": str(src_port_i),
            "dst_addr": dst_addr,
            "dst_port": str(dst_port_i),
            "index_id": next_index,
        }
        return nodes[node_uuid], next_index + 1
    return existing, next_index


def _emit_event(buffer, flush_every: int, writer, row):
    buffer.append(row)
    if len(buffer) >= flush_every:
        writer.writerows(buffer)
        buffer.clear()


def _infer_host_from_path(path: Path) -> str | None:
    parts = {p.lower() for p in path.parts}
    if "h1" in parts:
        return "h1"
    if "h2" in parts:
        return "h2"

    name = path.name.lower()
    if "-h1-" in name or name.startswith("h1-") or name.startswith("edr-h1-"):
        return "h1"
    if "-h2-" in name or name.startswith("h2-") or name.startswith("edr-h2-"):
        return "h2"
    return None


def _infer_attack_id_from_edr_file(path: Path) -> str | None:
    """Infer an attack/scenario ID like 'atlasv2/h1-m1' from an EDR jsonl filename."""
    stem = path.stem.lower()  # e.g., 'edr-h1-m1'
    parts = stem.split("-")
    if len(parts) < 3:
        return None
    if parts[0] != "edr":
        return None
    host = parts[1]
    if host not in {"h1", "h2"}:
        return None
    scenario = "-".join(parts[2:])  # 'benign', 'm1', ...
    if not scenario:
        return None
    return f"atlasv2/{host}-{scenario}"


def _filter_edr_files_by_host(edr_files: list[Path], host: str) -> list[Path]:
    host = host.lower().strip()
    if host == "all":
        return edr_files
    if host not in {"h1", "h2"}:
        raise ValueError(f"Unsupported host filter: {host}")

    filtered = []
    for p in edr_files:
        inferred = _infer_host_from_path(p)
        if inferred == host:
            filtered.append(p)
    return filtered


def _convert_edr_files_to_csv_tables(
    edr_files: list[Path],
    out_dir: Path,
    *,
    max_lines_per_file: int | None,
    progress_desc: str,
    output_format: str,
    cleanup_csv: bool,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    process_nodes: dict[str, dict] = {}
    file_nodes: dict[str, dict] = {}
    netflow_nodes: dict[str, dict] = {}
    next_index = 0
    next_event_id = 0

    event_csv = out_dir / "event_table.csv"
    flush_every = 50_000
    buffer = []

    op_counts: dict[str, int] = {}
    skipped_no_ts = 0
    skipped_no_process = 0

    with event_csv.open("w", newline="") as f_out:
        writer = csv.writer(f_out)
        writer.writerow(EVENT_COLUMNS)

        for path in tqdm(edr_files, desc=progress_desc):
            attack_id = _infer_attack_id_from_edr_file(path)
            with path.open() as f_in:
                for line_idx, line in enumerate(f_in):
                    if (
                        max_lines_per_file is not None
                        and line_idx >= max_lines_per_file
                    ):
                        break

                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue

                    device_name = (
                        obj.get("device_name") or ""
                    ).strip() or "unknown_device"
                    ts_ns = _parse_cbc_timestamp_to_ns(
                        obj.get("device_timestamp") or obj.get("backend_timestamp")
                    )
                    if ts_ns is None:
                        skipped_no_ts += 1
                        continue

                    event_type = obj.get("type")
                    action_tokens = _split_action_tokens(obj.get("action"))

                    # Common process fields (present across most endpoint.event.*)
                    proc_guid = obj.get("process_guid")
                    proc_pid = _safe_int(obj.get("process_pid"))
                    proc_path = obj.get("process_path")
                    proc_cmd = obj.get("process_cmdline")

                    if not proc_guid and event_type not in {"endpoint.event.procstart"}:
                        skipped_no_process += 1
                        continue

                    if event_type == "endpoint.event.filemod":
                        file_path = obj.get("filemod_name")
                        if not file_path or not proc_guid:
                            continue

                        proc_node, next_index = _get_or_create_process(
                            process_nodes,
                            next_index,
                            device_name,
                            proc_guid,
                            proc_path,
                            proc_cmd,
                            pid=proc_pid,
                            attack_id=attack_id,
                        )
                        file_node, next_index = _get_or_create_file(
                            file_nodes, next_index, device_name, file_path
                        )

                        is_read = bool(FILE_READ_ACTIONS & action_tokens)
                        is_write = bool(FILE_WRITE_ACTIONS & action_tokens)
                        is_open = "ACTION_FILE_MOD_OPEN" in action_tokens

                        # READ: file -> process
                        if is_read:
                            op = "EVENT_READ"
                            op_counts[op] = op_counts.get(op, 0) + 1
                            _emit_event(
                                buffer,
                                flush_every,
                                writer,
                                [
                                    file_node["hash_id"],
                                    file_node["index_id"],
                                    op,
                                    proc_node["hash_id"],
                                    proc_node["index_id"],
                                    next_event_id,
                                    ts_ns,
                                ],
                            )
                            next_event_id += 1

                        # WRITE: process -> file
                        if is_write:
                            op = "EVENT_WRITE"
                            op_counts[op] = op_counts.get(op, 0) + 1
                            _emit_event(
                                buffer,
                                flush_every,
                                writer,
                                [
                                    proc_node["hash_id"],
                                    proc_node["index_id"],
                                    op,
                                    file_node["hash_id"],
                                    file_node["index_id"],
                                    next_event_id,
                                    ts_ns,
                                ],
                            )
                            next_event_id += 1

                        # OPEN (fallback): file -> process
                        if not is_read and not is_write and is_open:
                            op = "EVENT_OPEN"
                            op_counts[op] = op_counts.get(op, 0) + 1
                            _emit_event(
                                buffer,
                                flush_every,
                                writer,
                                [
                                    file_node["hash_id"],
                                    file_node["index_id"],
                                    op,
                                    proc_node["hash_id"],
                                    proc_node["index_id"],
                                    next_event_id,
                                    ts_ns,
                                ],
                            )
                            next_event_id += 1

                    elif event_type == "endpoint.event.moduleload":
                        mod_path = obj.get("modload_name")
                        if not mod_path or not proc_guid:
                            continue

                        proc_node, next_index = _get_or_create_process(
                            process_nodes,
                            next_index,
                            device_name,
                            proc_guid,
                            proc_path,
                            proc_cmd,
                            pid=proc_pid,
                            attack_id=attack_id,
                        )
                        file_node, next_index = _get_or_create_file(
                            file_nodes, next_index, device_name, mod_path
                        )

                        # Treat module load as an OPEN-like flow (module -> process).
                        op = "EVENT_OPEN"
                        op_counts[op] = op_counts.get(op, 0) + 1
                        _emit_event(
                            buffer,
                            flush_every,
                            writer,
                            [
                                file_node["hash_id"],
                                file_node["index_id"],
                                op,
                                proc_node["hash_id"],
                                proc_node["index_id"],
                                next_event_id,
                                ts_ns,
                            ],
                        )
                        next_event_id += 1

                    elif event_type == "endpoint.event.regmod":
                        reg_key = obj.get("regmod_name")
                        if not reg_key or not proc_guid:
                            continue

                        proc_node, next_index = _get_or_create_process(
                            process_nodes,
                            next_index,
                            device_name,
                            proc_guid,
                            proc_path,
                            proc_cmd,
                            pid=proc_pid,
                            attack_id=attack_id,
                        )
                        reg_node, next_index = _get_or_create_registry(
                            file_nodes, next_index, device_name, reg_key
                        )

                        is_read = bool(REG_READ_ACTIONS & action_tokens)
                        is_write = bool(REG_WRITE_ACTIONS & action_tokens)

                        if is_read:
                            op = "EVENT_READ"
                            op_counts[op] = op_counts.get(op, 0) + 1
                            _emit_event(
                                buffer,
                                flush_every,
                                writer,
                                [
                                    reg_node["hash_id"],
                                    reg_node["index_id"],
                                    op,
                                    proc_node["hash_id"],
                                    proc_node["index_id"],
                                    next_event_id,
                                    ts_ns,
                                ],
                            )
                            next_event_id += 1

                        if is_write:
                            op = "EVENT_WRITE"
                            op_counts[op] = op_counts.get(op, 0) + 1
                            _emit_event(
                                buffer,
                                flush_every,
                                writer,
                                [
                                    proc_node["hash_id"],
                                    proc_node["index_id"],
                                    op,
                                    reg_node["hash_id"],
                                    reg_node["index_id"],
                                    next_event_id,
                                    ts_ns,
                                ],
                            )
                            next_event_id += 1

                        if not is_read and not is_write:
                            # Fallback: treat as OPEN
                            op = "EVENT_OPEN"
                            op_counts[op] = op_counts.get(op, 0) + 1
                            _emit_event(
                                buffer,
                                flush_every,
                                writer,
                                [
                                    reg_node["hash_id"],
                                    reg_node["index_id"],
                                    op,
                                    proc_node["hash_id"],
                                    proc_node["index_id"],
                                    next_event_id,
                                    ts_ns,
                                ],
                            )
                            next_event_id += 1

                    elif event_type == "endpoint.event.netconn":
                        flow_node, next_index = _get_or_create_netflow(
                            netflow_nodes,
                            next_index,
                            device_name,
                            obj.get("netconn_community_id"),
                            obj.get("local_ip"),
                            obj.get("local_port"),
                            obj.get("remote_ip"),
                            obj.get("remote_port"),
                        )
                        if flow_node is None or not proc_guid:
                            continue

                        proc_node, next_index = _get_or_create_process(
                            process_nodes,
                            next_index,
                            device_name,
                            proc_guid,
                            proc_path,
                            proc_cmd,
                            pid=proc_pid,
                            attack_id=attack_id,
                        )

                        inbound = bool(_safe_int(obj.get("netconn_inbound")))
                        op = "EVENT_RECVFROM" if inbound else "EVENT_CONNECT"
                        op_counts[op] = op_counts.get(op, 0) + 1

                        # Outbound: process -> netflow, Inbound: netflow -> process
                        if inbound:
                            src, dst = flow_node, proc_node
                        else:
                            src, dst = proc_node, flow_node

                        _emit_event(
                            buffer,
                            flush_every,
                            writer,
                            [
                                src["hash_id"],
                                src["index_id"],
                                op,
                                dst["hash_id"],
                                dst["index_id"],
                                next_event_id,
                                ts_ns,
                            ],
                        )
                        next_event_id += 1

                    elif event_type == "endpoint.event.procstart":
                        parent_guid = obj.get("process_guid")
                        child_guid = obj.get("childproc_guid")
                        child_path = obj.get("childproc_name")
                        child_cmd = obj.get("target_cmdline")

                        if not parent_guid or not child_guid:
                            continue

                        child_pid = _safe_int(obj.get("childproc_pid"))
                        parent_node, next_index = _get_or_create_process(
                            process_nodes,
                            next_index,
                            device_name,
                            parent_guid,
                            obj.get("process_path"),
                            obj.get("process_cmdline"),
                            pid=proc_pid,
                            attack_id=attack_id,
                        )
                        child_node, next_index = _get_or_create_process(
                            process_nodes,
                            next_index,
                            device_name,
                            child_guid,
                            child_path,
                            child_cmd,
                            pid=child_pid,
                            attack_id=attack_id,
                        )

                        # Process spawn: parent -> child
                        op = "EVENT_FORK"
                        op_counts[op] = op_counts.get(op, 0) + 1
                        _emit_event(
                            buffer,
                            flush_every,
                            writer,
                            [
                                parent_node["hash_id"],
                                parent_node["index_id"],
                                op,
                                child_node["hash_id"],
                                child_node["index_id"],
                                next_event_id,
                                ts_ns,
                            ],
                        )
                        next_event_id += 1

                        # Executable image -> child process (optional but useful)
                        if child_path:
                            exe_node, next_index = _get_or_create_file(
                                file_nodes, next_index, device_name, child_path
                            )
                            op = "EVENT_EXECUTE"
                            op_counts[op] = op_counts.get(op, 0) + 1
                            _emit_event(
                                buffer,
                                flush_every,
                                writer,
                                [
                                    exe_node["hash_id"],
                                    exe_node["index_id"],
                                    op,
                                    child_node["hash_id"],
                                    child_node["index_id"],
                                    next_event_id,
                                    ts_ns,
                                ],
                            )
                            next_event_id += 1

                    else:
                        # Ignore procend, crossproc, scriptload, and any other CBC types for now.
                        continue

        if buffer:
            writer.writerows(buffer)
            buffer.clear()

    # Node tables
    write_csv_tables = output_format in {"csv", "both"}
    write_parquet_tables = output_format in {"parquet", "both"}

    def write_csv(path: Path, header: list[str], rows: list[list[str | int | None]]):
        with path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)

    if write_csv_tables:
        process_rows = [
            [
                n["node_uuid"],
                n["hash_id"],
                n["path"],
                n["cmd"],
                "" if n.get("pid") is None else n["pid"],
                n.get("attack", ""),
                n["index_id"],
            ]
            for n in sorted(process_nodes.values(), key=lambda x: x["index_id"])
        ]
        file_rows = [
            [n["node_uuid"], n["hash_id"], n["path"], n["index_id"]]
            for n in sorted(file_nodes.values(), key=lambda x: x["index_id"])
        ]
        netflow_rows = [
            [
                n["node_uuid"],
                n["hash_id"],
                n["src_addr"],
                n["src_port"],
                n["dst_addr"],
                n["dst_port"],
                n["index_id"],
            ]
            for n in sorted(netflow_nodes.values(), key=lambda x: x["index_id"])
        ]

        write_csv(out_dir / "process_node_table.csv", PROCESS_COLUMNS, process_rows)
        write_csv(out_dir / "file_node_table.csv", FILE_COLUMNS, file_rows)
        write_csv(out_dir / "netflow_node_table.csv", NETFLOW_COLUMNS, netflow_rows)

    if write_parquet_tables:
        _require_polars()

        process_schema = {
            "node_uuid": pl.Utf8,
            "hash_id": pl.Utf8,
            "path": pl.Utf8,
            "cmd": pl.Utf8,
            "pid": pl.Int64,
            "attack": pl.Utf8,
            "index_id": pl.Int64,
        }
        file_schema = {
            "node_uuid": pl.Utf8,
            "hash_id": pl.Utf8,
            "path": pl.Utf8,
            "index_id": pl.Int64,
        }
        netflow_schema = {
            "node_uuid": pl.Utf8,
            "hash_id": pl.Utf8,
            "src_addr": pl.Utf8,
            "src_port": pl.Utf8,
            "dst_addr": pl.Utf8,
            "dst_port": pl.Utf8,
            "index_id": pl.Int64,
        }

        def df_from_records(records: list[dict], schema: dict):
            return (
                pl.DataFrame(records, schema=schema)
                if records
                else pl.DataFrame(schema=schema)
            )

        df_from_records(list(process_nodes.values()), process_schema).select(
            PROCESS_COLUMNS
        ).write_parquet(out_dir / "process_node_table.parquet", compression="zstd")
        df_from_records(list(file_nodes.values()), file_schema).select(
            FILE_COLUMNS
        ).write_parquet(out_dir / "file_node_table.parquet", compression="zstd")
        df_from_records(list(netflow_nodes.values()), netflow_schema).select(
            NETFLOW_COLUMNS
        ).write_parquet(out_dir / "netflow_node_table.parquet", compression="zstd")

        event_parquet = out_dir / "event_table.parquet"
        event_dtypes = {
            "src_node": pl.Utf8,
            "src_index_id": pl.Int64,
            "operation": pl.Utf8,
            "dst_node": pl.Utf8,
            "dst_index_id": pl.Int64,
            "event_uuid": pl.Int64,
            "timestamp_rec": pl.Int64,
        }
        pl.scan_csv(str(event_csv), schema_overrides=event_dtypes).select(
            EVENT_COLUMNS
        ).sink_parquet(str(event_parquet), compression="zstd", mkdir=True)

        if cleanup_csv:
            for csv_name in [
                "event_table.csv",
                "process_node_table.csv",
                "file_node_table.csv",
                "netflow_node_table.csv",
            ]:
                csv_path = out_dir / csv_name
                if csv_path.exists():
                    csv_path.unlink()

    print("==== ATLASv2 CBC-EDR preprocessing complete ====")
    print(f"Output dir: {out_dir}")
    print(f"Events: {next_event_id:,}")
    if (out_dir / "event_table.csv").exists():
        print(f"  event_table.csv: {out_dir / 'event_table.csv'}")
    if (out_dir / "event_table.parquet").exists():
        print(f"  event_table.parquet: {out_dir / 'event_table.parquet'}")
    print(f"Process nodes: {len(process_nodes):,}")
    if (out_dir / "process_node_table.csv").exists():
        print(f"  process_node_table.csv: {out_dir / 'process_node_table.csv'}")
    if (out_dir / "process_node_table.parquet").exists():
        print(f"  process_node_table.parquet: {out_dir / 'process_node_table.parquet'}")
    print(f"File+registry nodes: {len(file_nodes):,}")
    if (out_dir / "file_node_table.csv").exists():
        print(f"  file_node_table.csv: {out_dir / 'file_node_table.csv'}")
    if (out_dir / "file_node_table.parquet").exists():
        print(f"  file_node_table.parquet: {out_dir / 'file_node_table.parquet'}")
    print(f"Netflow nodes: {len(netflow_nodes):,}")
    if (out_dir / "netflow_node_table.csv").exists():
        print(f"  netflow_node_table.csv: {out_dir / 'netflow_node_table.csv'}")
    if (out_dir / "netflow_node_table.parquet").exists():
        print(f"  netflow_node_table.parquet: {out_dir / 'netflow_node_table.parquet'}")
    if skipped_no_ts or skipped_no_process:
        print(f"Skipped (no timestamp): {skipped_no_ts:,}")
        print(f"Skipped (no process guid): {skipped_no_process:,}")
    if op_counts:
        print("Operation counts:")
        for op, c in sorted(op_counts.items(), key=lambda kv: (-kv[1], kv[0])):
            print(f"  {op}: {c:,}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert ATLASv2 CBC-EDR JSONL logs to Theseus tables (CSV/Parquet)"
    )
    parser.add_argument(
        "--raw_dir",
        type=str,
        required=True,
        help="ATLASv2 raw data root (contains benign/attack folders)",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./data/ATLASV2",
        help=(
            "Output directory. By default, writes two datasets to "
            "<out_dir>/atlasv2_h1 and <out_dir>/atlasv2_h2 (use --no_split_by_host to disable)."
        ),
    )
    parser.add_argument(
        "--max_lines_per_file",
        type=int,
        default=None,
        help="Debug: stop after N lines per JSONL file",
    )
    parser.add_argument(
        "--output_format",
        type=str,
        default="parquet",
        choices=["csv", "parquet", "both"],
        help="Output format for tables. Parquet requires polars.",
    )
    parser.add_argument(
        "--cleanup_csv",
        action="store_true",
        help="If set and output_format includes parquet, delete CSV tables after writing Parquet.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="all",
        choices=["all", "h1", "h2"],
        help="Only process one host (based on raw path).",
    )
    parser.add_argument(
        "--split_by_host",
        action="store_true",
        default=True,
        help="Write separate datasets to <out_dir>/atlasv2_h1 and <out_dir>/atlasv2_h2 (default).",
    )
    parser.add_argument(
        "--no_split_by_host",
        dest="split_by_host",
        action="store_false",
        help="Write a single dataset to <out_dir> (no per-host splitting).",
    )
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    if args.cleanup_csv and args.output_format == "csv":
        raise ValueError("--cleanup_csv requires --output_format parquet or both.")

    edr_files = sorted(raw_dir.rglob("cbc-edr/*.jsonl"))
    if not edr_files:
        raise FileNotFoundError(f"No cbc-edr jsonl files found under {raw_dir}")

    if args.split_by_host:
        hosts = ["h1", "h2"] if args.host == "all" else [args.host]
        for host in hosts:
            host_files = _filter_edr_files_by_host(edr_files, host)
            if not host_files:
                continue

            host_out_dir = out_dir / f"atlasv2_{host}"
            _convert_edr_files_to_csv_tables(
                host_files,
                host_out_dir,
                max_lines_per_file=args.max_lines_per_file,
                progress_desc=f"Processing cbc-edr files ({host})",
                output_format=args.output_format,
                cleanup_csv=args.cleanup_csv,
            )
        return

    filtered_files = _filter_edr_files_by_host(edr_files, args.host)
    if not filtered_files:
        raise FileNotFoundError(
            f"No cbc-edr jsonl files found under {raw_dir} for host={args.host}"
        )

    _convert_edr_files_to_csv_tables(
        filtered_files,
        out_dir,
        max_lines_per_file=args.max_lines_per_file,
        progress_desc=f"Processing cbc-edr files ({args.host})",
        output_format=args.output_format,
        cleanup_csv=args.cleanup_csv,
    )


if __name__ == "__main__":
    main()
