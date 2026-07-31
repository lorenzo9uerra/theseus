import csv
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from utils.utils import log, read_node_table


@dataclass(frozen=True)
class AtlasProcessLabel:
    attack_id: str
    process_uuid: str
    pid: int | None
    path: str
    label: str


def _get_reapr_dir(root_dir: str) -> str:
    """Returns the absolute path to the REAPR ground truth directory if present."""
    candidate = os.path.join(
        root_dir, "ground_truth", "reapr-ground-truth", "darpa-tc-engagement3"
    )
    return candidate if os.path.isdir(candidate) else ""


def _get_atlasv2_dir(root_dir: str) -> str:
    """Returns the absolute path to the ATLASv2 REAPR-style ground truth directory if present."""
    candidate = os.path.join(root_dir, "ground_truth", "reapr-ground-truth", "atlasv2")
    return candidate if os.path.isdir(candidate) else ""


def _is_atlasv2_dataset(config) -> bool:
    name = getattr(getattr(config, "dataset_info", None), "name", "") or ""
    return str(name).lower().startswith("atlasv2")


def _infer_atlas_host_from_dataset_name(dataset_name: str) -> str | None:
    name = (dataset_name or "").lower()
    if name.endswith("_h1") or name.endswith("-h1") or name.endswith("h1"):
        return "h1"
    if name.endswith("_h2") or name.endswith("-h2") or name.endswith("h2"):
        return "h2"
    return None


def _normalize_windows_path(path: str) -> str:
    if not path:
        return ""
    normalized = str(path).strip().lower().replace("\\", "/")
    while "//" in normalized:
        normalized = normalized.replace("//", "/")
    return normalized


def _safe_int(value) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    try:
        string_value = str(value).strip()
        if string_value == "":
            return None
        if string_value.endswith(".0"):
            string_value = string_value[:-2]
        return int(string_value)
    except Exception:
        return None


def _infer_host_from_device_name(device_name: str) -> str | None:
    device = (device_name or "").lower()
    if device.endswith("h1") or "-h1" in device:
        return "h1"
    if device.endswith("h2") or "-h2" in device:
        return "h2"
    return None


def _get_atlasv2_labels_dir() -> str | None:
    override = os.environ.get("ATLASV2_LABEL_DIR")
    if override:
        override = os.path.abspath(override)
        return override if os.path.isdir(override) else None

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    atlas_dir = _get_atlasv2_dir(repo_root)
    return atlas_dir or None


def _normalize_atlas_attack_id(attack_id: str) -> str:
    attack_id = (attack_id or "").strip().lower()
    if attack_id and not attack_id.startswith("atlasv2/"):
        attack_id = f"atlasv2/{attack_id}"
    return attack_id


def _atlas_process_uuid_key(process_uuid: str) -> str:
    return (process_uuid or "").strip().split("|")[-1].lower()


def _parse_atlasv2_label_file(label_path: Path) -> list[AtlasProcessLabel]:
    labels: list[AtlasProcessLabel] = []
    with label_path.open(newline="") as file:
        reader = csv.DictReader(file, skipinitialspace=True)
        for raw_row in reader:
            if not raw_row:
                continue

            row = {
                (key or "").strip().lower(): (value or "").strip()
                for key, value in raw_row.items()
            }
            attack_id = _normalize_atlas_attack_id(row.get("attack", ""))
            label = row.get("label", "").lower()
            pid = _safe_int(row.get("process_id"))
            path = _normalize_windows_path(row.get("process_name", ""))
            process_uuid = _atlas_process_uuid_key(row.get("process_uuid", ""))

            if not attack_id:
                continue
            if label not in {"attack", "contaminated"}:
                continue
            if not process_uuid and (pid is None or not path):
                continue

            labels.append(
                AtlasProcessLabel(
                    attack_id=attack_id,
                    process_uuid=process_uuid,
                    pid=pid,
                    path=path,
                    label=label,
                )
            )
    return labels


def load_atlasv2_process_labels(labels_dir: str | Path) -> list[AtlasProcessLabel]:
    """Load revised UUID labels, falling back to the frozen legacy CSV."""
    labels_dir = Path(labels_dir)
    revised_files = sorted(labels_dir.glob("*.labels"))
    if revised_files:
        labels = [
            label
            for label_path in revised_files
            for label in _parse_atlasv2_label_file(label_path)
        ]
        log(
            f"ATLASv2 labels: loaded {len(labels)} rows from "
            f"{len(revised_files)} revised UUID label files"
        )
        return labels

    legacy_path = labels_dir / "atlasv2_labels.csv"
    if not legacy_path.is_file():
        raise ValueError(
            f"ATLASv2 ground truth not found under {labels_dir} "
            "(expected *.labels or atlasv2_labels.csv)."
        )

    labels = _parse_atlasv2_label_file(legacy_path)
    log(f"ATLASv2 labels: loaded {len(labels)} rows from legacy CSV")
    return labels


def _get_atlasv2_ground_truth(config):
    """Load ATLASv2 process-node ground truth and map it to node index IDs."""
    labels_dir = _get_atlasv2_labels_dir()
    if not labels_dir:
        raise ValueError("ATLASv2 ground truth directory not found.")

    data_dir = os.path.join(config.data_dir, config.dataset)
    process_df = read_node_table(
        data_dir,
        "process_node_table",
        columns=["index_id", "node_uuid", "path", "pid", "attack"],
    )
    if process_df is None:
        raise ValueError(f"process_node_table not found under {data_dir}")

    by_attack_pid_path: dict[tuple[str, int, str], set[int]] = defaultdict(set)
    by_host_pid_path: dict[tuple[str, int, str], set[int]] = defaultdict(set)
    by_process_uuid: dict[str, set[int]] = defaultdict(set)

    for row in process_df.iter_rows(named=True):
        idx = _safe_int(row.get("index_id"))
        pid = _safe_int(row.get("pid"))
        path = _normalize_windows_path(row.get("path", ""))
        attack_id = (row.get("attack") or "").strip()
        node_uuid = (row.get("node_uuid") or "").strip()
        device_name = node_uuid.split("|", 1)[0] if "|" in node_uuid else ""
        host = _infer_host_from_device_name(device_name)

        if idx is None:
            continue

        process_uuid = _atlas_process_uuid_key(node_uuid)
        if process_uuid:
            by_process_uuid[process_uuid].add(int(idx))

        if attack_id and pid is not None and path:
            by_attack_pid_path[(attack_id, pid, path)].add(int(idx))
        if host and pid is not None and path:
            by_host_pid_path[(host, pid, path)].add(int(idx))

    del process_df

    labels = load_atlasv2_process_labels(labels_dir)

    dataset_host = _infer_atlas_host_from_dataset_name(config.dataset_info.name)
    if dataset_host not in {"h1", "h2"}:
        raise ValueError(
            f"Could not infer ATLASv2 host from dataset name '{config.dataset_info.name}'"
        )

    attack_metadata = {}
    host_labels = [
        label
        for label in labels
        if label.attack_id.startswith(f"atlasv2/{dataset_host}-")
    ]
    labels_by_attack: dict[str, list[AtlasProcessLabel]] = defaultdict(list)
    for label in host_labels:
        labels_by_attack[label.attack_id].append(label)

    for attack_id, process_labels in labels_by_attack.items():
        attack_node_ids: set[int] = set()
        contaminated_node_ids: set[int] = set()
        attack_missing = 0
        contaminated_missing = 0

        for process_label in process_labels:
            target_ids = (
                attack_node_ids
                if process_label.label == "attack"
                else contaminated_node_ids
            )
            if process_label.process_uuid:
                matches = by_process_uuid.get(process_label.process_uuid, set())
                if len(matches) > 1:
                    raise ValueError(
                        f"ATLASv2 process UUID '{process_label.process_uuid}' "
                        f"maps to multiple nodes: {sorted(matches)}"
                    )
            elif process_label.pid is None:
                matches = set()
            else:
                matches = by_attack_pid_path.get(
                    (attack_id, process_label.pid, process_label.path), set()
                )
                if not matches:
                    matches = by_host_pid_path.get(
                        (dataset_host, process_label.pid, process_label.path), set()
                    )

            if matches:
                target_ids.update(matches)
            elif process_label.label == "attack":
                attack_missing += 1
            else:
                contaminated_missing += 1

        if attack_missing:
            log(
                f"WARNING: {attack_missing}/"
                f"{sum(label.label == 'attack' for label in process_labels)} "
                f"ATLASv2 attack processes from '{attack_id}' not found in node tables"
            )
        if contaminated_missing:
            log(
                f"WARNING: {contaminated_missing}/"
                f"{sum(label.label == 'contaminated' for label in process_labels)} "
                f"ATLASv2 contaminated processes from '{attack_id}' not found in node tables"
            )

        attack_metadata[attack_id] = {
            "nids": attack_node_ids,
            "contaminated_nids": contaminated_node_ids,
        }

    if not attack_metadata:
        raise ValueError("ATLASv2 ground truth produced no attack metadata.")

    total_attack_nodes = sum(len(value["nids"]) for value in attack_metadata.values())
    total_contaminated_nodes = sum(
        len(value["contaminated_nids"]) for value in attack_metadata.values()
    )
    log(
        f"ATLASv2 ground truth: {len(attack_metadata)} attack chains, "
        f"{total_attack_nodes} attack nodes, {total_contaminated_nodes} contaminated nodes"
    )
    if total_attack_nodes == 0:
        raise ValueError(
            "ATLASv2 ground truth mapped to 0 nodes. Ensure your generated "
            "process_node_table includes 'pid' and 'attack' columns and that "
            "paths match."
        )

    return attack_metadata


def _get_reapr_csv_path(config) -> str | None:
    """Gets the REAPR CSV path for the specific dataset in config."""
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    reapr_dir = _get_reapr_dir(repo_root)
    if not reapr_dir:
        return None

    csv_name = f"{config.dataset_info.name.lower().split('_')[0]}_labels.csv"
    if not csv_name:
        return None

    csv_path = os.path.join(reapr_dir, csv_name)
    return csv_path if os.path.isfile(csv_path) else None


def _parse_reapr_labels_by_attack(csv_path: str) -> dict[str, dict[str, set[str]]]:
    """Parses REAPR CSVs into a mapping of attack chains to sets of node UUIDs."""
    attack_to_uuids = {}

    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        first = True
        for row in reader:
            if not row:
                continue
            row = [c.strip() for c in row]

            if first:
                if row[0].lower() == "attack_chain" or row[1].lower() == "uuid":
                    first = False
                    continue
                first = False

            if len(row) < 4:
                continue

            attack_chain, uuid, label = row[0], row[1], row[-1].lower()
            if label in {"attack", "contaminated"} and uuid and attack_chain:
                if attack_chain not in attack_to_uuids:
                    attack_to_uuids[attack_chain] = {
                        "attack": set(),
                        "contaminated": set(),
                    }
                attack_to_uuids[attack_chain][label].add(uuid)

    return attack_to_uuids


def get_uuid_to_index_id_mapping(config):
    """Builds a map from node UUIDs to integer index IDs from dataset tables."""
    data_dir = os.path.join(config.data_dir, config.dataset)
    uuid_to_index_id = {}

    for table_name in ["file_node_table", "netflow_node_table", "process_node_table"]:
        df = read_node_table(data_dir, table_name, columns=["index_id", "node_uuid"])
        if df is not None:
            for row in df.iter_rows(named=True):
                uuid_to_index_id[row["node_uuid"]] = row["index_id"]
            del df

    return uuid_to_index_id


def get_excluded_node_ids(config) -> set[int]:
    excluded_attack_chains = set(
        getattr(getattr(config, "dataset_info", None), "excluded_attack_chains", [])
        or []
    )

    reapr_csv = _get_reapr_csv_path(config)
    if not reapr_csv:
        return set()

    uuid_to_index_id = get_uuid_to_index_id_mapping(config)
    attack_to_uuids = _parse_reapr_labels_by_attack(reapr_csv)

    excluded_uuids: set[str] = set()
    for chain_name in excluded_attack_chains:
        chain_data = attack_to_uuids.get(chain_name)
        if not chain_data:
            continue
        excluded_uuids.update(chain_data.get("attack", set()) or set())
        excluded_uuids.update(chain_data.get("contaminated", set()) or set())

    excluded_node_ids: set[int] = set()
    missing = 0
    for node_uuid in excluded_uuids:
        idx = uuid_to_index_id.get(node_uuid)
        if idx is None:
            missing += 1
            continue
        excluded_node_ids.add(int(idx))

    if excluded_node_ids:
        log(
            f"Excluding {len(excluded_node_ids)} node(s) from evaluation metrics: {sorted(excluded_attack_chains)}"
        )
    if missing:
        log(
            f"WARNING: {missing}/{len(excluded_uuids)} excluded UUIDs not found in node tables"
        )

    return excluded_node_ids


def get_ground_truth(config):
    """Loads REAPR ground truth labels and maps them to node IDs."""
    if _is_atlasv2_dataset(config):
        return _get_atlasv2_ground_truth(config)

    uuid_to_index_id = get_uuid_to_index_id_mapping(config)
    attack_metadata = {}

    reapr_csv = _get_reapr_csv_path(config)
    if not reapr_csv:
        raise ValueError(
            f"REAPR ground truth not available for '{config.dataset_info.name}'"
        )

    attack_to_uuids = _parse_reapr_labels_by_attack(reapr_csv)

    for chain_name, chain_data in attack_to_uuids.items():
        attack_node_ids = set()
        contaminated_node_ids = set()

        for uuid_set, target_set, label_type in [
            (chain_data["attack"], attack_node_ids, "attack"),
            (chain_data["contaminated"], contaminated_node_ids, "contaminated"),
        ]:
            missing_count = 0
            for node_uuid in uuid_set:
                if node_uuid in uuid_to_index_id:
                    target_set.add(int(uuid_to_index_id[node_uuid]))
                else:
                    missing_count += 1

            if missing_count > 0:
                log(
                    f"WARNING: {missing_count}/{len(uuid_set)} {label_type} UUIDs from '{chain_name}' not found"
                )

        log(
            f"Attack ('{chain_name}'): {len(attack_node_ids)} attack nodes, {len(contaminated_node_ids)} contaminated nodes"
        )
        attack_metadata[chain_name] = {
            "nids": attack_node_ids,
            "contaminated_nids": contaminated_node_ids,
        }

    return attack_metadata
