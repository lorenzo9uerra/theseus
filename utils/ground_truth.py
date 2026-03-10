import csv
import os

from utils.utils import log, read_node_table


def _get_reapr_dir(root_dir: str) -> str:
    """Returns the absolute path to the REAPR ground truth directory if present."""
    candidate = os.path.join(
        root_dir, "ground_truth", "reapr-ground-truth", "darpa-tc-engagement3"
    )
    return candidate if os.path.isdir(candidate) else ""


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
