#!/usr/bin/env python3
"""Simple script to verify that all ground truth nodes are process nodes."""

import argparse
import csv
import os
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.utils import read_node_table  # noqa: E402


@dataclass
class GroundTruthStats:
    """Statistics about ground truth node types."""

    dataset: str
    total_attack_uuids: int
    total_contaminated_uuids: int
    total_unique_uuids: int
    process_nodes_found: int
    file_nodes_found: int
    netflow_nodes_found: int
    nodes_not_found: int
    attack_in_process: int
    attack_in_file: int
    attack_in_netflow: int
    attack_not_found: int
    contaminated_in_process: int
    contaminated_in_file: int
    contaminated_in_netflow: int
    contaminated_not_found: int


def load_ground_truth(
    dataset_name: str, verbose: bool = True
) -> tuple[set[str], set[str]]:
    """Load ground truth labels from REAPr benchmark CSV files."""
    gt_path = os.path.join(
        PROJECT_ROOT,
        "ground_truth",
        "reapr-ground-truth",
        "darpa-tc-engagement3",
        f"{dataset_name}_labels.csv",
    )

    if not os.path.exists(gt_path):
        if verbose:
            print(f"Warning: Ground truth file not found at {gt_path}")
        return set(), set()

    if verbose:
        print(f"Loading ground truth from {gt_path}...")

    attack_uuids = set()
    contaminated_uuids = set()

    with open(gt_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        first = True
        line_num = 0

        for row in reader:
            line_num += 1

            if not row:
                continue  # Skip empty rows

            # Normalize whitespace
            row = [c.strip() for c in row]

            # Skip header row
            if first and (row[0].lower() == "attack_chain" or row[1].lower() == "uuid"):
                first = False
                continue
            first = False

            if len(row) < 4:
                if verbose:
                    print(f"Warning: Skipping malformed row {line_num}: {row}")
                continue

            uuid = row[1]
            label = row[-1].lower().strip()

            if uuid:
                if label == "attack":
                    attack_uuids.add(uuid)
                elif label == "contaminated":
                    contaminated_uuids.add(uuid)
                elif verbose:
                    print(f"Warning: Unknown label '{label}' at line {line_num}")

    if verbose:
        print(f"  Attack UUIDs: {len(attack_uuids)}")
        print(f"  Contaminated UUIDs: {len(contaminated_uuids)}")
        print(f"  Total unique UUIDs: {len(attack_uuids | contaminated_uuids)}")

    return attack_uuids, contaminated_uuids


def load_node_tables(dataset_name, verbose=True):
    """Load all node tables and create UUID -> node_type mapping.

    Returns:
        dict: uuid -> node_type (process/file/netflow)
    """
    data_dir = os.path.join(PROJECT_ROOT, "data", "DARPA", dataset_name)
    uuid_to_type = {}

    def _get_table_path(table_name):
        """Helper to get the actual path used for a table."""
        parquet_path = os.path.join(data_dir, f"{table_name}.parquet")
        csv_path = os.path.join(data_dir, f"{table_name}.csv")
        if os.path.exists(parquet_path):
            return parquet_path
        elif os.path.exists(csv_path):
            return csv_path
        return None

    # Load process nodes
    df = read_node_table(data_dir, "process_node_table", columns=["node_uuid"])
    if df is not None:
        if verbose:
            path = _get_table_path("process_node_table")
            print(f"  Loading {path}...")
        for uuid in df["node_uuid"].to_list():
            uuid_to_type[uuid] = "process"
        if verbose:
            print(f"    Found {len(df)} process nodes")

    # Load file nodes
    df = read_node_table(data_dir, "file_node_table", columns=["node_uuid"])
    if df is not None:
        if verbose:
            path = _get_table_path("file_node_table")
            print(f"  Loading {path}...")
        for uuid in df["node_uuid"].to_list():
            uuid_to_type[uuid] = "file"
        if verbose:
            print(f"    Found {len(df)} file nodes")

    # Load netflow nodes
    df = read_node_table(data_dir, "netflow_node_table", columns=["node_uuid"])
    if df is not None:
        if verbose:
            path = _get_table_path("netflow_node_table")
            print(f"  Loading {path}...")
        for uuid in df["node_uuid"].to_list():
            uuid_to_type[uuid] = "netflow"
        if verbose:
            print(f"    Found {len(df)} netflow nodes")

    # Also check subject_node_table (CADETS uses this instead of process)
    df = read_node_table(data_dir, "subject_node_table", columns=["node_uuid"])
    if df is not None:
        if verbose:
            path = _get_table_path("subject_node_table")
            print(f"  Loading {path}...")
        for uuid in df["node_uuid"].to_list():
            # Subject nodes are process nodes
            uuid_to_type[uuid] = "process"
        if verbose:
            print(f"    Found {len(df)} subject (process) nodes")

    if verbose:
        print(f"  Total nodes in tables: {len(uuid_to_type)}")

    return uuid_to_type


def verify_dataset(dataset_name: str, verbose: bool = True) -> GroundTruthStats | None:
    """Verify ground truth node types for a single dataset."""
    if verbose:
        print("=" * 80)
        print(f"Verifying Ground Truth Node Types: {dataset_name}")
        print("=" * 80)
        print()

    # Get base name for ground truth file (lowercase, without _E3 suffix)
    base_name = dataset_name.split("_")[0].lower()

    # Load ground truth
    attack_uuids, contaminated_uuids = load_ground_truth(base_name, verbose=verbose)

    if not attack_uuids and not contaminated_uuids:
        if verbose:
            print(f"No ground truth found for {dataset_name}")
        return None

    all_gt_uuids = attack_uuids | contaminated_uuids

    # Load node tables
    if verbose:
        print()

    uuid_to_type = load_node_tables(dataset_name, verbose=verbose)

    if not uuid_to_type:
        if verbose:
            print(f"ERROR: No node tables found for {dataset_name}")
        return None

    # Analyze node types
    if verbose:
        print()
        print("-" * 80)
        print("Ground Truth Node Type Analysis")
        print("-" * 80)

    # Count by type
    type_counts = {"process": 0, "file": 0, "netflow": 0, "not_found": 0}
    attack_type_counts = {"process": 0, "file": 0, "netflow": 0, "not_found": 0}
    contaminated_type_counts = {"process": 0, "file": 0, "netflow": 0, "not_found": 0}

    not_found_uuids = []
    non_process_uuids = []

    for uuid in all_gt_uuids:
        node_type = uuid_to_type.get(uuid, "not_found")
        type_counts[node_type] += 1

        if node_type == "not_found":
            not_found_uuids.append(uuid)
        elif node_type != "process":
            non_process_uuids.append((uuid, node_type))

        # Track attack vs contaminated separately
        if uuid in attack_uuids:
            attack_type_counts[node_type] += 1
        if uuid in contaminated_uuids:
            contaminated_type_counts[node_type] += 1

    if verbose:
        print("\nOverall Ground Truth UUID Distribution:")
        print(
            f"  Process nodes: {type_counts['process']} ({100 * type_counts['process'] / len(all_gt_uuids):.1f}%)"
        )
        print(
            f"  File nodes: {type_counts['file']} ({100 * type_counts['file'] / len(all_gt_uuids):.1f}%)"
        )
        print(
            f"  Netflow nodes: {type_counts['netflow']} ({100 * type_counts['netflow'] / len(all_gt_uuids):.1f}%)"
        )
        print(
            f"  Not found: {type_counts['not_found']} ({100 * type_counts['not_found'] / len(all_gt_uuids):.1f}%)"
        )

        print("\nAttack UUIDs Distribution:")
        print(
            f"  Process nodes: {attack_type_counts['process']} ({100 * attack_type_counts['process'] / len(attack_uuids):.1f}%)"
        )
        print(
            f"  File nodes: {attack_type_counts['file']} ({100 * attack_type_counts['file'] / len(attack_uuids):.1f}%)"
        )
        print(
            f"  Netflow nodes: {attack_type_counts['netflow']} ({100 * attack_type_counts['netflow'] / len(attack_uuids):.1f}%)"
        )
        print(
            f"  Not found: {attack_type_counts['not_found']} ({100 * attack_type_counts['not_found'] / len(attack_uuids):.1f}%)"
        )

        print("\nContaminated UUIDs Distribution:")
        print(
            f"  Process nodes: {contaminated_type_counts['process']} ({100 * contaminated_type_counts['process'] / len(contaminated_uuids):.1f}%)"
        )
        print(
            f"  File nodes: {contaminated_type_counts['file']} ({100 * contaminated_type_counts['file'] / len(contaminated_uuids):.1f}%)"
        )
        print(
            f"  Netflow nodes: {contaminated_type_counts['netflow']} ({100 * contaminated_type_counts['netflow'] / len(contaminated_uuids):.1f}%)"
        )
        print(
            f"  Not found: {contaminated_type_counts['not_found']} ({100 * contaminated_type_counts['not_found'] / len(contaminated_uuids):.1f}%)"
        )

    # Show non-process nodes if any
    if non_process_uuids and verbose:
        print(
            f"\nWARNING: Found {len(non_process_uuids)} non-process nodes in ground truth:"
        )
        for i, (uuid, node_type) in enumerate(non_process_uuids[:20]):
            is_attack = "ATTACK" if uuid in attack_uuids else "CONTAMINATED"
            print(f"  {i + 1}. {uuid} - {node_type} ({is_attack})")
        if len(non_process_uuids) > 20:
            print(f"  ... and {len(non_process_uuids) - 20} more")

    # Show not found UUIDs if any
    if not_found_uuids and verbose:
        print(f"\nWARNING: Found {len(not_found_uuids)} UUIDs not in any node table:")
        for i, uuid in enumerate(not_found_uuids[:20]):
            is_attack = "ATTACK" if uuid in attack_uuids else "CONTAMINATED"
            print(f"  {i + 1}. {uuid} ({is_attack})")
        if len(not_found_uuids) > 20:
            print(f"  ... and {len(not_found_uuids) - 20} more")

    # Summary verdict
    if verbose:
        print(f"\n{'=' * 80}")
        if type_counts["process"] == len(all_gt_uuids):
            print("ALL ground truth nodes are process nodes")
        else:
            print("NOT all ground truth nodes are process nodes:")
            print(
                f"  - {type_counts['process']}/{len(all_gt_uuids)} are process nodes ({100 * type_counts['process'] / len(all_gt_uuids):.1f}%)"
            )
            print(f"  - {type_counts['file']} file nodes")
            print(f"  - {type_counts['netflow']} netflow nodes")
            print(f"  - {type_counts['not_found']} not found in any table")
        print("=" * 80)

    return GroundTruthStats(
        dataset=dataset_name,
        total_attack_uuids=len(attack_uuids),
        total_contaminated_uuids=len(contaminated_uuids),
        total_unique_uuids=len(all_gt_uuids),
        process_nodes_found=type_counts["process"],
        file_nodes_found=type_counts["file"],
        netflow_nodes_found=type_counts["netflow"],
        nodes_not_found=type_counts["not_found"],
        attack_in_process=attack_type_counts["process"],
        attack_in_file=attack_type_counts["file"],
        attack_in_netflow=attack_type_counts["netflow"],
        attack_not_found=attack_type_counts["not_found"],
        contaminated_in_process=contaminated_type_counts["process"],
        contaminated_in_file=contaminated_type_counts["file"],
        contaminated_in_netflow=contaminated_type_counts["netflow"],
        contaminated_not_found=contaminated_type_counts["not_found"],
    )


def print_summary(results):
    """Print summary table across all datasets."""
    print("\n" + "=" * 120)
    print("SUMMARY: Ground Truth Node Types Across All Datasets")
    print("=" * 120)

    print(
        f"\n{'Dataset':<20} {'Total UUIDs':<15} {'Process':<15} {'File':<10} {'Netflow':<10} {'Not Found':<12}"
    )
    print("-" * 120)

    all_process = True
    for stats in results:
        process_pct = 100 * stats.process_nodes_found / stats.total_unique_uuids

        status = (
            "OK" if stats.process_nodes_found == stats.total_unique_uuids else "FAIL"
        )

        print(
            f"{status} {stats.dataset:<18} {stats.total_unique_uuids:<15} "
            f"{stats.process_nodes_found:<6} ({process_pct:>5.1f}%) "
            f"{stats.file_nodes_found:<10} "
            f"{stats.netflow_nodes_found:<10} "
            f"{stats.nodes_not_found:<12}"
        )

        if stats.process_nodes_found != stats.total_unique_uuids:
            all_process = False

    print("=" * 120)

    if all_process:
        print(
            "\nCONCLUSION: All ground truth nodes across ALL datasets are process nodes."
        )
        print(
            "  -> Evaluation on process nodes only is appropriate and matches ground truth."
        )
    else:
        print("\nCONCLUSION: NOT all ground truth nodes are process nodes.")
        print("  -> Some datasets have file/netflow nodes in ground truth.")
        print("  -> Consider whether evaluation should include all node types.")

    print("=" * 120)


def main():
    parser = argparse.ArgumentParser(
        description="Verify that all ground truth nodes are process nodes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["CADETS_E3", "THEIA_E3", "TRACE_E3", "FIVEDIRECTIONS_E3"],
        help="Single dataset to verify (default: all datasets)",
    )
    parser.add_argument(
        "--verbose", action="store_true", default=True, help="Print detailed output"
    )
    parser.add_argument("--quiet", action="store_true", help="Only print summary")

    args = parser.parse_args()

    verbose = args.verbose and not args.quiet

    # Determine which datasets to check
    if args.dataset:
        datasets = [args.dataset]
    else:
        datasets = ["CADETS_E3", "THEIA_E3", "TRACE_E3", "FIVEDIRECTIONS_E3"]

    # Verify each dataset
    results = []
    for dataset in datasets:
        stats = verify_dataset(dataset, verbose=verbose)
        if stats:
            results.append(stats)
        if verbose and len(datasets) > 1:
            print("\n")

    # Print summary if multiple datasets
    if len(results) > 1:
        print_summary(results)


if __name__ == "__main__":
    main()
