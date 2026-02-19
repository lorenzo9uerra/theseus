import csv
import os
import warnings

import numpy as np
import torch
from model.autoencoder import build_model
from model.eval import two_level_evaluation
from utils.loaddata import load_entity_level_dataset, load_metadata

from utils.config import build_args
from utils.utils import set_random_seed

warnings.filterwarnings("ignore")

_MAGIC_ROOT = os.path.dirname(os.path.abspath(__file__))
_EXCLUDED_ATTACK_CHAINS = {"wwtawwtal_bad_neighborhood"}

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. Run 'pip install wandb' to enable logging.")


def _resolve_device(device_arg: int) -> torch.device:
    if device_arg < 0:
        return torch.device("cpu")
    if not torch.cuda.is_available():
        print("Warning: CUDA not available. Falling back to CPU.")
        return torch.device("cpu")
    device_count = torch.cuda.device_count()
    if device_count and device_arg >= device_count:
        print(
            f"Warning: requested cuda:{device_arg} but only {device_count} CUDA device(s) available. "
            "Using cuda:0."
        )
        device_arg = 0
    device = torch.device(f"cuda:{device_arg}")

    try:
        import dgl

        test_g = dgl.graph((torch.tensor([0]), torch.tensor([0])), num_nodes=1)
        test_g.to(device)
    except Exception as exc:
        msg = str(exc).lower()
        if "cuda is not enabled" in msg or "device api cuda is not enabled" in msg:
            print(
                "Warning: DGL was installed without CUDA support (CPU-only build). "
                "Falling back to CPU. To enable GPU, install a CUDA-enabled DGL wheel "
                "(e.g., dgl-cu116 for CUDA 11.6)."
            )
        else:
            print(f"Warning: failed to use DGL on {device}: {exc}. Falling back to CPU.")
        return torch.device("cpu")

    return device


def _load_excluded_uuids(dataset_name: str) -> set[str]:
    if not _EXCLUDED_ATTACK_CHAINS:
        return set()

    project_root = os.path.abspath(os.path.join(_MAGIC_ROOT, os.pardir, os.pardir))
    gt_dir = os.path.join(
        project_root, "ground_truth", "reapr-ground-truth", "darpa-tc-engagement3"
    )
    gt_path = os.path.join(gt_dir, f"{dataset_name}_labels.csv")
    if not os.path.isfile(gt_path):
        return set()

    excluded = set()
    with open(gt_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            chain = (row.get("attack_chain") or "").strip()
            uuid = (row.get("uuid") or "").strip()
            if chain in _EXCLUDED_ATTACK_CHAINS and uuid:
                excluded.add(uuid)

    if excluded:
        print(
            f"Excluding {len(excluded)} UUIDs from evaluation metrics: {sorted(_EXCLUDED_ATTACK_CHAINS)}"
        )

    return excluded


def main(main_args):
    device = _resolve_device(main_args.device)
    dataset_name = main_args.dataset

    # Initialize wandb if available
    if WANDB_AVAILABLE and hasattr(main_args, "wandb") and main_args.wandb:
        wandb.init(
            project=main_args.wandb_project
            if hasattr(main_args, "wandb_project")
            else "MAGIC-eval",
            name=f"eval_{dataset_name}",
            config={"dataset": dataset_name, "device": str(device)},
        )

    main_args.num_hidden = 64
    main_args.num_layers = 3
    set_random_seed(0)

    metadata = load_metadata(dataset_name)
    main_args.n_dim = metadata["node_feature_dim"]
    main_args.e_dim = metadata["edge_feature_dim"]
    model = build_model(main_args)

    # Load seed-specific checkpoint
    seed = getattr(main_args, "seed", 0)
    checkpoint_path = os.path.join(
        _MAGIC_ROOT, "checkpoints", f"checkpoint-{dataset_name}-seed{seed}.pt"
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device)
    model.eval()

    n_train = metadata["n_train"]
    n_val = metadata.get("n_val", 0)
    n_test = metadata["n_test"]

    # Require two-level labels
    two_level_labels = metadata.get("two_level_labels", None)
    if not two_level_labels:
        raise ValueError(
            f"Dataset {dataset_name} is missing 'two_level_labels' in metadata.json. "
            "Please re-run the data parser (utils/csv_parser_daily.py) to generate proper labels."
        )

    with torch.no_grad():
        # Determine the node type ID for SUBJECT_PROCESS
        # Assume node_type_dict maps type names to IDs
        node_type_dict = metadata.get("node_type_dict", {})
        process_type_id = None

        if not node_type_dict:
            print(
                "Warning: node_type_dict not found in metadata. Evaluating on all nodes."
            )
            print("  To enable process-only filtering, re-run the data parser:")
            print(
                "  python utils/csv_parser_daily.py --dataset <name> --csv_dir <dir> --ground_truth <file>"
            )
            filter_process_only = False
        else:
            for type_name, type_id in node_type_dict.items():
                if "SUBJECT" in type_name or "PROCESS" in type_name:
                    process_type_id = type_id
                    print(f"Found process type: {type_name} with ID {type_id}")
                    break

            if process_type_id is None:
                print(
                    "Warning: Could not identify process node type. Evaluating on all nodes."
                )
                filter_process_only = False
            else:
                filter_process_only = True
                print(
                    f"Filtering evaluation to process nodes only (type ID: {process_type_id})"
                )

        # Embed training graphs (with process filtering)
        x_train = []
        n_train_nodes = 0
        train_process_mask = []
        for i in range(n_train):
            g = load_entity_level_dataset(dataset_name, "train", i).to(device)
            embeddings = model.embed(g).cpu().numpy()

            if filter_process_only:
                # Get node types and filter to process nodes only
                node_types = g.ndata["type"].cpu().numpy()
                mask = node_types == process_type_id
                embeddings = embeddings[mask]
                train_process_mask.append(mask)

            x_train.append(embeddings)
            n_train_nodes += embeddings.shape[0]
            del g
        x_train = np.concatenate(x_train, axis=0)

        # Embed validation graphs
        x_val = []
        n_val_nodes = 0
        val_process_mask = []
        if n_val > 0:
            for i in range(n_val):
                g = load_entity_level_dataset(dataset_name, "val", i).to(device)
                embeddings = model.embed(g).cpu().numpy()

                if filter_process_only:
                    node_types = g.ndata["type"].cpu().numpy()
                    mask = node_types == process_type_id
                    embeddings = embeddings[mask]
                    val_process_mask.append(mask)

                x_val.append(embeddings)
                n_val_nodes += embeddings.shape[0]
                del g
            x_val = np.concatenate(x_val, axis=0)

        # Embed test graphs
        x_test = []
        n_test_nodes = 0
        test_process_mask = []
        for i in range(n_test):
            g = load_entity_level_dataset(dataset_name, "test", i).to(device)
            embeddings = model.embed(g).cpu().numpy()

            if filter_process_only:
                node_types = g.ndata["type"].cpu().numpy()
                mask = node_types == process_type_id
                embeddings = embeddings[mask]
                test_process_mask.append(mask)

            x_test.append(embeddings)
            n_test_nodes += embeddings.shape[0]
            del g
        x_test = np.concatenate(x_test, axis=0)

        print("\nDataset Split Info:")
        print(f"  Training nodes: {n_train_nodes}")
        print(f"  Validation nodes: {n_val_nodes}")
        print(f"  Test nodes: {n_test_nodes}")
        if filter_process_only:
            print("  (Filtered to process/subject nodes only)")

        print("\n" + "=" * 60)
        print("TWO-LEVEL EVALUATION MODE")
        print("=" * 60)

        # Extract two-level labels
        val_attack_idx = two_level_labels["val"]["attack"]
        val_contaminated_idx = two_level_labels["val"]["contaminated"]
        test_attack_idx = two_level_labels["test"]["attack"]
        test_contaminated_idx = two_level_labels["test"]["contaminated"]

        # If filtering to process nodes only, remap indices
        if filter_process_only:
            # Create mapping from original indices to filtered indices
            def remap_indices(indices, masks):
                """Remap global indices to filtered process-only indices."""
                # Create a cumulative mask across all graphs
                cumulative_mask = np.concatenate(masks)
                # Get mapping: original_idx -> new_idx (only for process nodes)
                original_to_new = {}
                new_idx = 0
                for orig_idx, is_process in enumerate(cumulative_mask):
                    if is_process:
                        original_to_new[orig_idx] = new_idx
                        new_idx += 1

                # Remap the indices, keeping only those that are process nodes
                remapped = []
                for idx in indices:
                    if idx in original_to_new:
                        remapped.append(original_to_new[idx])
                return remapped

            val_attack_idx = remap_indices(val_attack_idx, val_process_mask)
            val_contaminated_idx = remap_indices(val_contaminated_idx, val_process_mask)
            test_attack_idx = remap_indices(test_attack_idx, test_process_mask)
            test_contaminated_idx = remap_indices(
                test_contaminated_idx, test_process_mask
            )

            print("\nFiltered ground truth to process nodes:")
            print(
                f"  Val attack: {len(two_level_labels['val']['attack'])} -> {len(val_attack_idx)}"
            )
            print(
                f"  Val contaminated: {len(two_level_labels['val']['contaminated'])} -> {len(val_contaminated_idx)}"
            )
            print(
                f"  Test attack: {len(two_level_labels['test']['attack'])} -> {len(test_attack_idx)}"
            )
            print(
                f"  Test contaminated: {len(two_level_labels['test']['contaminated'])} -> {len(test_contaminated_idx)}"
            )

        # Extract attack chain mappings (for ADP metric)
        # JSON stores keys as strings, need to convert back to int
        # For Level 1 (Causal Scope): use all malicious nodes (attack + contaminated)
        val_node_to_attack_chain_all = two_level_labels["val"].get(
            "node_to_attack_chain_all", {}
        )
        if not val_node_to_attack_chain_all:
            # Backward compatibility: fall back to old field
            val_node_to_attack_chain_all = two_level_labels["val"].get(
                "node_to_attack_chain", {}
            )
        if val_node_to_attack_chain_all:
            val_node_to_attack_chain_all = {
                int(k): v for k, v in val_node_to_attack_chain_all.items()
            }

        # For Level 2 (Attack Chain Scope): use only attack nodes
        val_node_to_attack_chain_attack_only = two_level_labels["val"].get(
            "node_to_attack_chain_attack_only", {}
        )
        if val_node_to_attack_chain_attack_only:
            val_node_to_attack_chain_attack_only = {
                int(k): v for k, v in val_node_to_attack_chain_attack_only.items()
            }
        else:
            # Backward compatibility: filter the all mapping to only attack nodes
            val_node_to_attack_chain_attack_only = {
                k: v
                for k, v in val_node_to_attack_chain_all.items()
                if k in val_attack_idx
            }

        test_node_to_attack_chain_all = two_level_labels["test"].get(
            "node_to_attack_chain_all", {}
        )
        if not test_node_to_attack_chain_all:
            # Backward compatibility: fall back to old field
            test_node_to_attack_chain_all = two_level_labels["test"].get(
                "node_to_attack_chain", {}
            )
        if test_node_to_attack_chain_all:
            test_node_to_attack_chain_all = {
                int(k): v for k, v in test_node_to_attack_chain_all.items()
            }

        test_node_to_attack_chain_attack_only = two_level_labels["test"].get(
            "node_to_attack_chain_attack_only", {}
        )
        if test_node_to_attack_chain_attack_only:
            test_node_to_attack_chain_attack_only = {
                int(k): v for k, v in test_node_to_attack_chain_attack_only.items()
            }
        else:
            # Backward compatibility: filter the all mapping to only attack nodes
            test_node_to_attack_chain_attack_only = {
                k: v
                for k, v in test_node_to_attack_chain_all.items()
                if k in test_attack_idx
            }

        # Remap attack chain indices if filtering to process nodes
        if filter_process_only:
            # For validation
            if val_node_to_attack_chain_all or val_node_to_attack_chain_attack_only:
                val_cumulative_mask = np.concatenate(val_process_mask)
                val_original_to_new = {}
                new_idx = 0
                for orig_idx, is_process in enumerate(val_cumulative_mask):
                    if is_process:
                        val_original_to_new[orig_idx] = new_idx
                        new_idx += 1

                val_node_to_attack_chain_all = {
                    val_original_to_new[k]: v
                    for k, v in val_node_to_attack_chain_all.items()
                    if k in val_original_to_new
                }
                val_node_to_attack_chain_attack_only = {
                    val_original_to_new[k]: v
                    for k, v in val_node_to_attack_chain_attack_only.items()
                    if k in val_original_to_new
                }

            # For test
            if test_node_to_attack_chain_all or test_node_to_attack_chain_attack_only:
                test_cumulative_mask = np.concatenate(test_process_mask)
                test_original_to_new = {}
                new_idx = 0
                for orig_idx, is_process in enumerate(test_cumulative_mask):
                    if is_process:
                        test_original_to_new[orig_idx] = new_idx
                        new_idx += 1

                test_node_to_attack_chain_all = {
                    test_original_to_new[k]: v
                    for k, v in test_node_to_attack_chain_all.items()
                    if k in test_original_to_new
                }
                test_node_to_attack_chain_attack_only = {
                    test_original_to_new[k]: v
                    for k, v in test_node_to_attack_chain_attack_only.items()
                    if k in test_original_to_new
                }

        print(
            f"  Validation - Attack: {len(val_attack_idx)}, Contaminated: {len(val_contaminated_idx)}"
        )
        print(
            f"  Test - Attack: {len(test_attack_idx)}, Contaminated: {len(test_contaminated_idx)}"
        )
        if test_node_to_attack_chain_attack_only or test_node_to_attack_chain_all:
            unique_chains_attack_only = set(
                test_node_to_attack_chain_attack_only.values()
            )
            unique_chains_all = set(test_node_to_attack_chain_all.values())
            print(
                f"  Test - Attack chains: {len(unique_chains_attack_only)} (attack-only for L2), {len(unique_chains_all)} (all for L1)"
            )

        # Extract idx_to_uuid mappings for entity-level aggregation
        val_idx_to_uuid = two_level_labels["val"].get("idx_to_uuid", {})
        test_idx_to_uuid = two_level_labels["test"].get("idx_to_uuid", {})

        # Convert string keys to int if needed (JSON stores keys as strings)
        if val_idx_to_uuid:
            val_idx_to_uuid = {int(k): v for k, v in val_idx_to_uuid.items()}
        if test_idx_to_uuid:
            test_idx_to_uuid = {int(k): v for k, v in test_idx_to_uuid.items()}

        # Remap idx_to_uuid if filtering to process nodes
        if filter_process_only and (val_idx_to_uuid or test_idx_to_uuid):
            if val_idx_to_uuid and val_process_mask:
                val_cumulative_mask = np.concatenate(val_process_mask)
                val_remapped_uuid = {}
                new_idx = 0
                for orig_idx, is_process in enumerate(val_cumulative_mask):
                    if is_process and orig_idx in val_idx_to_uuid:
                        val_remapped_uuid[new_idx] = val_idx_to_uuid[orig_idx]
                        new_idx += 1
                    elif is_process:
                        new_idx += 1
                val_idx_to_uuid = val_remapped_uuid

            if test_idx_to_uuid and test_process_mask:
                test_cumulative_mask = np.concatenate(test_process_mask)
                test_remapped_uuid = {}
                new_idx = 0
                for orig_idx, is_process in enumerate(test_cumulative_mask):
                    if is_process and orig_idx in test_idx_to_uuid:
                        test_remapped_uuid[new_idx] = test_idx_to_uuid[orig_idx]
                        new_idx += 1
                    elif is_process:
                        new_idx += 1
                test_idx_to_uuid = test_remapped_uuid

        print(
            f"  UUID mappings: val={len(val_idx_to_uuid)}, test={len(test_idx_to_uuid)}"
        )

        # Log dataset info to wandb
        if WANDB_AVAILABLE and wandb.run is not None:
            wandb.log(
                {
                    "n_train_nodes": n_train_nodes,
                    "n_val_nodes": n_val_nodes,
                    "n_test_nodes": n_test_nodes,
                    "val_attack": len(val_attack_idx),
                    "val_contaminated": len(val_contaminated_idx),
                    "test_attack": len(test_attack_idx),
                    "test_contaminated": len(test_contaminated_idx),
                }
            )

        # Run two-level evaluation
        excluded_uuids = _load_excluded_uuids(dataset_name)
        results = two_level_evaluation(
            dataset_name,
            x_train,
            x_val,
            x_test,
            val_attack_idx,
            val_contaminated_idx,
            test_attack_idx,
            test_contaminated_idx,
            n_val_nodes,
            n_test_nodes,
            # Separate attack chain mappings for proper ADP computation
            val_node_to_attack_chain_attack_only=val_node_to_attack_chain_attack_only,
            val_node_to_attack_chain_all=val_node_to_attack_chain_all,
            test_node_to_attack_chain_attack_only=test_node_to_attack_chain_attack_only,
            test_node_to_attack_chain_all=test_node_to_attack_chain_all,
            # UUID mappings for entity-level aggregation
            val_idx_to_uuid=val_idx_to_uuid,
            test_idx_to_uuid=test_idx_to_uuid,
            excluded_uuids=excluded_uuids,
        )

        # Log key metrics to wandb
        if WANDB_AVAILABLE and wandb.run is not None:
            # Level 1 (Causal Scope)
            l1 = results["level1_causal"]
            # Level 2 (Attack Chain Scope)
            l2 = results["level2_strict"]

            wandb.log(
                {
                    # Level 1 metrics
                    "test_f1": l1["test_f1"],
                    "test_prauc": l1["test_prauc"],
                    "test_fpr": l1["test_fpr"],
                    "test_mcc": l1["test_mcc"],
                    "test_adp": l1.get("test_adp", 0.0),
                    # Level 2 metrics (strict)
                    "strict_test_f1": l2["test_f1"],
                    "strict_test_prauc": l2["test_prauc"],
                    "strict_test_fpr": l2["test_fpr"],
                    "strict_test_mcc": l2["test_mcc"],
                    "strict_test_adp": l2.get("test_adp", 0.0),
                }
            )
            wandb.finish()

    return


if __name__ == "__main__":
    args = build_args()
    main(args)
