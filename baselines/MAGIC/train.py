import os
import random
import warnings

import numpy as np
import torch
from model.autoencoder import build_model
from sklearn.metrics import auc as compute_auc
from sklearn.metrics import precision_recall_curve
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
from utils.loaddata import load_entity_level_dataset, load_metadata

from utils.config import build_args
from utils.utils import create_optimizer, set_random_seed

warnings.filterwarnings("ignore")

_MAGIC_ROOT = os.path.dirname(os.path.abspath(__file__))


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
    return torch.device(f"cuda:{device_arg}")


def main(main_args):
    device = _resolve_device(main_args.device)
    dataset_name = main_args.dataset
    main_args.num_hidden = 64
    main_args.max_epoch = 200
    main_args.num_layers = 3
    set_random_seed(main_args.seed)

    os.makedirs(os.path.join(_MAGIC_ROOT, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(_MAGIC_ROOT, "eval_result"), exist_ok=True)

    metadata = load_metadata(dataset_name)
    main_args.n_dim = metadata["node_feature_dim"]
    main_args.e_dim = metadata["edge_feature_dim"]
    n_train_graphs = metadata["n_train"]
    n_val_graphs = metadata["n_val"]
    n_test_graphs = metadata["n_test"]

    # Determine the node type ID for SUBJECT_PROCESS (for process-only filtering)
    node_type_dict = metadata.get("node_type_dict", {})
    process_type_id = None
    filter_process_only = False

    if not node_type_dict:
        print(
            "Warning: node_type_dict not found in metadata. Training validation on all nodes."
        )
        print("  To enable process-only filtering, re-run the data parser.")
    else:
        for type_name, type_id in node_type_dict.items():
            if "SUBJECT" in type_name or "PROCESS" in type_name:
                process_type_id = type_id
                print(f"Found process type: {type_name} with ID {type_id}")
                break

        if process_type_id is None:
            print(
                "Warning: Could not identify process node type. Training validation on all nodes."
            )
        else:
            filter_process_only = True
            print(
                f"Filtering training validation to process nodes only (type ID: {process_type_id})"
            )

    model = build_model(main_args)
    model = model.to(device)
    model.train()
    optimizer = create_optimizer(
        main_args.optimizer, model, main_args.lr, main_args.weight_decay
    )

    # Count validation nodes and build process masks
    val_node_offset = 0
    val_process_masks = []
    for i in range(n_val_graphs):
        g = load_entity_level_dataset(dataset_name, "val", i).to(device)
        if filter_process_only:
            node_types = g.ndata["type"].cpu().numpy()
            mask = node_types == process_type_id
            val_process_masks.append(mask)
            val_node_offset += mask.sum()
        else:
            val_node_offset += g.number_of_nodes()
        del g

    # Create validation labels from two_level_labels
    y_val = np.zeros(val_node_offset)
    two_level_labels = metadata.get("two_level_labels", None)

    if not two_level_labels or "val" not in two_level_labels:
        raise ValueError(
            f"Dataset {dataset_name} is missing 'two_level_labels' in metadata.json. "
            "Please re-run the data parser (utils/csv_parser_daily.py) to generate proper labels."
        )

    # Use two-level labels (attack + contaminated = positives for detection)
    val_attack_idx = two_level_labels["val"].get("attack", [])
    val_contaminated_idx = two_level_labels["val"].get("contaminated", [])

    # Remap indices if filtering to process nodes only
    if filter_process_only and val_process_masks:
        cumulative_mask = np.concatenate(val_process_masks)
        original_to_new = {}
        new_idx = 0
        for orig_idx, is_process in enumerate(cumulative_mask):
            if is_process:
                original_to_new[orig_idx] = new_idx
                new_idx += 1

        # Remap and filter indices
        remapped_attack = [
            original_to_new[idx] for idx in val_attack_idx if idx in original_to_new
        ]
        remapped_contaminated = [
            original_to_new[idx]
            for idx in val_contaminated_idx
            if idx in original_to_new
        ]

        for idx in remapped_attack + remapped_contaminated:
            if idx < val_node_offset:
                y_val[idx] = 1.0

        print(
            f"  Validation labels (process-only): {len(remapped_attack)} attack, {len(remapped_contaminated)} contaminated"
        )
    else:
        for idx in val_attack_idx + val_contaminated_idx:
            if idx < val_node_offset:
                y_val[idx] = 1.0
        print(
            f"  Validation labels: {len(val_attack_idx)} attack, {len(val_contaminated_idx)} contaminated"
        )

    n_val_attack = int(y_val.sum())
    n_val_benign = val_node_offset - n_val_attack

    print("\nTraining/Validation Setup:")
    print(f"  Training graphs: {n_train_graphs}")
    print(
        f"  Validation nodes: {val_node_offset} ({n_val_benign} benign, {n_val_attack} attack)"
    )
    print(f"  Test graphs: {n_test_graphs}")

    # Setup KNN parameters
    n_neighbors = 200

    # Early stopping setup based on validation PR-AUC
    best_val_prauc = 0.0
    patience_counter = 0
    best_model_path = os.path.join(
        _MAGIC_ROOT,
        "checkpoints",
        f"checkpoint-{dataset_name}-seed{main_args.seed}-best.pt",
    )

    epoch_iter = tqdm(range(main_args.max_epoch))
    for epoch in epoch_iter:
        # Training phase
        model.train()
        train_loss = 0.0
        for i in range(n_train_graphs):
            g = load_entity_level_dataset(dataset_name, "train", i).to(device)
            loss = model(g)
            loss /= n_train_graphs
            optimizer.zero_grad()
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            del g

        # Validation phase - compute PR-AUC on labeled validation set
        model.eval()
        with torch.no_grad():
            # Get training embeddings for KNN baseline (filter to process nodes if enabled)
            x_train_list = []
            for i in range(n_train_graphs):
                g = load_entity_level_dataset(dataset_name, "train", i).to(device)
                embeddings = model.embed(g).cpu().numpy()
                if filter_process_only:
                    node_types = g.ndata["type"].cpu().numpy()
                    mask = node_types == process_type_id
                    embeddings = embeddings[mask]
                x_train_list.append(embeddings)
                del g
            x_train = np.concatenate(x_train_list, axis=0)

            # Get validation embeddings (filter to process nodes if enabled)
            x_val_list = []
            for i in range(n_val_graphs):
                g = load_entity_level_dataset(dataset_name, "val", i).to(device)
                embeddings = model.embed(g).cpu().numpy()
                if filter_process_only:
                    node_types = g.ndata["type"].cpu().numpy()
                    mask = node_types == process_type_id
                    embeddings = embeddings[mask]
                x_val_list.append(embeddings)
                del g
            x_val = np.concatenate(x_val_list, axis=0)

            # Normalize
            x_train_mean, x_train_std = x_train.mean(axis=0), x_train.std(axis=0)
            x_train_norm = (x_train - x_train_mean) / (x_train_std + 1e-9)
            x_val_norm = (x_val - x_train_mean) / (x_train_std + 1e-9)

            # For large datasets, subsample training data for KNN
            max_knn_train_samples = 100000
            if len(x_train_norm) > max_knn_train_samples:
                idx_subsample = np.random.choice(
                    len(x_train_norm), max_knn_train_samples, replace=False
                )
                x_train_knn = x_train_norm[idx_subsample]
            else:
                x_train_knn = x_train_norm

            # Fit KNN and compute anomaly scores
            nbrs = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=-1)
            nbrs.fit(x_train_knn)

            # Get baseline distance from training
            idx_sample = list(range(len(x_train_knn)))
            random.shuffle(idx_sample)
            train_dist, _ = nbrs.kneighbors(
                x_train_knn[idx_sample][: min(10000, len(x_train_knn))],
                n_neighbors=n_neighbors,
            )
            mean_distance = train_dist.mean()

            # Validation anomaly scores (batched for large datasets)
            batch_size = 100000
            if len(x_val_norm) <= batch_size:
                val_dist, _ = nbrs.kneighbors(x_val_norm, n_neighbors=n_neighbors)
                val_scores = val_dist.mean(axis=1) / mean_distance
            else:
                val_scores = np.zeros(len(x_val_norm))
                for start in range(0, len(x_val_norm), batch_size):
                    end = min(start + batch_size, len(x_val_norm))
                    batch_dist, _ = nbrs.kneighbors(
                        x_val_norm[start:end], n_neighbors=n_neighbors
                    )
                    val_scores[start:end] = batch_dist.mean(axis=1) / mean_distance

            # Compute PR-AUC on validation set
            prec_val, rec_val, _ = precision_recall_curve(y_val, val_scores)
            val_prauc = compute_auc(rec_val, prec_val)

        epoch_iter.set_description(
            f"Epoch {epoch} | train_loss: {train_loss:.4f} | val_PR-AUC: {val_prauc:.4f} | "
            f"best: {best_val_prauc:.4f} | patience: {patience_counter}/{main_args.patience}"
        )

        # Check for improvement (higher PR-AUC is better)
        if val_prauc > best_val_prauc:
            best_val_prauc = val_prauc
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= main_args.patience:
            print(f"\nEarly stopping triggered at epoch {epoch}")
            print(f"Best validation PR-AUC: {best_val_prauc:.4f}")
            break

    # Load best model and save as final checkpoint
    if os.path.exists(best_model_path):
        print(f"\nLoading best model with val_PR-AUC={best_val_prauc:.4f}")
        model.load_state_dict(torch.load(best_model_path))
        # Save as final checkpoint
        final_ckpt = os.path.join(
            _MAGIC_ROOT,
            "checkpoints",
            f"checkpoint-{dataset_name}-seed{main_args.seed}.pt",
        )
        torch.save(model.state_dict(), final_ckpt)
        print(f"Saved best model to {final_ckpt}")

    save_dict_path = os.path.join(
        _MAGIC_ROOT, "eval_result", f"distance_save_{dataset_name}.pkl"
    )
    if os.path.exists(save_dict_path):
        os.unlink(save_dict_path)
    return


if __name__ == "__main__":
    args = build_args()
    main(args)
