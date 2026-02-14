import os

import psutil
import torch
from torch_geometric.loader import DataLoader

import wandb
from models.theseus import Theseus
from tasks.evaluate import validate
from utils.utils import log, log_tqdm, timed_execution


def initialize_model(config, data, checkpoint_path=None):
    """Initialize Theseus model from data dimensions, optionally loading from checkpoint."""
    graph = next(iter(data))
    if isinstance(graph, (list, tuple)) and len(graph) > 0:
        graph = graph[0]

    in_dim = graph.x.shape[1]

    edge_dim = graph.edge_attr.shape[1]
    model = Theseus(
        in_dim,
        edge_dim,
        config.node_out_dim,
        sage_dropout=config.sage_dropout,
        transformer_dropout=config.transformer_dropout,
        num_heads=config.num_heads,
        embed_dim=config.embed_dim,
        num_layers=config.num_layers,
        nhops=config.nhops,
        mask_ratio=config.mask_ratio,
        agg_type="mean",
    ).to(config.device)

    if checkpoint_path and os.path.exists(checkpoint_path) and not config.force_restart:
        log(f"Loading model weights from {checkpoint_path}...")
        model.load_checkpoint(checkpoint_path)

    return model


@timed_execution
def train(config, train_data, val_data, test_data):
    if config.device == "cuda":
        torch.cuda.reset_peak_memory_stats("cuda")

    # Setup checkpoint path
    if os.path.dirname(config.checkpoint):
        checkpoint_path = config.checkpoint
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    else:
        theseus_checkpoint_dir = os.path.join(config.checkpoint_dir, "theseus")
        os.makedirs(theseus_checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(theseus_checkpoint_dir, config.checkpoint)

    model = initialize_model(config, train_data)

    param_groups = [
        {
            "params": model.encoder.parameters(),
            "lr": config.sage_lr,
            "weight_decay": config.sage_weight_decay,
        },
        {
            "params": model.autoencoder.parameters(),
            "lr": config.transformer_lr,
            "weight_decay": config.transformer_weight_decay,
        },
    ]
    optimizer = torch.optim.AdamW(param_groups)

    start_epoch = 0
    if os.path.exists(checkpoint_path) and not config.force_restart:
        log(f"Resuming from checkpoint: {checkpoint_path}")
        loaded_epoch, threshold = model.load_checkpoint(checkpoint_path, optimizer)
        start_epoch = loaded_epoch + 1

    num_epochs = config.num_epochs
    best_val_ap = float("-inf")
    patience_counter = 0
    max_patience = config.patience

    train_loader = DataLoader(
        train_data, batch_size=config.graph_batch_size, shuffle=False
    )

    for epoch in (
        pbar := log_tqdm(range(start_epoch, num_epochs), desc="Training Epochs")
    ):
        total_loss = 0.0
        num_batches = 0

        model.train()
        for batch in train_loader:
            benign_mask = batch.y == 0
            if not benign_mask.any():
                continue

            benign_indices = torch.where(benign_mask)[0]
            batch_benign = batch.subgraph(benign_indices).to(device=config.device)

            # Process nodes only (index 0 in NODE_TYPES one-hot encoding)
            process_mask = batch_benign.x[:, 0] == 1

            outputs, encoded_target = model(batch_benign)

            if process_mask.any():
                loss = model.loss(
                    outputs, encoded_target.detach(), "mean", node_mask=process_mask
                )
            else:
                # Fallback: If no processes exist, reconstruct all nodes to maintain
                # gradient flow for environmental embeddings
                loss = model.loss(outputs, encoded_target.detach(), "mean")

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

        model.eval()
        val_ap, val_loss, threshold = validate(model, val_data, config)
        test_ap, test_loss, _ = validate(model, test_data, config)

        if val_ap > best_val_ap:
            best_val_ap = val_ap
            patience_counter = 0

            model.save_checkpoint(
                checkpoint_path, threshold=threshold, optimizer=optimizer, epoch=epoch
            )
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                log(f"Early stopping at epoch {epoch} with best val_ap: {best_val_ap}")
                break

        wandb.log(
            {
                "epoch": epoch,
                "train_loss": avg_loss,
                "val_loss": val_loss,
                "val_ap": val_ap,
                "test_loss": test_loss,
                "test_ap": test_ap,
            }
        )
        pbar.set_postfix({"train_loss": avg_loss, "val_ap": val_ap})

    peak_train_cpu_mem = psutil.Process().memory_info().rss / 1024**3
    peak_train_gpu_mem = (
        torch.cuda.max_memory_allocated(device=config.device) / (1024**3)
        if config.device == "cuda"
        else 0.0
    )
    wandb.log(
        {
            "peak_train_cpu_memory": peak_train_cpu_mem,
            "peak_train_gpu_memory": peak_train_gpu_mem,
        }
    )

    log("Training completed.")

    best_epoch, _ = model.load_checkpoint(checkpoint_path)
    log(f"Best validation PR-AUC: {best_val_ap:.4f} at epoch {best_epoch}")

    return model
