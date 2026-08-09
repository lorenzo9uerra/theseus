import os
import random

import numpy as np
import torch
import wandb

from tasks.build_graphs import build_graphs
from tasks.evaluate import evaluate
from tasks.training import initialize_model, train
from utils.parsers import parse_args, parse_config
from utils.utils import log

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def main(config):
    set_seed(config.seed)

    graphs, ground_truth = build_graphs(config)

    train_data = graphs["train"]
    val_data = graphs["val"]
    test_data = graphs["test"]
    calibration_data = val_data if val_data else train_data
    calibration_split_name = "Validation" if val_data else "Train"

    if not val_data:
        log(
            "No validation split is configured for this dataset. "
            "Using the maximum benign training score for threshold calibration at evaluation time."
        )

    if config.test:
        # Test mode: checkpoint must be specified
        if config.checkpoint is None:
            raise ValueError(
                "config.checkpoint must be specified when config.test is True"
            )
        if not os.path.exists(config.checkpoint):
            raise FileNotFoundError(
                f"Checkpoint not found: {config.checkpoint}. "
                "Please specify a valid checkpoint path with --checkpoint"
            )

        log(f"Test mode: Loading model from {config.checkpoint}")
        model = initialize_model(config, train_data, config.checkpoint)
    else:
        # Training mode: train or resume training
        model = train(config, train_data, val_data, test_data, ground_truth)

    model.eval()
    metrics = evaluate(
        model,
        val_data,
        test_data,
        config,
        ground_truth,
        calibration_data=calibration_data,
        calibration_split_name=calibration_split_name,
    )
    wandb.log(metrics)

    log("==" * 30)
    log("Run finished.")
    log("==" * 30)


if __name__ == "__main__":
    args = parse_args()
    config = parse_config(args)

    embed_dim = config.embed_dim
    num_heads = config.num_heads
    if embed_dim % num_heads != 0:
        raise ValueError(
            f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads}). "
            f"Current configuration would result in {embed_dim}/{num_heads} = {embed_dim / num_heads:.2f} "
            f"dimensions per head, which must be an integer."
        )

    wandb.init(
        mode=("online" if args.wandb else "offline"),
        project=args.project,
        config=config.to_dict(),
        tags=(args.tags.split(",") if args.tags else None),
    )
    # Update config name to match the WandB run name
    if args.wandb and wandb.run is not None:
        config.name = wandb.run.name
    else:
        import secrets

        config.name = f"theseus_{secrets.token_hex(5)}"
    if config.checkpoint is None:
        # Default checkpoint will be placed in ./checkpoints/theseus/
        config.checkpoint = f"checkpoint_{config.name}.pt"

    main(config)

    wandb.finish()
