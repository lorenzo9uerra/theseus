import argparse

from utils.config import Config


def str_to_bool(value):
    """Convert string to boolean for argparse."""
    if isinstance(value, bool):
        return value
    if value.lower() in ("true", "1", "yes", "on"):
        return True
    elif value.lower() in ("false", "0", "no", "off"):
        return False
    else:
        raise argparse.ArgumentTypeError(f"Boolean value expected, got '{value}'")


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset",
        type=str,
        choices=["CADETS_E3", "THEIA_E3", "FIVEDIRECTIONS_E3", "TRACE_E3"],
        help="Name of the dataset",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--wandb", action="store_true", help="Whether to submit logs to wandb"
    )
    parser.add_argument(
        "--project", type=str, default="theseus", help="Name of the wandb project"
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Whether to run the framework on CPU rather than GPU",
    )
    parser.add_argument(
        "--cache_dir", default="./cache/", help="Directory to store cached data"
    )
    parser.add_argument(
        "--outputs_dir", default="./outputs", help="Base directory for storing outputs"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./checkpoints",
        help="Directory to store model checkpoints",
    )
    parser.add_argument(
        "--force_restart",
        action="store_true",
        help="Force restart of the dataset and the models, even if they exist",
    )
    parser.add_argument(
        "--test", action="store_true", help="If set, run evaluation only"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint for testing or resuming training",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to custom configuration file (overrides default configs/models/<model>.yml)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data/DARPA",
        help="Directory where raw data files are stored",
    )

    # Hyperparameters
    parser.add_argument(
        "--sage_lr", type=float, default=None, help="Learning rate for SAGE encoder"
    )
    parser.add_argument(
        "--transformer_lr",
        type=float,
        default=None,
        help="Learning rate for Transformer autoencoder",
    )
    parser.add_argument(
        "--sage_weight_decay",
        type=float,
        default=None,
        help="Weight decay for SAGE encoder",
    )
    parser.add_argument(
        "--transformer_weight_decay",
        type=float,
        default=None,
        help="Weight decay for Transformer autoencoder",
    )
    parser.add_argument(
        "--sage_dropout", type=float, default=None, help="Dropout rate for SAGE"
    )
    parser.add_argument(
        "--transformer_dropout",
        type=float,
        default=None,
        help="Dropout rate for Transformer",
    )
    parser.add_argument(
        "--embed_dim",
        type=int,
        default=None,
        help="Embedding dimension for transformer",
    )
    parser.add_argument(
        "--num_layers", type=int, default=None, help="Number of transformer layers"
    )
    parser.add_argument(
        "--num_heads", type=int, default=None, help="Number of attention heads"
    )
    parser.add_argument(
        "--node_out_dim",
        type=int,
        default=None,
        help="Output dimension for node encoder",
    )
    parser.add_argument(
        "--num_epochs", type=int, default=None, help="Number of training epochs"
    )
    parser.add_argument(
        "--patience", type=int, default=None, help="Early stopping patience"
    )
    parser.add_argument(
        "--nhops", type=int, default=None, help="Number of hops for graph sampling"
    )
    parser.add_argument(
        "--mask_ratio",
        type=float,
        default=0.0,
        help="Ratio of nodes to mask during training",
    )
    parser.add_argument(
        "--window_size_minutes",
        type=int,
        default=None,
        help="Time window size in minutes for temporal batching",
    )
    parser.add_argument(
        "--max_nodes",
        type=int,
        default=None,
        help="Maximum nodes per subgraph for splitting large graphs",
    )
    parser.add_argument(
        "--graph_batch_size",
        type=int,
        default=None,
        help="Number of graphs per batch (for datasets with multiple graphs)",
    )
    parser.add_argument(
        "--tags", type=str, default=None, help="Comma-separated tags for wandb runs"
    )
    parser.add_argument(
        "--use_node_degrees",
        type=str_to_bool,
        default=None,
        help="Whether to use node degrees as additional node features",
    )
    parser.add_argument(
        "--use_fused_edge_count",
        type=str_to_bool,
        default=None,
        help="Whether to use fused edge count as an additional edge feature",
    )
    parser.add_argument(
        "--bidirectional_edges",
        type=str_to_bool,
        default=None,
        help="Whether to treat edges as bidirectional",
    )
    parser.add_argument(
        "--exclude_malicious_from_training",
        type=str_to_bool,
        default=None,
        help="If true, train Theseus only on benign-labeled nodes (y==0) within the training split; if false, train on all nodes in the training split.",
    )
    return parser.parse_args(args)


def parse_config(args):
    # Create config from arguments
    config = Config.from_args(args)

    # Override any dot-notation arguments from command line
    config.override_from_args(args)

    return config
