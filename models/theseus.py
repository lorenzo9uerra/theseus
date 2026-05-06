import torch
import torch.nn as nn
from torch.nn.functional import mse_loss

from models.esage_encoder import EGraphSAGE
from models.transformer_autoencoder import TransformerAutoencoder


class Theseus(nn.Module):
    def __init__(
        self,
        in_dim,
        edge_dim,
        node_out_dim,
        sage_dropout=0.0,
        transformer_dropout=0.0,
        num_heads=4,
        embed_dim=40,
        num_layers=1,
        nhops=1,
        mask_ratio=0.0,
        agg_type="mean",
    ):
        super().__init__()
        self.encoder = EGraphSAGE(
            in_dim, edge_dim, node_out_dim, nhops, sage_dropout, agg_type=agg_type
        )
        self.autoencoder = TransformerAutoencoder(
            node_out_dim,
            embed_dim,
            num_heads,
            num_layers,
            transformer_dropout,
            mask_ratio,
            output_dim=node_out_dim,  # Reconstruct in representation space
        )

    def forward(self, batch):
        x_encoded = self.encoder(batch.x, batch.edge_index, batch.edge_attr)
        x_reconstructed = self.autoencoder(x_encoded)
        return x_reconstructed, x_encoded

    def loss(self, output, encoded_target, reduction="none", node_mask=None):
        if node_mask is not None:
            encoded_target = encoded_target[node_mask]
            output = output[node_mask]

        return mse_loss(output, encoded_target, reduction=reduction)

    def save_checkpoint(self, path, threshold, optimizer=None, epoch=0):
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "epoch": int(epoch),
            "threshold": float(threshold),
        }
        if optimizer:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()
        torch.save(checkpoint, path)

    def load_checkpoint(self, path, optimizer=None):
        device = next(self.parameters()).device
        try:
            checkpoint = torch.load(path, map_location=device, weights_only=True)
        except TypeError:
            checkpoint = torch.load(path, map_location=device)
        self.load_state_dict(checkpoint["model_state_dict"])
        if optimizer and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return checkpoint["epoch"], checkpoint["threshold"]
