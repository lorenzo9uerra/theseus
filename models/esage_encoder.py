import torch
import torch.nn as nn
from torch.nn.functional import relu
from torch_geometric.nn import MessagePassing


class ESAGEConv(MessagePassing):
    """GraphSAGE convolution layer with edge feature support."""

    def __init__(self, in_dim, edge_dim, out_dim, aggr="mean"):
        super().__init__(aggr=aggr)

        self.msg_mlp = nn.Linear(in_dim + edge_dim, out_dim)

        self.update_mlp = nn.Linear(in_dim + out_dim, out_dim)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.msg_mlp.weight)
        nn.init.zeros_(self.msg_mlp.bias)
        nn.init.xavier_uniform_(self.update_mlp.weight)
        nn.init.zeros_(self.update_mlp.bias)

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)  # ty:ignore[missing-argument]

    def message(self, x_j, edge_attr):  # ty:ignore[invalid-method-override]
        """Create messages from source node + edge features."""
        msg_input = torch.cat([x_j, edge_attr], dim=-1)
        return self.msg_mlp(msg_input)

    def update(self, aggr_out, x):  # ty:ignore[invalid-method-override]
        """Update node features from self + aggregated messages."""
        update_input = torch.cat([x, aggr_out], dim=-1)
        return self.update_mlp(update_input)


class EGraphSAGE(nn.Module):
    """Multi-hop GraphSAGE encoder with edge feature support."""

    def __init__(
        self, in_dim, edge_dim, out_dim, nhops=1, dropout=0.0, agg_type="mean"
    ):
        super().__init__()

        self.in_dim: int = in_dim
        self.edge_dim: int = edge_dim
        self.out_dim: int = out_dim
        self.nhops: int = nhops
        self.dropout: float = dropout

        self.layers = nn.ModuleList()

        if nhops == 1:
            self.layers.append(ESAGEConv(in_dim, edge_dim, out_dim, aggr=agg_type))
        else:
            # First layer
            self.layers.append(ESAGEConv(in_dim, edge_dim, out_dim, aggr=agg_type))

            # Hidden layers
            for _ in range(nhops - 2):
                self.layers.append(ESAGEConv(out_dim, edge_dim, out_dim, aggr=agg_type))

            # Last layer
            self.layers.append(ESAGEConv(out_dim, edge_dim, out_dim, aggr=agg_type))

        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr):
        """Forward pass: apply nhops of message passing with edge features."""
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index, edge_attr)

            if i < len(self.layers) - 1:
                x = relu(x)
                x = self.dropout_layer(x)

        return x
