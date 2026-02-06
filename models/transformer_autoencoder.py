import torch
import torch.nn as nn


class TransformerAutoencoder(nn.Module):
    def __init__(
        self,
        input_dim,
        embed_dim,
        num_heads,
        num_layers,
        dropout,
        mask_ratio=0.0,
        output_dim=None,
    ):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.input_projection = nn.Linear(input_dim, embed_dim)
        self.reconstruction_dim = output_dim if output_dim is not None else input_dim
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, nhead=num_heads, dropout=dropout, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)
        self.output_projection = nn.Linear(embed_dim, self.reconstruction_dim)
        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.xavier_uniform_(self.input_projection.weight)
        nn.init.zeros_(self.input_projection.bias)
        nn.init.xavier_uniform_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

        for name, param in self.encoder.named_parameters():
            if "weight" in name and param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

        for name, param in self.decoder.named_parameters():
            if "weight" in name and param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(self, src, padding_mask=None):
        src = self.input_projection(src)

        squeeze_output = False
        if src.dim() == 2:
            src = src.unsqueeze(0)
            squeeze_output = True

        if padding_mask is not None:
            if padding_mask.dim() == 2:
                padding_mask = padding_mask.unsqueeze(0)
            padding_mask = ~torch.any(padding_mask, dim=-1)

        if self.training and self.mask_ratio > 0:
            seq_len = src.size(1)
            mask = torch.triu(
                torch.ones(seq_len, seq_len, device=src.device), diagonal=1
            )
            mask = mask * (
                torch.rand(seq_len, seq_len, device=src.device) < self.mask_ratio
            )
            attention_mask = (mask + mask.T).bool()  # make it symmetric
        else:
            attention_mask = None

        memory = self.encoder(
            src, mask=attention_mask, src_key_padding_mask=padding_mask
        )

        output = self.decoder(
            src,
            memory,
            memory_key_padding_mask=padding_mask,
            tgt_mask=attention_mask,
            tgt_key_padding_mask=padding_mask,
        )

        output = self.output_projection(output)
        if squeeze_output:
            output = output.squeeze(0)
        return output
