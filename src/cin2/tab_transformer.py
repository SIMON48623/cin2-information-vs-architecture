"""Article Tabular Transformer architecture.

This is the model used by the original ``11_tab_transformer_cv.py`` run:
numeric feature tokens, per-column categorical embeddings, a learned [CLS]
token, learned positional embeddings, and a two-block Transformer encoder.
"""
from __future__ import annotations

import torch
from torch import nn


class FTTransformer(nn.Module):
    """The article's custom Tabular Transformer (not the rtdl model)."""

    def __init__(self, n_num: int, cat_cardinalities: list[int], d_token: int = 64,
                 n_head: int = 4, n_layers: int = 2, d_ff: int = 128,
                 dropout: float = 0.15):
        super().__init__()
        self.n_num = n_num
        self.n_cat = len(cat_cardinalities)
        self.d_token = d_token
        self.num_linears = nn.ModuleList([nn.Linear(1, d_token) for _ in range(n_num)])
        self.cat_embeds = nn.ModuleList([
            nn.Embedding(int(card) + 1, d_token) for card in cat_cardinalities
        ])
        self.cls = nn.Parameter(torch.zeros(1, 1, d_token))
        self.pos_embed = nn.Embedding(1 + n_num + self.n_cat, d_token)
        layer = nn.TransformerEncoderLayer(
            d_model=d_token,
            nhead=n_head,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=False,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Sequential(nn.LayerNorm(d_token), nn.Dropout(dropout), nn.Linear(d_token, 1))
        nn.init.normal_(self.cls, mean=0.0, std=0.02)

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        batch = x_num.size(0)
        cls_token = self.cls.expand(batch, -1, -1)
        num_tokens = [self.num_linears[i](x_num[:, i:i + 1]).unsqueeze(1)
                      for i in range(self.n_num)]
        num_tokens = (torch.cat(num_tokens, dim=1) if num_tokens else
                      torch.empty(batch, 0, self.d_token, device=x_num.device))
        cat_tokens = [embedding(x_cat[:, j]).unsqueeze(1)
                      for j, embedding in enumerate(self.cat_embeds)]
        cat_tokens = (torch.cat(cat_tokens, dim=1) if cat_tokens else
                      torch.empty(batch, 0, self.d_token, device=x_num.device))
        sequence = torch.cat([cls_token, num_tokens, cat_tokens], dim=1)
        positions = torch.arange(sequence.size(1), device=sequence.device).unsqueeze(0).repeat(batch, 1)
        encoded = self.encoder(sequence + self.pos_embed(positions))
        return self.head(encoded[:, 0, :]).squeeze(-1)
