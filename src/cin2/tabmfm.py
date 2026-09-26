"""Article Tab-MFM architecture and masking operation.

This preserves the model used by ``14_tab_mfm_pretrain_and_finetune_cv.py``:
numeric and categorical tokens, lookup embeddings, optional column-identity
embeddings, masked reconstruction, mean pooling, LayerNorm, and dropout.
"""
from __future__ import annotations

import torch
from torch import nn


class TabTokTransformer(nn.Module):
    def __init__(self, n_num: int, cat_cards: list[int], d_model: int = 64,
                 n_head: int = 4, n_layers: int = 2, d_ff: int = 128,
                 dropout: float = 0.15, use_col_id_emb: bool = True):
        super().__init__()
        self.n_num = n_num
        self.n_cat = len(cat_cards)
        self.n_features = n_num + self.n_cat
        self.d_model = d_model
        self.num_value_proj = nn.Linear(1, d_model)
        self.cat_embs = nn.ModuleList([nn.Embedding(card, d_model) for card in cat_cards])
        self.col_id_emb = nn.Embedding(self.n_features, d_model) if use_col_id_emb else None
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.cls_head = nn.Sequential(nn.LayerNorm(d_model), nn.Dropout(dropout), nn.Linear(d_model, 1))
        self.num_recon = nn.Linear(d_model, 1)
        self.cat_recon = nn.ModuleList([nn.Linear(d_model, card) for card in cat_cards])

    def forward_tokens(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        batch = x_num.size(0)
        num_tokens = self.num_value_proj(x_num.unsqueeze(-1))
        cat_tokens = [embedding(x_cat[:, j]) for j, embedding in enumerate(self.cat_embs)]
        cat_tokens = (torch.stack(cat_tokens, dim=1) if cat_tokens else
                      torch.empty(batch, 0, self.d_model, device=x_num.device))
        tokens = torch.cat([num_tokens, cat_tokens], dim=1) if self.n_cat else num_tokens
        if self.col_id_emb is not None:
            feature_ids = torch.arange(self.n_features, device=tokens.device).unsqueeze(0).repeat(batch, 1)
            tokens = tokens + self.col_id_emb(feature_ids)
        return tokens

    def encode(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        return self.encoder(self.forward_tokens(x_num, x_cat))

    def classify(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        return self.cls_head(self.encode(x_num, x_cat).mean(dim=1)).squeeze(-1)

    def recon_num(self, hidden_num: torch.Tensor) -> torch.Tensor:
        return self.num_recon(hidden_num).squeeze(-1)

    def recon_cat(self, hidden_cat: torch.Tensor) -> list[torch.Tensor]:
        return [head(hidden_cat[:, j, :]) for j, head in enumerate(self.cat_recon)]


def apply_mfm_mask(x_num: torch.Tensor, x_cat: torch.Tensor, cat_cards: list[int],
                   mask_ratio: float = 0.30):
    """Mask numeric values with zero and categorical values with their mask id."""
    batch, n_num = x_num.shape
    n_cat = x_cat.shape[1]
    mask = torch.rand(batch, n_num + n_cat, device=x_num.device) < float(mask_ratio)
    x_num_masked = x_num.clone()
    x_cat_masked = x_cat.clone()
    if n_num:
        x_num_masked[mask[:, :n_num]] = 0.0
    for j in range(n_cat):
        x_cat_masked[mask[:, n_num + j], j] = cat_cards[j] - 1
    return x_num_masked, x_cat_masked, mask
