from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialSelfAttention(nn.Module):
    """A lightweight spatial attention block over agents (N dimension)."""
    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model),
        )
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None):
        # x: [B, N, D]
        h, _ = self.mha(x, x, x, key_padding_mask=key_padding_mask, need_weights=False)
        x = self.ln1(x + h)
        x = self.ln2(x + self.ff(x))
        return x

class TemporalSelfAttention(nn.Module):
    """Causal temporal self-attention per agent (T dimension)."""
    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model),
        )
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, causal: bool = True, key_padding_mask: Optional[torch.Tensor] = None):
        # x: [B, T, D]
        attn_mask = None
        if causal:
            T = x.size(1)
            attn_mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        h, _ = self.mha(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False)
        x = self.ln1(x + h)
        x = self.ln2(x + self.ff(x))
        return x

class STEncoder(nn.Module):
    """Spatio-temporal encoder.

    Two modes:
      - spatial-only: input [B,N,D] -> output [B,N,D]
      - spatio-temporal: input [B,T,N,D] -> output [B,T,N,D]
    The default mode is spatial-only to keep integration minimal.
    """
    def __init__(self, d_model: int, n_heads_s: int = 4, n_heads_t: int = 4, dropout: float = 0.0, use_temporal: bool = False):
        super().__init__()
        self.use_temporal = use_temporal
        self.spatial = SpatialSelfAttention(d_model, n_heads=n_heads_s, dropout=dropout)
        if use_temporal:
            self.temporal = TemporalSelfAttention(d_model, n_heads=n_heads_t, dropout=dropout)
        self.fuse = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU())
        self.credit_head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor, agent_key_padding_mask: Optional[torch.Tensor] = None):
        """Forward.

        If self.use_temporal == False:
          x: [B,N,D]
          returns z:[B,N,D], credit_logits:[B,N,1]
        If True:
          x: [B,T,N,D]
          returns z:[B,T,N,D], credit_logits:[B,T,N,1]
        """
        if not self.use_temporal:
            z = self.spatial(x, key_padding_mask=agent_key_padding_mask)
            z = self.fuse(z)
            credit_logits = self.credit_head(z)
            return z, credit_logits

        # temporal + spatial
        B, T, N, D = x.shape
        # temporal per agent: [B*N,T,D]
        xt = x.permute(0, 2, 1, 3).contiguous().view(B * N, T, D)
        xt = self.temporal(xt, causal=True)
        xt = xt.view(B, N, T, D).permute(0, 2, 1, 3).contiguous()  # [B,T,N,D]
        # spatial per timestep: [B*T,N,D]
        xs = xt.view(B * T, N, D)
        # expand agent mask: [B*T,N]
        kpm = None
        if agent_key_padding_mask is not None:
            kpm = agent_key_padding_mask.unsqueeze(1).repeat(1, T, 1).contiguous().view(B * T, N)
        z = self.spatial(xs, key_padding_mask=kpm)
        z = self.fuse(z)
        credit_logits = self.credit_head(z)
        return z.view(B, T, N, D), credit_logits.view(B, T, N, 1)
