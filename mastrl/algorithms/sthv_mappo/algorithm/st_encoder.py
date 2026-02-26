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
        # x: [T*B, N, D]
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
        # x: [T, B*N, D]
        attn_mask = None
        if causal:
            T = x.size(0)
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
    def __init__(self, d_model: int, n_heads_s: int = 4, n_heads_t: int = 4, dropout: float = 0.0):
        super().__init__()
        self.spatial = SpatialSelfAttention(d_model, n_heads=n_heads_s, dropout=dropout)
        self.temporal = TemporalSelfAttention(d_model, n_heads=n_heads_t, dropout=dropout)
        self.fuse = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU())
        self.credit_head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor, agent_key_padding_mask: Optional[torch.Tensor] = None):
        """Forward.

        (废弃只用空间，空间时间都使用)
        If self.use_temporal == False:
          x: [1,B,N,D]
          returns z:[1,B,N,D], credit_logits:[1,B,N,1]
        If True:
          x: [T,B,N,D]
          returns z:[T,B,N,D], credit_logits:[T,B,N,1]
        """

        # temporal + spatial
        T, B, N, D = x.shape

        # --- Temporal Attention per Agent ---
        # Reshape to [T, B*N, D] as per comment
        xt = x.view(T, B * N, D)
        xt = self.temporal(xt, causal=True)  # Process temporal attention
        # Reshape back to [T, B, N, D]
        xt = xt.view(T, B, N, D)
    
        # --- Spatial Attention per Timestep ---
        # Reshape to [T*B, N, D] for spatial attention
        xs = xt.view(T*B, N, D)
        # Expand agent mask to [T*B, N]
        kpm = None
        # 掩码应该提供的维度是[T, B, N]
        if agent_key_padding_mask is not None:
            kpm = agent_key_padding_mask.view(T*B, N)
        z = self.spatial(xs, key_padding_mask=kpm)

        # Apply fusion layer
        z = self.fuse(z)

        # Compute credit logits
        credit_logits = self.credit_head(z)

        # Reshape outputs to [T, B, N, D] and [T, B, N, 1]
        z = z.view(T, B, N, D)
        credit_logits = credit_logits.view(T, B, N, 1)

        return z, credit_logits