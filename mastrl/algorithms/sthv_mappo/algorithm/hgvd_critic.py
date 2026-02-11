import torch
import torch.nn as nn
from typing import Tuple


class HypergraphCritic(nn.Module):
    """HGVD-Lite: fixed sparse graph + K-hop message passing, O(N*k).

    Inputs:
        z: [B, N, D]    node embeddings
        a: [B, N, A]    action one-hot (or continuous vector)
        nei_idx: [N, k] fixed neighbor indices (int)
    Outputs:
        Q_tot: [B, 1]
        Q_i:   [B, N]
    """

    def __init__(self, embed_dim: int, act_dim: int, hidden_dim: int = 128, msg_hops: int = 2, k: int = 3,
                 symmetric_graph: bool = True):
        super().__init__()
        self.msg_hops = msg_hops
        self.k = k
        self.symmetric_graph = symmetric_graph

        dh = hidden_dim
        self.proj_in = nn.Linear(embed_dim, dh)
        self.msg_mlp = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dh * 2, dh),
                nn.ReLU(),
                nn.Linear(dh, dh),
            ) for _ in range(msg_hops)
        ])
        self.msg_ln = nn.ModuleList([nn.LayerNorm(dh) for _ in range(msg_hops)])

        # edge-level scorer
        self.edge_mlp = nn.Sequential(
            nn.Linear(dh * 2 + act_dim * 2, dh),
            nn.ReLU(),
            nn.Linear(dh, 1)
        )

    def _message_passing(self, z: torch.Tensor, nei_idx: torch.Tensor) -> torch.Tensor:
        """
        z: [B,N,D]
        nei_idx: [N,k]
        return h: [B,N,dh]
        """
        h = self.proj_in(z)
        # nei gather index shape [N,k]; expand batch inside torch.gather via advanced indexing
        for t in range(self.msg_hops):
            h_nei = h[:, nei_idx, :]  # [B,N,k,dh]
            agg = h_nei.mean(dim=2)   # [B,N,dh]
            msg_in = torch.cat([h, agg], dim=-1)
            delta = self.msg_mlp[t](msg_in)
            h = self.msg_ln[t](h + delta)
        return h

    def forward(self, z: torch.Tensor, a: torch.Tensor, nei_idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        z: [B,N,D], a: [B,N,A], nei_idx: [N,k]
        """
        h = self._message_passing(z, nei_idx)          # [B,N,dh]
        a = a

        # gather neighbors
        h_nei = h[:, nei_idx, :]                       # [B,N,k,dh]
        a_nei = a[:, nei_idx, :]                       # [B,N,k,A]

        h_i = h.unsqueeze(2).expand(-1, -1, self.k, -1)   # [B,N,k,dh]
        a_i = a.unsqueeze(2).expand(-1, -1, self.k, -1)   # [B,N,k,A]

        edge_in = torch.cat([h_i, h_nei, a_i, a_nei], dim=-1)  # [B,N,k,2dh+2A]
        q_ij = self.edge_mlp(edge_in).squeeze(-1)              # [B,N,k]

        Q_i = q_ij.sum(dim=2)                                  # [B,N]
        if self.symmetric_graph:
            Q_tot = 0.5 * Q_i.sum(dim=1, keepdim=True)         # [B,1]
        else:
            Q_tot = Q_i.sum(dim=1, keepdim=True)
        return Q_tot, Q_i
