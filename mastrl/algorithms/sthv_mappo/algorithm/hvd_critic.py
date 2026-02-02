import torch
import torch.nn as nn
from typing import List, Tuple

class HypergraphCritic(nn.Module):
    """Hyperedge critics + aggregation.

    Inputs are agent embeddings z_i and actions a_i.
    For each hyperedge e, we pool (mean) group embeddings and group actions,
    then compute Q_e with an MLP. Finally:
      Q_tot = sum_e Q_e
      Q_i   = sum_{e contains i} Q_e

    This is the simplest stable instantiation; you can replace mean pooling
    with attention pooling later.
    """
    def __init__(self, embed_dim: int, act_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(embed_dim + act_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, z: torch.Tensor, a: torch.Tensor, hyperedges: List[List[int]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward.

        Args:
            z: [N, D]
            a: [N, A] (one-hot for discrete; raw action for continuous)
            hyperedges: list of groups
        Returns:
            Q_tot: [1]
            Q_i: [N]
        """
        device = z.device
        N = z.size(0)
        q_edges = []
        for e in hyperedges:
            idx = torch.as_tensor(e, device=device, dtype=torch.long)
            z_e = z.index_select(0, idx).mean(dim=0)     # [D]
            a_e = a.index_select(0, idx).mean(dim=0)     # [A]
            q_e = self.edge_mlp(torch.cat([z_e, a_e], dim=-1))  # [1]
            q_edges.append(q_e)

        q_edges_t = torch.stack(q_edges, dim=0)  # [E,1]
        Q_tot = q_edges_t.sum(dim=0)             # [1]

        Q_i = torch.zeros(N, device=device)
        for ei, e in enumerate(hyperedges):
            for i in e:
                Q_i[i] += q_edges_t[ei, 0]
        return Q_tot, Q_i
