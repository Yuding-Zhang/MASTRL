import torch
import torch.nn.functional as F
from typing import List

def discover_hyperedges_knn(z: torch.Tensor, k: int = 3, max_group_size: int = 6) -> List[List[int]]:
    """Discover hyperedges using KNN + connected components.

    Args:
        z: [N,D] agent embeddings (single timestep, single env)
        k: KNN neighbors per agent
        max_group_size: split large components to keep critic cost bounded
    Returns:
        hyperedges: list of lists of agent indices
    """
    N = z.size(0)
    if N == 1:
        return [[0]]

    z = F.normalize(z, dim=-1)
    sim = z @ z.t()  # [N,N]
    sim.fill_diagonal_(-1e9)
    k = min(k, max(1, N - 1))
    knn = sim.topk(k, dim=-1).indices  # [N,k]

    adj = torch.zeros(N, N, dtype=torch.bool, device=z.device)
    for i in range(N):
        adj[i, knn[i]] = True
    adj = adj | adj.t()

    visited = [False] * N
    hyperedges: List[List[int]] = []
    for i in range(N):
        if visited[i]:
            continue
        stack = [i]
        visited[i] = True
        comp = []
        while stack:
            u = stack.pop()
            comp.append(u)
            neigh = torch.where(adj[u])[0].tolist()
            for v in neigh:
                if not visited[v]:
                    visited[v] = True
                    stack.append(v)

        # split overly large components
        if len(comp) <= max_group_size:
            hyperedges.append(comp)
        else:
            for j in range(0, len(comp), max_group_size):
                hyperedges.append(comp[j:j+max_group_size])

    return hyperedges

def discover_hyperedges_knn_with_adj(z: torch.Tensor, k: int = 3, max_group_size: int = 6):
    """Discover hyperedges using KNN + connected components and return the KNN adjacency (symmetric).
    Args:
        z: [N,D] agent embeddings
    Returns:
        hyperedges: list of lists of agent indices
        adj: [N,N] bool symmetric adjacency (no self loops)
    """
    N = z.size(0)
    if N == 1:
        return [[0]], torch.zeros(1, 1, dtype=torch.bool, device=z.device)

    z = F.normalize(z, dim=-1)
    sim = z @ z.t()
    sim.fill_diagonal_(-1e9)
    k = min(k, max(1, N - 1))
    knn = sim.topk(k, dim=-1).indices  # [N,k]

    adj = torch.zeros(N, N, dtype=torch.bool, device=z.device)
    for i in range(N):
        adj[i, knn[i]] = True
    adj = adj | adj.t()

    visited = [False] * N
    hyperedges: List[List[int]] = []
    for i in range(N):
        if visited[i]:
            continue
        stack = [i]
        visited[i] = True
        comp = []
        while stack:
            u = stack.pop()
            comp.append(u)
            neigh = torch.where(adj[u])[0].tolist()
            for v in neigh:
                if not visited[v]:
                    visited[v] = True
                    stack.append(v)

        if len(comp) <= max_group_size:
            hyperedges.append(comp)
        else:
            for j in range(0, len(comp), max_group_size):
                hyperedges.append(comp[j:j+max_group_size])

    return hyperedges, adj
