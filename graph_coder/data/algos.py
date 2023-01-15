import torch
from torch.linalg import eigh


@torch.jit.script
def lap_eig(
    edge_index: torch.LongTensor, num_nodes: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Computes Laplacian eigenvalues and eigenvectors with symmetric normalization."""
    dense_adj = torch.zeros([num_nodes, num_nodes], dtype=torch.bool)
    dense_adj[edge_index[0, :], edge_index[1, :]] = True
    in_degree = dense_adj.long().sum(dim=1).view(-1)
    A = dense_adj.float()
    D = torch.diag(in_degree.clip(1).pow(-0.5))
    L = torch.eye(num_nodes) - D @ A @ D

    return eigh(L)
