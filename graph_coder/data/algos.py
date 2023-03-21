#  Copyright 2023 German Arutyunov
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from typing import Tuple

import torch
from torch.linalg import eigh


def lap_eig(
    edge_index: torch.Tensor, num_nodes: int, dtype: torch.dtype = torch.float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Computes Laplacian eigenvalues and eigenvectors with symmetric normalization."""
    dense_adj = torch.zeros(
        [num_nodes, num_nodes], dtype=torch.bool
    )
    dense_adj[edge_index[0, :], edge_index[1, :]] = True
    in_degree = dense_adj.float().sum(dim=1).view(-1)
    A = dense_adj.float()
    D = torch.diag(in_degree.clip(1).pow(-0.5))
    L = torch.eye(num_nodes, dtype=dtype) - D @ A @ D

    lap_eigval, lap_eigvec = eigh(L)
    return lap_eigval.type(dtype), lap_eigvec.type(dtype)
