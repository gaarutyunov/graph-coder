import dataclasses

import torch
from torch_geometric.data import Data


@dataclasses.dataclass
class BaseExample:
    """Base class for examples."""

    source: torch.LongTensor
    graph: Data
    docstring: torch.LongTensor
