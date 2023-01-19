import dataclasses

import torch

from .base import BaseExample


@dataclasses.dataclass
class AstData:
    x: list[str]
    edge_attr: list[str]
    edge_index: torch.LongTensor
    idx: int


@dataclasses.dataclass
class AstExample(BaseExample[AstData]):
    """Example for AST dataset."""

    pass
