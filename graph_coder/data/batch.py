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

import dataclasses

import torch
from typing import Dict


@dataclasses.dataclass
class GraphCoderBatch:
    idx: torch.Tensor
    source_: Dict[str, torch.Tensor]
    docstring_: Dict[str, torch.Tensor]
    edge_index: torch.Tensor
    edge_data_: Dict[str, torch.Tensor]
    node_data_: Dict[str, torch.Tensor]
    node_num: torch.Tensor
    edge_num: torch.Tensor
    lap_eigval: torch.Tensor
    lap_eigvec: torch.Tensor

    @property
    def batch_size(self) -> int:
        return self.idx.size(0)

    @property
    def source(self) -> torch.Tensor:
        return self.source_["input_ids"]

    @property
    def source_attn_mask(self):
        return self.source_["attention_mask"]

    @property
    def source_size(self) -> int:
        return self.source.size(-1)

    @property
    def docstring(self) -> torch.Tensor:
        return self.docstring_["input_ids"]

    @property
    def docstring_attn_mask(self) -> torch.Tensor:
        return self.docstring_["attention_mask"]

    @property
    def docstring_size(self) -> int:
        return self.docstring.size(-1)

    @property
    def edge_data(self):
        return self.edge_data_["input_ids"]

    @property
    def edge_data_attn_mask(self):
        return self.edge_data_["attention_mask"]

    @property
    def edge_data_size(self):
        return self.edge_data.size(-2)

    @property
    def node_data(self):
        return self.node_data_["input_ids"]

    @property
    def node_data_attn_mask(self):
        return self.node_data_["attention_mask"]

    @property
    def node_data_size(self):
        return self.node_data.size(-2)

    @property
    def graph_size(self) -> int:
        return self.node_data_size + self.edge_data_size

    @property
    def has_source(self) -> bool:
        return self.source is not None and self.source_size > 0

    @property
    def has_docstring(self) -> bool:
        return self.docstring is not None and self.docstring_size > 0

    @property
    def has_graph(self) -> bool:
        return self.node_data is not None and self.node_data.size(-2) > 0
