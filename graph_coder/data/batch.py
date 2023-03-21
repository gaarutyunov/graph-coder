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
from typing import Dict, Tuple, Type

import torch

__ARGS_MAPPING__ = {
    "idx": 0,
    "source": 1,
    "source_attn_mask": 2,
    "docstring": 3,
    "docstring_attn_mask": 4,
    "edge_index": 5,
    "edge_data": 6,
    "edge_data_attn_mask": 7,
    "node_data": 8,
    "node_data_attn_mask": 9,
    "node_num": 10,
    "edge_num": 11,
    "lap_eigval": 12,
    "lap_eigvec": 13,
}

ARGS_SIZE = len(__ARGS_MAPPING__)


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

    def to_dict(self) -> Dict[str, torch.Tensor]:
        return {
            "idx": self.idx,
            "source": self.source,
            "source_attn_mask": self.source_attn_mask,
            "docstring": self.docstring,
            "docstring_attn_mask": self.docstring_attn_mask,
            "edge_index": self.edge_index,
            "edge_data": self.edge_data,
            "edge_data_attn_mask": self.edge_data_attn_mask,
            "node_data": self.node_data,
            "node_data_attn_mask": self.node_data_attn_mask,
            "node_num": self.node_num,
            "edge_num": self.edge_num,
            "lap_eigval": self.lap_eigval,
            "lap_eigvec": self.lap_eigvec,
        }

    def to_tuple(self) -> Tuple[torch.Tensor, ...]:
        return (
            self.idx,
            self.source,
            self.source_attn_mask,
            self.docstring,
            self.docstring_attn_mask,
            self.edge_index,
            self.edge_data,
            self.edge_data_attn_mask,
            self.node_data,
            self.node_data_attn_mask,
            self.node_num,
            self.edge_num,
            self.lap_eigval,
            self.lap_eigvec,
        )

    @classmethod
    def from_dict(
        cls: Type["GraphCoderBatch"], obj: Dict[str, torch.Tensor]
    ) -> "GraphCoderBatch":
        return cls(
            idx=obj["idx"],
            source_={
                "input_ids": obj["source"],
                "attention_mask": obj["source_attn_mask"],
            },
            docstring_={
                "input_ids": obj["docstring"],
                "attention_mask": obj["docstring_attn_mask"],
            },
            edge_index=obj["edge_index"],
            edge_data_={
                "input_ids": obj["edge_data"],
                "attention_mask": obj["edge_data_attn_mask"],
            },
            node_data_={
                "input_ids": obj["node_data"],
                "attention_mask": obj["node_data_attn_mask"],
            },
            node_num=obj["node_num"],
            edge_num=obj["edge_num"],
            lap_eigval=obj["lap_eigval"],
            lap_eigvec=obj["lap_eigvec"],
        )

    @classmethod
    def from_tuple(
        cls: Type["GraphCoderBatch"], obj: Tuple[torch.Tensor, ...]
    ) -> "GraphCoderBatch":
        assert len(obj) >= ARGS_SIZE, f"Expected len to be at least {ARGS_SIZE}, got: {len(obj)}\n{obj}"
        return cls(
            idx=obj[0],
            source_={
                "input_ids": obj[1],
                "attention_mask": obj[2],
            },
            docstring_={
                "input_ids": obj[3],
                "attention_mask": obj[4],
            },
            edge_index=obj[5],
            edge_data_={
                "input_ids": obj[6],
                "attention_mask": obj[7],
            },
            node_data_={
                "input_ids": obj[8],
                "attention_mask": obj[9],
            },
            node_num=obj[10],
            edge_num=obj[11],
            lap_eigval=obj[12],
            lap_eigvec=obj[13],
        )

    @classmethod
    def get_arg_idx(cls: Type["GraphCoderBatch"], name: str) -> int:
        return __ARGS_MAPPING__[name]

    @classmethod
    def get_arg(
        cls: Type["GraphCoderBatch"], name: str, obj: Tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        return obj[cls.get_arg_idx(name)]
