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

__PROPS__ = [
    "idx",
    "source",
    "source_attn_mask",
    "docstring",
    "docstring_attn_mask",
    "padded_feature",
    "padded_feature_attn_mask",
    "edge_index",
    "node_num",
    "edge_num",
    "padded_index",
    "padding_mask",
]

__ARGS_MAPPING__ = dict(zip(__PROPS__, range(len(__PROPS__))))

ARGS_SIZE = len(__ARGS_MAPPING__)


@dataclasses.dataclass
class GraphCoderBatch:
    idx: torch.Tensor
    source_: Dict[str, torch.Tensor]
    docstring_: Dict[str, torch.Tensor]
    edge_index: torch.Tensor
    padded_feature_: Dict[str, torch.Tensor]
    node_num: torch.Tensor
    edge_num: torch.Tensor
    padded_index: torch.Tensor
    padding_mask: torch.Tensor

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
    def padded_feature(self):
        return self.padded_feature_["input_ids"]

    @property
    def padded_feature_attn_mask(self):
        return self.padded_feature_["attention_mask"]

    @property
    def padded_feature_size(self):
        return self.padded_feature.size(-2)

    @property
    def graph_size(self) -> int:
        return self.padded_feature_size

    @property
    def has_source(self) -> bool:
        return self.source is not None and self.source_size > 0

    @property
    def has_docstring(self) -> bool:
        return self.docstring is not None and self.docstring_size > 0

    @property
    def has_graph(self) -> bool:
        return self.padded_feature is not None and self.padded_feature.size(-2) > 0

    def to_dict(self) -> Dict[str, torch.Tensor]:
        return {
            "idx": self.idx,
            "source": self.source,
            "source_attn_mask": self.source_attn_mask,
            "docstring": self.docstring,
            "docstring_attn_mask": self.docstring_attn_mask,
            "edge_index": self.edge_index,
            "padded_feature": self.padded_feature,
            "padded_feature_attn_mask": self.padded_feature_attn_mask,
            "node_num": self.node_num,
            "edge_num": self.edge_num,
            "padded_index": self.padded_index,
            "padding_mask": self.padding_mask,
        }

    def to_tuple(self) -> Tuple[torch.Tensor, ...]:
        return (
            self.idx,
            self.source,
            self.source_attn_mask,
            self.docstring,
            self.docstring_attn_mask,
            self.padded_feature,
            self.padded_feature_attn_mask,
            self.edge_index,
            self.node_num,
            self.edge_num,
            self.padded_index,
            self.padding_mask,
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
            padded_feature_={
                "input_ids": obj["padded_feature"],
                "attention_mask": obj["padded_feature_attn_mask"]
            },
            node_num=obj["node_num"],
            edge_num=obj["edge_num"],
            padded_index=obj["padded_index"],
            padding_mask=obj["padding_mask"],
        )

    @classmethod
    def from_tuple(
        cls: Type["GraphCoderBatch"], obj: Tuple[torch.Tensor, ...]
    ) -> "GraphCoderBatch":
        assert (
            len(obj) >= ARGS_SIZE
        ), f"Expected len to be at least {ARGS_SIZE}, got: {len(obj)}\n{obj}"
        return cls(
            idx=get_arg("idx", obj),
            source_={
                "input_ids": get_arg("source", obj),
                "attention_mask": get_arg("source_attn_mask", obj),
            },
            docstring_={
                "input_ids": get_arg("docstring", obj),
                "attention_mask": get_arg("docstring_attn_mask", obj),
            },
            edge_index=get_arg("edge_index", obj),
            padded_feature_={
                "input_ids": get_arg("padded_feature", obj),
                "attention_mask": get_arg("padded_feature_attn_mask", obj),
            },
            node_num=get_arg("node_num", obj),
            edge_num=get_arg("edge_num", obj),
            padded_index=get_arg("padded_index", obj),
            padding_mask=get_arg("padding_mask", obj),
        )


def get_arg(
    name: str, obj: Tuple[torch.Tensor, ...]
) -> torch.Tensor:
    return obj[get_arg_idx(name)]


def get_arg_idx(name: str) -> int:
    return __ARGS_MAPPING__[name]
