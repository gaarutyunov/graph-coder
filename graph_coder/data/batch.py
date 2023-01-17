import dataclasses

import torch


@dataclasses.dataclass
class GraphCoderBatch:
    idx: torch.Tensor
    source_: dict[str, torch.Tensor]
    docstring_: dict[str, torch.Tensor]
    edge_index: torch.LongTensor
    edge_data_: dict[str, torch.Tensor]
    node_data_: dict[str, torch.Tensor]
    node_num: torch.LongTensor
    edge_num: torch.LongTensor
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
