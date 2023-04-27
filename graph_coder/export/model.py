#  Copyright 2023 German Arutyunov
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
import os
from io import BytesIO
from typing import Tuple, Union

import hiddenlayer as hl

import numpy as np
import torch
from onnxruntime_extensions import onnx_op, PyOp
from torch import nn
from torch.onnx import register_custom_op_symbolic

from graph_coder.models import LmLayer
from graph_coder.modules import PerformerEncoder, TokenEmbedding, TokenGTEncoder


def eigh_op(g, x, _):
    return g.op("ai.onnx.contrib::LinalgEigh", x, outputs=2)


@onnx_op(
    op_type="LinalgEigh", inputs=[PyOp.dt_float], outputs=[PyOp.dt_float, PyOp.dt_float]
)
def eigh(x):
    return np.linalg.eigh(x)


register_custom_op_symbolic("::linalg_eigh", eigh_op, 1)


class GraphCoder(nn.Module):
    """GraphCoder model suitable for visualization"""

    def __init__(
        self,
        hidden_size: int = 4,
        vocab_size: int = 50277,
        max_length: int = 4,
        max_seq_length: int = 256,
        num_layers: int = 1,
        num_heads: int = 1,
        head_size: int = 4,
        ffn_size: int = 4,
        dropout: float = 0.1,
        lap_node_id: bool = True,
        lap_node_id_k: int = 4,
        lap_node_id_sign_flip: bool = True,
        shift: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.max_seq_length = max_seq_length

        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.token_embedding = TokenEmbedding(
            embedding=self.embedding,
            ff=nn.Linear(max_length, 1),
        )
        self.docstring_encoder = PerformerEncoder(
            dim=hidden_size,
            dim_head=head_size,
            depth=num_layers,
            heads=num_heads,
            causal=True,
            max_seq_len=max_seq_length,
        )
        self.code_encoder = PerformerEncoder(
            dim=hidden_size,
            dim_head=head_size,
            depth=num_layers,
            heads=num_heads,
            causal=True,
            max_seq_len=max_seq_length,
        )
        self.graph_encoder = TokenGTEncoder(
            embedding=self.token_embedding,
            encoder_layers=num_layers,
            encoder_attention_heads=num_heads,
            encoder_embed_dim=hidden_size,
            encoder_ffn_embed_dim=ffn_size,
            dropout=dropout,
            attention_dropout=0.0,
            lap_node_id=lap_node_id,
            lap_node_id_k=lap_node_id_k,
            lap_node_id_sign_flip=lap_node_id_sign_flip,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(
                d_model=hidden_size,
                nhead=num_heads,
                dim_feedforward=ffn_size,
                dropout=dropout,
            ),
            num_layers=num_layers,
        )
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()
        self.lm_layer = LmLayer(
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            max_length=max_length,
            shift=shift,
        )

    def forward(
        self,
        idx,
        source,
        source_attn_mask,
        docstring,
        docstring_attn_mask,
        padded_feature,
        padded_feature_attn_mask,
        edge_index,
        node_num,
        edge_num,
        padded_index,
        padding_mask,
    ) -> torch.Tensor:
        docstring_emb = self.embedding(docstring)
        text = self.docstring_encoder(docstring_emb)
        code_emb = self.embedding(source)
        code = self.code_encoder(code_emb)

        graph_emb = self.token_embedding(padded_feature).squeeze(-1)

        graph = self.graph_encoder(
            edge_index, node_num, edge_num, padded_index, padding_mask, padded_feature
        )

        tgt, memory = torch.cat([docstring_emb, graph_emb, code_emb], dim=1), torch.cat(
            [text, graph, code], dim=1
        )

        hidden_states = self.decoder(tgt, memory)
        hidden_states = self.activation(self.dense(hidden_states)).contiguous()

        return self.lm_layer(
            idx,
            source,
            source_attn_mask,
            docstring,
            docstring_attn_mask,
            padded_feature,
            padded_feature_attn_mask,
            edge_index,
            node_num,
            edge_num,
            padded_index,
            padding_mask,
            hidden_states,
        )


def export_model(
    out: Union[str, BytesIO, os.PathLike],
    dummy_input: Tuple[torch.Tensor, ...],
    model: nn.Module = GraphCoder(),
    format: str = "onnx",
    **kwargs,
) -> None:
    """Export model"""
    if format == "onnx":
        _export_onnx(out, dummy_input, model)
    elif format == "hl":
        _export_hl(out, dummy_input, model, **kwargs)


def _export_onnx(
    out: Union[str, BytesIO, os.PathLike],
    dummy_input: Tuple[torch.Tensor, ...],
    model: nn.Module = GraphCoder(),
):
    """Export model to ONNX"""
    model.eval()

    torch.onnx.export(
        model,
        dummy_input,
        out,
        verbose=True,
        input_names=[
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
        ],
        output_names=["logits"],
    )


def _export_hl(
    out: Union[str, BytesIO, os.PathLike],
    dummy_input: Tuple[torch.Tensor, ...],
    model: nn.Module = GraphCoder(),
    **kwargs,
):
    """Export model with HiddenLayer"""
    assert not isinstance(out, BytesIO), "`out` only supports string path in `hl` format"

    build_graph_kwargs = kwargs.get("build_graph", {})
    save_kwargs = kwargs.get("save", {})

    hl.build_graph(model, dummy_input, **build_graph_kwargs).save(out, **save_kwargs)
