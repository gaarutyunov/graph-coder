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

from functools import partial
from pathlib import Path

import torch

from graph_coder.config.functional import get_pretrained_tokenizer

from graph_coder.data import collate_ast
from graph_coder.datasets import AstDataset
from graph_coder.modules import TokenEmbedding, TokenGTEncoder, TokenGTEncoderPipe

from torch import nn as nn
from torch.utils.data import DataLoader


def test_tokengt_encoder():
    tokenizer = get_pretrained_tokenizer("EleutherAI/gpt-neox-20b")
    dataset = AstDataset(
        root=Path(__file__).parent / "./data",
    )
    loader = DataLoader(
        dataset,
        collate_fn=partial(collate_ast, tokenizer=tokenizer, max_length=8),
        batch_size=2,
    )
    embedding = TokenEmbedding(
        embedding=nn.Embedding(len(tokenizer.vocab), 16, padding_idx=1),
        ff=nn.Linear(8, 1, bias=False),
    )

    encoder = TokenGTEncoder(
        embedding=embedding,
        encoder_embed_dim=16,
        encoder_ffn_embed_dim=32,
        lap_node_id=True,
        type_id=True,
        encoder_layers=2,
        encoder_attention_heads=2,
    )

    for batch in loader:
        encoded = encoder(
            batch["edge_index"],
            batch["edge_data"],
            batch["node_data"],
            batch["node_num"],
            batch["edge_num"],
            batch["lap_eigvec"],
        )
        assert encoded.size(-1) == 16


def test_sign_flip():
    tokenizer = get_pretrained_tokenizer("EleutherAI/gpt-neox-20b")
    dataset = AstDataset(
        root=Path(__file__).parent / "./data",
    )
    loader = DataLoader(
        dataset,
        collate_fn=partial(collate_ast, tokenizer=tokenizer, max_length=8),
        batch_size=2,
    )
    embedding = TokenEmbedding(
        embedding=nn.Embedding(len(tokenizer.vocab), 16, padding_idx=1),
        ff=nn.Linear(8, 1, bias=False),
    )

    encoder = TokenGTEncoder(
        embedding=embedding,
        encoder_embed_dim=16,
        encoder_ffn_embed_dim=32,
        lap_node_id=True,
        type_id=True,
        lap_node_id_sign_flip=True,
        lap_node_id_eig_dropout=0.1,
        encoder_layers=2,
        encoder_attention_heads=2,
    )

    for batch in loader:
        encoded = encoder(
            batch["edge_index"],
            batch["edge_data"],
            batch["node_data"],
            batch["node_num"],
            batch["edge_num"],
            batch["lap_eigvec"],
        )
        assert encoded.size(-1) == 16


def test_performer():
    tokenizer = get_pretrained_tokenizer("EleutherAI/gpt-neox-20b")
    dataset = AstDataset(
        root=Path(__file__).parent / "./data",
    )
    loader = DataLoader(
        dataset,
        collate_fn=partial(collate_ast, tokenizer=tokenizer, max_length=8),
        batch_size=2,
    )
    embedding = TokenEmbedding(
        embedding=nn.Embedding(len(tokenizer.vocab), 16, padding_idx=1),
        ff=nn.Linear(8, 1, bias=False),
    )

    encoder = TokenGTEncoder(
        embedding=embedding,
        encoder_embed_dim=16,
        encoder_ffn_embed_dim=32,
        lap_node_id=True,
        type_id=True,
        performer=True,
        attention_dropout=0.0,  # necessary for performer
        causal=True,
        encoder_layers=2,
        encoder_attention_heads=2,
    )

    for batch in loader:
        encoded = encoder(
            batch["edge_index"],
            batch["edge_data"],
            batch["node_data"],
            batch["node_num"],
            batch["edge_num"],
            batch["lap_eigvec"],
        )
        assert encoded.size(-1) == 16


def test_graphormer_init():
    tokenizer = get_pretrained_tokenizer("EleutherAI/gpt-neox-20b")
    dataset = AstDataset(
        root=Path(__file__).parent / "./data",
    )
    loader = DataLoader(
        dataset,
        collate_fn=partial(collate_ast, tokenizer=tokenizer, max_length=8),
        batch_size=2,
    )
    embedding = TokenEmbedding(
        embedding=nn.Embedding(len(tokenizer.vocab), 16, padding_idx=1),
        ff=nn.Linear(8, 1, bias=False),
    )

    encoder = TokenGTEncoder(
        embedding=embedding,
        encoder_embed_dim=16,
        encoder_ffn_embed_dim=32,
        lap_node_id=True,
        type_id=True,
        performer=True,
        attention_dropout=0.0,  # necessary for performer
        causal=True,
        apply_graphormer_init=True,
        encoder_layers=2,
        encoder_attention_heads=2,
    )

    for batch in loader:
        encoded = encoder(
            batch["edge_index"],
            batch["edge_data"],
            batch["node_data"],
            batch["node_num"],
            batch["edge_num"],
            batch["lap_eigvec"],
        )
        assert encoded.size(-1) == 16


def test_pipe():
    tokenizer = get_pretrained_tokenizer("EleutherAI/gpt-neox-20b")
    dataset = AstDataset(
        root=Path(__file__).parent / "./data",
    )
    loader = DataLoader(
        dataset,
        collate_fn=partial(
            collate_ast, tokenizer=tokenizer, max_length=8, use_dict=False
        ),
        batch_size=2,
    )
    embedding = TokenEmbedding(
        embedding=nn.Embedding(len(tokenizer.vocab), 16, padding_idx=1),
        ff=nn.Linear(8, 1, bias=False),
    )

    encoder = TokenGTEncoderPipe(
        embedding=embedding,
        encoder_embed_dim=16,
        encoder_ffn_embed_dim=32,
        lap_node_id=True,
        type_id=True,
        performer=True,
        attention_dropout=0.0,  # necessary for performer
        causal=True,
        apply_graphormer_init=True,
        encoder_layers=2,
        encoder_attention_heads=2,
    )

    for batch in loader:
        args = batch
        for layer in encoder.to_layers():
            args = layer(*args)

        for arg in args:
            assert torch.is_tensor(arg)
