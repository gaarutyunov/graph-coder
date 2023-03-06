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
from graph_coder.models import GraphCoderGenerator, GraphCoderGeneratorPipe
from graph_coder.modules import (
    TokenEmbedding,
    TokenGTEncoder,
    TokenGTEncoderPipe,
    TransformerDecoderPipe,
    TransformerEncoderPipe,
)
from graph_coder.runners import GraphCoderGeneratorRunner
from torch import nn
from torch.utils.data import DataLoader


def test_generator():
    tokenizer = get_pretrained_tokenizer("EleutherAI/gpt-neox-20b")
    dataset = AstDataset(
        collate_fn=partial(
            collate_ast, tokenizer=get_pretrained_tokenizer("EleutherAI/gpt-neox-20b")
        ),
        root=Path(__file__).parent / "./data",
    )
    loader = DataLoader(
        dataset,
        batch_size=2,
        collate_fn=partial(collate_ast, tokenizer=tokenizer, max_length=8),
    )
    embedding = nn.Embedding(
        len(tokenizer.vocab), 16, padding_idx=tokenizer.pad_token_id
    )

    graph_embedding = TokenEmbedding(
        embedding=embedding,
        ff=nn.Linear(8, 1, bias=False),
    )

    encoder = TokenGTEncoder(
        embedding=graph_embedding,
        encoder_embed_dim=16,
        encoder_ffn_embed_dim=32,
        lap_node_id=True,
        type_id=True,
        encoder_layers=2,
        encoder_attention_heads=2,
    )
    text_encoder = nn.TransformerEncoder(
        encoder_layer=nn.TransformerEncoderLayer(d_model=16, nhead=2), num_layers=2
    )
    code_encoder = nn.TransformerEncoder(
        encoder_layer=nn.TransformerEncoderLayer(d_model=16, nhead=2), num_layers=2
    )
    decoder = nn.TransformerDecoder(
        decoder_layer=nn.TransformerDecoderLayer(d_model=16, nhead=2), num_layers=2
    )

    generator = GraphCoderGenerator(
        embedding=embedding,
        text_encoder=text_encoder,
        code_encoder=code_encoder,
        graph_encoder=encoder,
        decoder=decoder,
        hidden_size=16,
        vocab_size=len(tokenizer.vocab),
        eos_token_id=tokenizer.eos_token_id,
        max_length=8,
    )

    for batch in loader:
        decoded = generator(**batch)
        if "docstring" in decoded:
            assert decoded["docstring"].size(-1) == len(tokenizer.vocab)
        if "graph" in decoded:
            assert decoded["graph"].size(-1) == len(tokenizer.vocab)
        if "source" in decoded:
            assert decoded["source"].size(-1) == len(tokenizer.vocab)


def test_generator_runner():
    tokenizer = get_pretrained_tokenizer("EleutherAI/gpt-neox-20b")
    dataset = AstDataset(
        collate_fn=partial(
            collate_ast,
            tokenizer=get_pretrained_tokenizer("EleutherAI/gpt-neox-20b"),
            max_length=8,
        ),
        root=Path(__file__).parent / "./data",
        batch_size=2,
    )
    embedding = nn.Embedding(
        len(tokenizer.vocab), 16, padding_idx=tokenizer.pad_token_id
    )

    graph_embedding = TokenEmbedding(
        embedding=embedding,
        ff=nn.Linear(8, 1, bias=False),
    )

    encoder = TokenGTEncoder(
        embedding=graph_embedding,
        encoder_embed_dim=16,
        encoder_ffn_embed_dim=32,
        lap_node_id=True,
        type_id=True,
        encoder_layers=2,
        encoder_attention_heads=2,
    )
    text_encoder = nn.TransformerEncoder(
        encoder_layer=nn.TransformerEncoderLayer(d_model=16, nhead=2), num_layers=2
    )
    code_encoder = nn.TransformerEncoder(
        encoder_layer=nn.TransformerEncoderLayer(d_model=16, nhead=2), num_layers=2
    )
    decoder = nn.TransformerDecoder(
        decoder_layer=nn.TransformerDecoderLayer(d_model=16, nhead=2), num_layers=2
    )

    generator = GraphCoderGenerator(
        embedding=embedding,
        text_encoder=text_encoder,
        code_encoder=code_encoder,
        graph_encoder=encoder,
        decoder=decoder,
        hidden_size=16,
        vocab_size=len(tokenizer.vocab),
        eos_token_id=tokenizer.eos_token_id,
        max_length=8,
    )

    runner = GraphCoderGeneratorRunner(
        generator,
        vocab_size=len(tokenizer.vocab),
        eos_token_id=tokenizer.eos_token_id,
    )
    runner._loaders = dataset.loaders
    runner.criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    runner._print_summary()

    loader = DataLoader(
        dataset,
        collate_fn=partial(collate_ast, tokenizer=tokenizer, max_length=8),
        batch_size=2,
    )

    for batch in loader:
        loss = runner._calc_loss(**batch)
        assert torch.is_floating_point(loss)


def test_generator_pipe():
    tokenizer = get_pretrained_tokenizer("EleutherAI/gpt-neox-20b")
    collator = partial(
        collate_ast,
        tokenizer=get_pretrained_tokenizer("EleutherAI/gpt-neox-20b"),
        max_length=4,
    )
    dataset = AstDataset(
        collate_fn=collator,
        root=Path(__file__).parent / "./data",
    )
    loader = DataLoader(
        dataset,
        batch_size=2,
        collate_fn=collator,
    )
    embedding = nn.Embedding(
        len(tokenizer.vocab), 16, padding_idx=tokenizer.pad_token_id
    )

    graph_embedding = TokenEmbedding(
        embedding=embedding,
        ff=nn.Linear(4, 1, bias=False),
    )

    encoder = TokenGTEncoderPipe(
        embedding=graph_embedding,
        encoder_embed_dim=16,
        encoder_ffn_embed_dim=16,
        lap_node_id=True,
        type_id=True,
    )
    text_encoder = TransformerEncoderPipe(
        encoder_layer=nn.TransformerEncoderLayer(d_model=16, nhead=2), num_layers=2
    )
    code_encoder = TransformerEncoderPipe(
        encoder_layer=nn.TransformerEncoderLayer(d_model=16, nhead=2), num_layers=2
    )
    decoder = TransformerDecoderPipe(
        decoder_layer=nn.TransformerDecoderLayer(d_model=16, nhead=2), num_layers=2
    )

    generator = GraphCoderGeneratorPipe(
        embedding=embedding,
        text_encoder=text_encoder,
        code_encoder=code_encoder,
        graph_encoder=encoder,
        decoder=decoder,
        hidden_size=16,
        vocab_size=len(tokenizer.vocab),
        eos_token_id=tokenizer.eos_token_id,
        max_length=4,
    )

    layers = generator.to_layers()

    for batch in loader:
        decoded = generator(**batch)

        kwargs = batch
        for layer in layers:
            kwargs = layer(**kwargs)

        if "docstring" in kwargs:
            assert kwargs["docstring"].shape == decoded["docstring"].shape
        if "graph" in kwargs:
            assert kwargs["graph"].shape == decoded["graph"].shape
        if "source" in kwargs:
            assert kwargs["source"].shape == decoded["source"].shape
