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
from torch import nn
from torch.utils.data import DataLoader

from graph_coder.data import collate_ast
from graph_coder.datasets import AstDataset
from graph_coder.models import GraphCoderGenerator
from graph_coder.modules import TokenGTEncoder, TokenEmbedding
from graph_coder.runners import GraphCoderGeneratorRunner
from graph_coder.config.functional import get_pretrained_tokenizer


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
        collate_fn=partial(collate_ast, tokenizer=tokenizer),
    )
    embedding = nn.Embedding(
        len(tokenizer.vocab), 128, padding_idx=tokenizer.pad_token_id
    )

    graph_embedding = TokenEmbedding(
        embedding=embedding,
        ff=nn.Linear(64, 1, bias=False),
    )

    encoder = TokenGTEncoder(
        embedding=graph_embedding,
        encoder_embed_dim=128,
        encoder_ffn_embed_dim=128,
        lap_node_id=True,
        type_id=True,
    )
    text_encoder = nn.TransformerEncoder(
        encoder_layer=nn.TransformerEncoderLayer(d_model=128, nhead=8), num_layers=6
    )
    decoder = nn.TransformerDecoder(
        decoder_layer=nn.TransformerDecoderLayer(d_model=128, nhead=8), num_layers=6
    )

    generator = GraphCoderGenerator(
        embedding=embedding,
        encoder=text_encoder,
        graph_encoder=encoder,
        decoder=decoder,
        hidden_size=128,
        vocab_size=len(tokenizer.vocab),
        eos_token_id=tokenizer.eos_token_id,
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
            collate_ast, tokenizer=get_pretrained_tokenizer("EleutherAI/gpt-neox-20b")
        ),
        root=Path(__file__).parent / "./data",
        batch_size=2,
    )
    embedding = nn.Embedding(
        len(tokenizer.vocab), 128, padding_idx=tokenizer.pad_token_id
    )

    graph_embedding = TokenEmbedding(
        embedding=embedding,
        ff=nn.Linear(64, 1, bias=False),
    )

    encoder = TokenGTEncoder(
        embedding=graph_embedding,
        encoder_embed_dim=128,
        encoder_ffn_embed_dim=128,
        lap_node_id=True,
        type_id=True,
    )
    text_encoder = nn.TransformerEncoder(
        encoder_layer=nn.TransformerEncoderLayer(d_model=128, nhead=8), num_layers=6
    )
    decoder = nn.TransformerDecoder(
        decoder_layer=nn.TransformerDecoderLayer(d_model=128, nhead=8), num_layers=6
    )

    generator = GraphCoderGenerator(
        embedding=embedding,
        encoder=text_encoder,
        graph_encoder=encoder,
        decoder=decoder,
        hidden_size=128,
        vocab_size=len(tokenizer.vocab),
        eos_token_id=tokenizer.eos_token_id,
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
        dataset, collate_fn=partial(collate_ast, tokenizer=tokenizer), batch_size=2
    )

    for batch in loader:
        loss = runner._calc_loss(**batch)
        assert torch.is_floating_point(loss)
