#  Copyright 2023 German Arutyunov
#
#     Licensed under the Apache License, Version 2.0 (the "License");
#     you may not use this file except in compliance with the License.
#     You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#     Unless required by applicable law or agreed to in writing, software
#     distributed under the License is distributed on an "AS IS" BASIS,
#     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#     See the License for the specific language governing permissions and
#     limitations under the License.

from pathlib import Path

import torch
from torch import nn

from graph_coder.data import collate_ast
from graph_coder.datasets import AstDataset
from graph_coder.models import GraphCoderDocumenter
from graph_coder.modules import TokenGTEncoder
from graph_coder.runners import GraphCoderDocumenterRunner
from graph_coder.utils import get_pretrained_tokenizer, partial


def test_documenter_runner():
    tokenizer = get_pretrained_tokenizer("EleutherAI/gpt-neox-20b")
    dataset = AstDataset(
        collate_fn=partial(
            collate_ast, tokenizer=get_pretrained_tokenizer("EleutherAI/gpt-neox-20b")
        ),
        root=Path(__file__).parent / "./data",
        batch_size=2,
    )
    loader = dataset.loaders["train"]
    embedding = nn.Embedding(
        len(tokenizer.vocab), 128, padding_idx=tokenizer.pad_token_id
    )

    encoder = TokenGTEncoder(
        embedding=embedding,
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

    generator = GraphCoderDocumenter(
        embedding=embedding,
        encoder=text_encoder,
        graph_encoder=encoder,
        decoder=decoder,
        hidden_size=128,
        vocab_size=len(tokenizer.vocab),
        eos_token_id=tokenizer.eos_token_id,
    )

    runner = GraphCoderDocumenterRunner(
        generator,
        vocab_size=len(tokenizer.vocab),
        eos_token_id=tokenizer.eos_token_id,
    )
    runner._loaders = dataset.loaders
    runner.criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    runner._print_summary()

    for batch in iter(loader):
        loss = runner._calc_loss(batch)
        assert torch.is_floating_point(loss)
