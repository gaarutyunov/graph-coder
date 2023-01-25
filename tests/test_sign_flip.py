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

from functools import partial
from pathlib import Path

import torch.nn as nn
from torch.utils.data import DataLoader

from graph_coder.data import collate_ast
from graph_coder.datasets import AstDataset
from graph_coder.modules import TokenGTEncoder
from graph_coder.utils import get_pretrained_tokenizer


def test_performer():
    tokenizer = get_pretrained_tokenizer("EleutherAI/gpt-neox-20b")
    dataset = AstDataset(
        root=Path(__file__).parent / "./data",
    )
    loader = DataLoader(
        dataset, collate_fn=partial(collate_ast, tokenizer=tokenizer), batch_size=2
    )
    embedding = nn.Embedding(
        len(tokenizer.vocab), 128, padding_idx=tokenizer.pad_token_id
    )

    encoder = TokenGTEncoder(
        embedding=embedding,
        encoder_embed_dim=128,
        encoder_ffn_embed_dim=128,
        lap_node_id=True,
        type_id=True,
        lap_node_id_sign_flip=True,
        lap_node_id_eig_dropout=0.1,
    )

    for batch in loader:
        encoded = encoder(batch)
        assert encoded.size(-1) == 128
