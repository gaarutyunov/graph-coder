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
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from graph_coder.config.functional import get_pretrained_tokenizer
from graph_coder.data import collate_ast
from graph_coder.datasets import HumanEvalDataset
from graph_coder.models import GraphCoderGenerator
from graph_coder.modules import TokenEmbedding, TokenGTEncoder

from graph_coder.runners.human_eval import HumanEvalRunner
from torch import nn
from torch.utils.data import DataLoader, Subset


def test_human_eval():
    root = Path(__file__).parent / "human_eval_data"
    root.mkdir(exist_ok=True)
    tokenizer = get_pretrained_tokenizer("EleutherAI/gpt-neox-20b")
    collator = partial(
        collate_ast,
        max_length=8,
        max_seq_length=256,
        tokenizer=tokenizer,
        num_samples=10,
    )

    dataset = HumanEvalDataset(
        root=root,
        batch_size=1,
        collate_fn=collator,
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
        max_seq_length=256,
    )

    log_folder = Path(__file__).parent / "logs"

    torch.save(generator.state_dict(), log_folder / "model.best.pth")

    np.random.seed(42)
    subset = np.random.randint(0, len(dataset), 2)

    loader = DataLoader(
        Subset(dataset, subset), collate_fn=collator, batch_size=dataset.batch_size
    )

    runner = HumanEvalRunner(
        model=generator,
        log_path=log_folder,
        tokenizer=tokenizer,
        eos_token_id=generator.eos_token_id,
        vocab_size=generator.vocab_size,
        problem_file=root / "HumanEval.jsonl.gz",
        num_samples=10,
        repetition_penalty=1.5,
        temperature=0.7,
        top_p=0.95,
        top_k=1000,
    )

    runner.evaluate(loader)

    results_file = log_folder / "human_eval/samples.jsonl_results.jsonl"

    assert results_file.exists()

    df = pd.read_json(results_file, lines=True)

    assert df.columns.tolist() == ["task_id", "completion", "result", "passed"]
