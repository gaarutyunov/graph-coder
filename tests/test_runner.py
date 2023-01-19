from pathlib import Path

import torch
from torch import nn

from graph_coder.datasets import AstDataset
from graph_coder.models import GraphCoderGenerator
from graph_coder.modules import TokenGTEncoder
from graph_coder.runners import GraphCoderGeneratorRunner
from graph_coder.utils import get_pretrained_tokenizer


def test_runner():
    tokenizer = get_pretrained_tokenizer("EleutherAI/gpt-neox-20b")
    dataset = AstDataset(
        tokenizer=tokenizer, root=Path(__file__).parent / "./data", batch_size=2
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

    generator = GraphCoderGenerator(
        embedding=embedding,
        encoder=text_encoder,
        graph_encoder=encoder,
        decoder=decoder,
        hidden_size=128,
        vocab_size=len(tokenizer.vocab),
        eos_token_id=tokenizer.eos_token_id,
    )

    runner = GraphCoderGeneratorRunner(generator)
    runner.criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    for batch in loader:
        loss = runner._calc_loss(batch)
        assert torch.is_floating_point(loss)
