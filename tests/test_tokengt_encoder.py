from functools import partial
from pathlib import Path

from torch.utils.data import DataLoader
import torch.nn as nn

from graph_coder.data import collate
from graph_coder.datasets import AstDataset
from graph_coder.modules import TokenGTEncoder
from graph_coder.utils import get_pretrained_tokenizer


def test_tokengt_encoder():
    tokenizer = get_pretrained_tokenizer("EleutherAI/gpt-neox-20b")
    dataset = AstDataset(tokenizer=tokenizer, root=Path(__file__).parent / "./data")
    loader = DataLoader(dataset, batch_size=2, collate_fn=partial(collate, pad_token_id=tokenizer.pad_token_id))
    embedding = nn.Embedding(tokenizer.vocab_size, 128, padding_idx=tokenizer.pad_token_id)

    encoder = TokenGTEncoder(
        embedding=embedding,
        encoder_embed_dim=128,
        encoder_ffn_embed_dim=128,
        lap_node_id=True,
        type_id=True,
    )

    for batch in loader:
        encoded = encoder(batch)
        assert encoded.size(-1) == 128
