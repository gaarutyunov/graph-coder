from functools import partial
from pathlib import Path

from torch.utils.data import DataLoader

from graph_coder.data.collator import collate_ast
from graph_coder.datasets import AstDataset
from graph_coder.utils import get_pretrained_tokenizer


def test_collator():
    tokenizer = get_pretrained_tokenizer("EleutherAI/gpt-neox-20b")

    dataset = AstDataset(tokenizer=tokenizer, root=Path(__file__).parent / "./data")
    loader = DataLoader(
        dataset, batch_size=2, collate_fn=partial(collate_ast, tokenizer=tokenizer)
    )

    for batch in loader:
        assert batch.edge_index.size(0) == 2
        assert batch.edge_data.size(0) == 2 or batch.edge_data.size(0) == 1
        assert (
            batch.edge_data.size(1) == (batch.edge_data != -100).sum(dim=1).max().item()
        )
        assert batch.node_data.size(0) == 2 or batch.node_data.size(0) == 1
        assert (
            batch.node_data.size(1) == (batch.node_data != -100).sum(dim=1).max().item()
        )
