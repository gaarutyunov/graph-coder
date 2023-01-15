import math
from pathlib import Path

from transformers import GPTNeoXTokenizerFast

from graph_coder.datasets.ast_dataset import AstDataset
from graph_coder.utils import get_pretrained_tokenizer


def test_collator():
    tokenizer = get_pretrained_tokenizer("EleutherAI/gpt-neox-20b")
    dataset = AstDataset(tokenizer=tokenizer, root=Path(__file__).parent / "./data")

    train = dataset.loaders["train"]
    val = dataset.loaders["val"]
    test = dataset.loaders["test"]

    assert len(train) == math.ceil(len(dataset.index) * 0.6)
    assert len(val) == math.floor(len(dataset.index) * 0.2)
    assert len(test) == math.floor(len(dataset.index) * 0.2)
