from pathlib import Path

from graph_coder.datasets import AstDataset
from graph_coder.utils import get_pretrained_tokenizer


def test_error():
    tokenizer = get_pretrained_tokenizer("EleutherAI/gpt-neox-20b")
    dataset = AstDataset(tokenizer=tokenizer, root=Path(__file__).parent / "./data", introspect=True)
    with open(dataset.log_file, "r") as log:
        lines = log.readlines()
        assert len(lines) == 1
        assert lines[0].endswith("Refactoring error_001.py: bad input: type=0, value='', context=('\\n', (2, 0))\n")

