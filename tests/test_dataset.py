from pathlib import Path

import numpy as np
from transformers import GPTNeoXTokenizerFast

from graph_coder.datasets.ast_dataset import AstDataset
from graph_coder.utils import get_pretrained_tokenizer


def test_dataset():
    tokenizer = get_pretrained_tokenizer("EleutherAI/gpt-neox-20b")
    dataset = AstDataset(tokenizer=tokenizer, root=Path(__file__).parent / "./data")
    assert dataset is not None
    assert np.all(
        dataset.index["source"] != "function_002.py"
    ), "Small graph should be skipped"

    for i in range(len(dataset)):
        data = dataset[i]
        assert data is not None
        assert data.graph.edge_index.size(0) == 2, "Edge should have 2 nodes"
        assert data.graph.edge_index.size(1) > 0, "Graph should have edges"

        if dataset.index.iloc[i]["source"] == "function_003.py":
            assert data.docstring.size(0) == 0, "Docstring should be empty"
