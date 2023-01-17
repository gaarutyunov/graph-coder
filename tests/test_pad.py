from graph_coder.data import pad

import torch

from graph_coder.utils import get_pretrained_tokenizer


def test_pad():
    tokenizer = get_pretrained_tokenizer("EleutherAI/gpt-neox-20b")

    texts = [
        "This is a test",
        "This is another test",
        "This is a third test",
        "This is a fourth test",
        """This is not a test, but a fifth test. This is some huge text. It is needed to test the truncation.
        This is not a test, but a fifth test. This is some huge text. It is needed to test the truncation.
        This is not a test, but a fifth test. This is some huge text. It is needed to test the truncation.
        This is not a test, but a fifth test. This is some huge text. It is needed to test the truncation.
        This is not a test, but a fifth test. This is some huge text. It is needed to test the truncation.""",
    ]
    num = [2, 3]

    padded = pad(texts, num, tokenizer)

    assert padded["input_ids"].shape == (2, 3, 64)
    assert padded["attention_mask"].shape == (2, 3, 64)
    assert torch.all(padded["input_ids"][0, -1] == 1)
    assert padded["attention_mask"][0, -1].sum() == 0
