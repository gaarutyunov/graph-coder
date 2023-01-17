import typing

from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from graph_coder.datasets.base import BaseDataset


def get_loaders(dataset: BaseDataset) -> dict[str, DataLoader]:
    return dataset.loaders


def get_model_parameters(model: nn.Module) -> typing.Iterator[nn.Parameter]:
    return model.parameters()


def get_pretrained_tokenizer(name: str, pad_token_id: int = 1, eos_token_id: int = 0) -> PreTrainedTokenizerFast:
    tokenizer = AutoTokenizer.from_pretrained(name)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = pad_token_id

    if tokenizer.eos_token_id is None:
        tokenizer.eos_token_id = eos_token_id

    return tokenizer


def get_vocab_size(tokenizer: PreTrainedTokenizerFast) -> int:
    return tokenizer._tokenizer.get_vocab_size(with_added_tokens=True)