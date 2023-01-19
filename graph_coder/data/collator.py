import torch
from torch.nn.utils.rnn import pad_sequence
from torch.functional import F
from transformers import PreTrainedTokenizerFast

from .batch import GraphCoderBatch
from .ast import AstExample
from .algos import lap_eig


def pad(
    batch: list[str],
    num: list[int],
    tokenizer: PreTrainedTokenizerFast,
    max_length: int = 64,
) -> dict[str, torch.Tensor]:
    """Pad a batch of strings restoring the original packs."""
    encoding = tokenizer(
        batch,
        padding="max_length",
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    ).convert_to_tensors()
    inputs_ids = []
    attention_mask = []

    start = 0

    for n in num:
        end = start + n
        inputs_ids.append(encoding["input_ids"][start:end])
        attention_mask.append(encoding["attention_mask"][start:end])
        start = end

    return {
        "input_ids": pad_sequence(
            inputs_ids, batch_first=True, padding_value=tokenizer.pad_token_id
        ),
        "attention_mask": pad_sequence(
            attention_mask, batch_first=True, padding_value=False
        ),
    }


@torch.no_grad()
def collate_ast(
    batch: list[AstExample], tokenizer: PreTrainedTokenizerFast, max_length: int = 64
) -> GraphCoderBatch:
    """Collate a batch of examples into a batch of tensors."""
    idx = []
    edge_index = []
    edge_data = []
    node_data = []
    node_num = []
    edge_num = []
    sources = []
    docstrings = []
    lap_eigvals = []
    lap_eigvecs = []

    for data in batch:
        sources.append(data.source)
        docstrings.append(data.docstring)

        data = data.graph
        lap_eigval, lap_eigvec = lap_eig(data.edge_index, len(data.x))
        lap_eigval = lap_eigval[None, :].expand_as(lap_eigvec)
        lap_eigvals.append(lap_eigval)
        lap_eigvecs.append(lap_eigvec)

        idx.append(data.idx)
        edge_index.append(data.edge_index)

        node_data.extend(data.x)
        edge_data.extend(data.edge_attr)

        node_num.append(len(data.x))
        edge_num.append(len(data.edge_attr))

    max_n = max(node_num)

    return GraphCoderBatch(
        idx=torch.LongTensor(idx),
        edge_index=torch.LongTensor(torch.cat(edge_index, dim=1)),
        node_num=torch.LongTensor(node_num),
        edge_num=torch.LongTensor(edge_num),
        lap_eigval=torch.cat(
            [F.pad(i, (0, max_n - i.size(1)), value=float("0")) for i in lap_eigvals]
        ),
        lap_eigvec=torch.cat(
            [F.pad(i, (0, max_n - i.size(1)), value=float("0")) for i in lap_eigvecs]
        ),
        source_=tokenizer(
            sources, padding=True, return_tensors="pt", return_attention_mask=True
        ).convert_to_tensors(),
        docstring_=tokenizer(
            docstrings, padding=True, return_tensors="pt", return_attention_mask=True
        ).convert_to_tensors(),
        edge_data_=pad(edge_data, edge_num, tokenizer, max_length=max_length),
        node_data_=pad(node_data, node_num, tokenizer, max_length=max_length),
    )
