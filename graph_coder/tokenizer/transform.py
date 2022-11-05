import typing
from argparse import ArgumentParser
import torch
from tokenizers import Encoding
from tqdm.auto import trange

from graph_coder.data.features import replace_special_tokens
from graph_coder.datasets.base import GraphCoderDatasetBase
from graph_coder.tokenizer.graph_coder_tokenizer import GraphCoderTokenizer
from graph_coder.utils.cli import expand_user


def setup_parser(parser: ArgumentParser):
    parser.add_argument("--data", type=expand_user, default="~/git-py")
    parser.add_argument("--vocab", type=expand_user, default="~/git-py/vocab.json")


def transform(args):
    tokenizer = GraphCoderTokenizer.from_file(str(args.vocab))
    tokenizer.enable_padding()

    dataset = GraphCoderDatasetBase(args.data)

    for idx in trange(dataset.len(), desc="Transforming dataset"):
        data = dataset.get(idx)
        vocab = dataset.vocab(idx)
        source = replace_special_tokens(dataset.source(idx))
        batch: typing.List[Encoding] = tokenizer.encode_batch(
            list(vocab.key_value.values())
        )
        size = len(batch[0].ids)

        x = torch.zeros((*data.x.shape, size), dtype=torch.long)

        for i, node in enumerate(data.x):
            for j, feat in enumerate(node):
                x[i, j] = torch.LongTensor(batch[feat].ids)

        edge_attr = torch.zeros((*data.edge_attr.shape, size), dtype=torch.long)

        for i, edge in enumerate(data.edge_attr):
            for j, feat in enumerate(edge):
                edge_attr[i, j] = torch.LongTensor(batch[feat].ids)

        data.x = x
        data.edge_attr = edge_attr
        data.y = torch.LongTensor(tokenizer.encode(source).ids)

        dataset.save_transformed(idx, data)


def main(**kwargs):
    parser = ArgumentParser()
    setup_parser(parser)

    args = parser.parse_args()

    for k, v in kwargs.items():
        setattr(args, k, v)

    transform(args)


if __name__ == "__main__":
    main()
