import pathlib
from argparse import ArgumentParser

import numpy as np
import torch
from tqdm.auto import trange

from graph_coder.datasets.base import GraphCoderDatasetBase
from graph_coder.embedding.graph_coder_embedder import GraphCoderEmbedder


def setup_parser(parser: ArgumentParser):
    parser.add_argument("--data", type=pathlib.Path, default="~/git-py")
    parser.add_argument(
        "--embedder", type=pathlib.Path, default="~/git-py/embedder.bin"
    )
    parser.add_argument("--embed_dim", type=int, default=1024)


def transform(args):
    embedder = GraphCoderEmbedder.from_pretrained(args.embedder)
    dataset = GraphCoderDatasetBase(args.data)

    for idx in trange(dataset.len(), desc="Transforming dataset"):
        data = dataset.get(idx)
        vocab = dataset.vocab(idx)
        source = dataset.source(idx)
        embeddings = np.zeros((len(vocab) - 1, args.embed_dim))

        i = 0
        for k, v in vocab.key_value.items():
            if vocab.value_types[k] == "special":
                continue
            elif vocab.value_types[k] == "word":
                embeddings[i] = embedder.get_word_vector(v)
            elif vocab.value_types[k] == "sentence":
                embeddings[i] = embedder.get_sentence_vector(v)
            i += 1

        embeddings = torch.FloatTensor(embeddings)

        x = torch.zeros((*data.x.shape, args.embed_dim))

        for i, node in enumerate(data.x):
            for j, feat in enumerate(node):
                if feat == 0:
                    continue
                x[i, j] = embeddings[feat - 1]

        edge_attr = torch.zeros((*data.edge_attr.shape, args.embed_dim))

        for i, edge in enumerate(data.edge_attr):
            for j, feat in enumerate(edge):
                if feat == 0:
                    continue
                edge_attr[i, j] = embeddings[feat - 1]

        data.x = x
        data.edge_attr = edge_attr
        data.y = embedder.get_sentence_vector(source)

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
