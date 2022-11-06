from argparse import ArgumentParser, Namespace
from fasttext.FastText import unsupervised_default

from graph_coder.embedding.graph_coder_embedder import GraphCoderEmbedder
from graph_coder.utils.cli import expand_user

default_overwrites = {
    "dim": 512,
}


def setup_parser(parser: ArgumentParser):
    parser.add_argument(
        "--data", type=expand_user, default="~/git-py/processed/ft_data"
    )
    parser.add_argument("--output", type=expand_user, default="~/git-py/embedder.bin")
    for k, v in unsupervised_default.items():
        parser.add_argument(
            f"--{k}", default=default_overwrites[k] if k in default_overwrites else v
        )


def train(args: Namespace):
    dict_args = vars(args)
    data = dict_args.pop("data")
    output = dict_args.pop("output")

    model = GraphCoderEmbedder.train_unsupervised(str(data), **dict_args)
    model.save(output)


def main(**kwargs):
    parser = ArgumentParser()
    setup_parser(parser)

    args = parser.parse_args()

    for k, v in kwargs.items():
        setattr(args, k, v)

    train(args)


if __name__ == "__main__":
    main()
