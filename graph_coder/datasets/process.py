from argparse import ArgumentParser

from graph_coder.datasets.base import GraphCoderDatasetBase
from graph_coder.utils.cli import expand_user


def setup_parser(parser):
    parser.add_argument("--data", type=expand_user, default="~/git-py")


def process(args):
    dataset = GraphCoderDatasetBase(args.data)
    print(dataset.len())


def main(**kwargs):
    parser = ArgumentParser()
    setup_parser(parser)

    args = parser.parse_args()

    for k, v in kwargs.items():
        setattr(args, k, v)

    process(args)


if __name__ == "__main__":
    main()
