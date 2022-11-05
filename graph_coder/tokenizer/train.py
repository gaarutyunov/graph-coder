import pathlib
from argparse import ArgumentParser, Namespace

from tokenizers import trainers, AddedToken

from graph_coder.datasets.base import GraphCoderDatasetBase
from graph_coder.tokenizer.graph_coder_tokenizer import GraphCoderTokenizer
from graph_coder.ast.vocabulary import Vocabulary
from graph_coder.utils.cli import expand_user


def get_training_corpus(dataset: GraphCoderDatasetBase):
    for idx in range(dataset.len()):
        vocab: Vocabulary = dataset.vocab(idx)
        source = dataset.source(idx)
        yield source

        for v in vocab.key_value.values():
            yield v


def setup_parser(parser: ArgumentParser):
    parser.add_argument("--data", type=expand_user, default="~/git-py")
    parser.add_argument("--output", type=expand_user, default="~/git-py/vocab.json")
    parser.add_argument("--vocab-size", type=int, default=16000)


def train(args: Namespace):
    tokenizer = GraphCoderTokenizer()
    dataset = GraphCoderDatasetBase(args.data)
    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        special_tokens=[
            AddedToken(
                token, single_word=True, rstrip=False, lstrip=False, normalized=True
            )
            for token in [
                "[PAD]",
                "[EMP]",
                "[UNK]",
                "[CLS]",
                "[SEP]",
                "[MASK]",
                "[NEW]",
                "[TAB]",
                "[NET]",
            ]
        ],
    )

    corpus = get_training_corpus(dataset)
    tokenizer.train_from_iterator(corpus, trainer=trainer)

    tokenizer.save(str(args.output), pretty=True)


def main(**kwargs):
    parser = ArgumentParser()
    setup_parser(parser)

    args = parser.parse_args()

    for k, v in kwargs.items():
        setattr(args, k, v)

    train(args)


if __name__ == "__main__":
    main()
