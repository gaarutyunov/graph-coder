import pathlib
from argparse import ArgumentParser, Namespace

from tokenizers import trainers, AddedToken

from graph_coder.datasets.base import GraphCoderDatasetBase
from graph_coder.tokenizer.graph_coder_tokenizer import GraphCoderTokenizer
from graph_coder.ast.vocabulary import Vocabulary


def get_training_corpus(dataset: GraphCoderDatasetBase):
    for idx in range(dataset.len()):
        vocab: Vocabulary = dataset.vocab(idx)

        for word in vocab.key_value.values():
            yield word


def setup_parser(parser: ArgumentParser):
    parser.add_argument("--data", type=pathlib.Path, default="./data")
    parser.add_argument("--output", type=pathlib.Path, default="./vocab.json")
    parser.add_argument("--vocab-size", type=int, default=10000)


def train(args: Namespace):
    tokenizer = GraphCoderTokenizer()
    dataset = GraphCoderDatasetBase(args.data)
    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        special_tokens=[
            AddedToken(
                token, single_word=True, rstrip=False, lstrip=False, normalized=True
            )
            for token in ["[EMP]", "[UNK]"]
        ],
    )

    corpus = get_training_corpus(dataset)

    tokenizer.train_from_iterator(corpus, trainer=trainer)

    tokenizer.save(str(args.output), pretty=True)


if __name__ == "__main__":
    parser = ArgumentParser()
    setup_parser(parser)

    args = parser.parse_args()
    train(args)
