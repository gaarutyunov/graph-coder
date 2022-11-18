from argparse import ArgumentParser, Namespace

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

from graph_coder.datasets.graph_coder_dataset import GraphCoderLightningDataset
from graph_coder.models.graph_coder import GraphCoder
from graph_coder.utils.activation import get_available_activation_fns
from graph_coder.utils.cli import expand_user


def setup_parser(parser: ArgumentParser):
    parser.add_argument("--remove-head", action="store_true", default=False)
    parser.add_argument(
        "--layernorm-style",
        type=str,
        default="postnorm",
        choices=["prenorm", "postnorm"],
    )
    parser.add_argument("--vocab-size", type=int, default=16000)
    parser.add_argument("--text-embed-size", type=int, default=512)
    parser.add_argument("--num-features", type=int, default=8)
    parser.add_argument(
        "--repr-mode", type=str, default="token", choices=["token", "embedding"]
    )

    parser.add_argument("--num-nodes", type=int, default=1024)
    parser.add_argument("--num-edges", type=int, default=2048)
    parser.add_argument("--num-in-degree", type=int, default=64)
    parser.add_argument("--num-out-degree", type=int, default=64)
    parser.add_argument("--num-spatial", type=int, default=64)
    parser.add_argument("--num-edge-dis", type=int, default=16)
    parser.add_argument("--data", type=expand_user, default="~/git-py")
    parser.add_argument(
        "--dropout", type=float, metavar="D", help="dropout prob", default=0.1
    )
    parser.add_argument(
        "--attention-dropout",
        type=float,
        metavar="D",
        help="dropout prob for attention weights",
        default=0.1,
    )
    parser.add_argument(
        "--act-dropout",
        type=float,
        metavar="D",
        help="dropout prob after activation in FFN",
        default=0.0,
    )

    parser.add_argument(
        "--encoder-ffn-embed-dim",
        type=int,
        metavar="N",
        help="encoder embedding dim for FFN",
        default=4096,
    )
    parser.add_argument(
        "--encoder-layers",
        type=int,
        metavar="N",
        help="num encoder layers",
        default=6,
    )
    parser.add_argument(
        "--encoder-attention-heads",
        type=int,
        metavar="N",
        help="num encoder attention heads",
        default=8,
    )
    parser.add_argument(
        "--encoder-embed-dim",
        type=int,
        metavar="N",
        help="encoder embedding dimension",
        default=1024,
    )
    parser.add_argument(
        "--share-encoder-input-output-embed",
        action="store_true",
        help="share encoder input and output embeddings",
        default=False,
    )

    parser.add_argument(
        "--lap-node-id",
        action="store_true",
        help="use Laplacian eigenvector node identifiers",
        default=True,
    )
    parser.add_argument(
        "--lap-node-id-k",
        type=int,
        metavar="N",
        help="number of Laplacian eigenvectors to use, from smallest eigenvalues",
        default=8,
    )
    parser.add_argument(
        "--lap-node-id-sign-flip",
        action="store_true",
        help="randomly flip the signs of eigvecs",
        default=False,
    )
    parser.add_argument(
        "--lap-node-id-eig-dropout",
        type=float,
        metavar="D",
        help="dropout prob for Lap eigvecs",
        default=0.0,
    )
    parser.add_argument(
        "--type-id",
        action="store_true",
        help="use type identifiers",
        default=False,
    )
    parser.add_argument(
        "--graph-id",
        action="store_true",
        help="use graph identifier",
        default=False,
    )
    parser.add_argument(
        "--null-id",
        action="store_true",
        help="use null identifier",
        default=False,
    )

    parser.add_argument(
        "--stochastic-depth",
        action="store_true",
        help="use stochastic depth regularizer",
        default=False,
    )

    parser.add_argument(
        "--performer",
        action="store_true",
        help="linearized self-attention with Performer kernel",
        default=False,
    )
    parser.add_argument(
        "--performer-nb-features",
        type=int,
        metavar="N",
        help="number of random features for Performer, defaults to (d*log(d)) where d is head dim",
        default=None,
    )
    parser.add_argument(
        "--performer-feature-redraw-interval",
        type=int,
        metavar="N",
        help="how frequently to redraw the projection matrix for Performer",
        default=1000,
    )
    parser.add_argument(
        "--performer-generalized-attention",
        action="store_true",
        help="defaults to softmax approximation, but can be set to True for generalized attention",
        default=False,
    )
    parser.add_argument(
        "--performer-finetune",
        action="store_true",
        help="load softmax checkpoint and fine-tune with performer",
        default=False,
    )

    parser.add_argument(
        "--activation-fn",
        choices=get_available_activation_fns(),
        help="activation to use",
        default="gelu",
    )
    parser.add_argument(
        "--encoder-normalize-before",
        action="store_true",
        help="apply layernorm before encoder",
        default=True,
    )
    parser.add_argument(
        "--decoder-normalize-before",
        action="store_true",
        help="apply layernorm before decoder",
        default=True,
    )
    parser.add_argument(
        "--return-attention",
        action="store_true",
        help="obtain attention maps from all layers",
        default=False,
    )
    parser.add_argument(
        "--max-iters",
        type=int,
        metavar="N",
        help="max number of iterations for training",
        default=100000,
    )
    parser.add_argument(
        "--lr",
        type=float,
        metavar="LR",
        help="learning rate",
        default=0.0001,
    )
    parser.add_argument(
        "--warmup",
        type=float,
        metavar="N",
        help="warmup steps",
        default=10000,
    )
    parser.add_argument(
        "--batch-size",
        help="batch size",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--num-workers",
        help="number of workers",
        default=4,
    )
    parser.add_argument(
        "--test-size",
        help="test size",
        default=0.1,
    )
    parser.add_argument(
        "--val-size",
        help="val size",
        default=0.05,
    )


def train(args: Namespace):
    model = GraphCoder(args)
    logger = TensorBoardLogger("logs", name="graphcoder")
    trainer: Trainer = Trainer.from_argparse_args(args, logger=logger)
    dm = GraphCoderLightningDataset(
        args.data,
        num_nodes=args.num_nodes,
        num_edges=args.num_edges,
        batch_size=args.batch_size,
        test_size=args.test_size,
        val_size=args.val_size,
        num_workers=args.num_workers,
    )

    trainer.fit(model, dm)


def main(**kwargs):
    parser = ArgumentParser()
    setup_parser(parser)
    parser = Trainer.add_argparse_args(parser)

    args = parser.parse_args()

    for k, v in kwargs.items():
        setattr(args, k, v)

    train(args)


if __name__ == "__main__":
    main(accelerator="cpu", strategy="single_device", batch_size=2)
