import typing

from torch import nn

activations = {
    "gelu": nn.GELU(),
    "relu": nn.ReLU(),
}


def get_activation_fn(name: str):
    return activations.get(name)


def get_available_activation_fns() -> typing.List[str]:
    return list(activations.keys())
