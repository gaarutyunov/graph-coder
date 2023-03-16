#  Copyright 2023 German Arutyunov
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
import typing
from typing import Callable, Union

import torch
from torch import nn

from .types import Kwargs


class PassThroughLayer(nn.Module):
    """Layer that calls inner module with args from `inp` and returns them into `out`.

    Args:
        inner: Inner module.
        inp: Indexes of args to pass to inner module.
        out: Indexes of args to insert the result into. If None, the result is appended.
        callback: Callback to call after inner module.
        args_getter: Function to get args from `*args`. By default, it returns `args[inp]`.

    Returns:
        Tuple of args with the result inserted into `out`.
    """

    def __init__(
        self,
        inner: nn.Module,
        inp: typing.Optional[typing.Union[typing.List[int], int]] = None,
        out: typing.Optional[typing.Union[typing.List[int], int]] = None,
        callback: typing.Optional[typing.Callable] = None,
        args_getter: typing.Optional[typing.Callable] = None,
    ):
        super().__init__()
        self.inner = inner
        self.inp = inp
        self.out = out
        self.args_getter: Callable = args_getter or self._default_args_getter
        self.callback = callback

    def _default_args_getter(self, *args):
        if isinstance(self.inp, int):
            return [args[self.inp]]
        return [args[k] for k in self.inp]

    def forward(
        self,
        *args: Kwargs,
    ) -> typing.Tuple[Kwargs, ...]:
        selected_args = self.args_getter(*args)

        if isinstance(selected_args, (list, tuple)):
            res = self.inner(*selected_args)
        else:
            res = self.inner(selected_args)

        is_single = not isinstance(res, (list, tuple))
        if is_single:
            res = (res,)

        if self.callback is not None:
            res = self.callback(*res, *args)

            is_single = not isinstance(res, (list, tuple))
            if is_single:
                res = (res,)

        if self.out is None:
            return tuple(
                [
                    *args,
                    *res,
                ]
            )
        elif isinstance(self.out, int):
            largs = list(args)
            if len(args) <= abs(self.out):
                largs.append(res[0])
            else:
                largs[self.out] = res[0]
            return tuple(largs)

        largs = list(args)

        for i in range(len(res)):
            if len(args) <= abs(self.out[i]):
                largs.append(res[i])
            else:
                largs[self.out[i]] = res[i]

        return tuple(largs)

    def __repr__(self):
        return self.inner.__repr__()


class ConditionalLayer(nn.Module):
    """Layer that calls inner module conditionally.

    Args:
        inner: Inner module.
        condition: Condition to call inner module.
    """

    def __init__(self, inner: Union[nn.Module, Callable], condition: Callable) -> None:
        super().__init__()
        self.inner = inner
        self.condition = condition

    def forward(self, *args, **kwargs):
        if not self.condition(*args, **kwargs):
            return args

        return self.inner(*args, **kwargs)

    def __repr__(self):
        return self.inner.__repr__()


class RemoveArgsLayer(nn.Module):
    """Layer that removes args from `*args` by indexes."""

    def __init__(self, *idx: int) -> None:
        super().__init__()
        self.idx = idx

    def forward(self, *args, **kwargs):
        largs = list(args)
        for i in self.idx:
            del largs[i]

        return tuple(largs)


class CloneLayer(nn.Module):
    """Layer that clones input."""

    def forward(self, x):
        return torch.clone(x)


class ReorderLayer(nn.Module):
    """Layer that reorders args in `*args` by indexes."""

    def __init__(self, *idx: int) -> None:
        super().__init__()
        self.idx = idx

    def forward(self, *args, **kwargs):
        return tuple([args[i] for i in self.idx])