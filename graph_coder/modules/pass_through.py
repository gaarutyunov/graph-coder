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

import torch
from deepspeed.runtime.pipe import LayerSpec
from torch import nn


class PassThroughLayer(nn.Module):
    def __init__(
        self,
        inner: nn.Module,
        out: str,
        inp: typing.Optional[typing.List[str]] = None,
        callback: typing.Optional[typing.Callable] = None,
        args_getter: typing.Optional[typing.Callable] = None,
    ):
        super().__init__()
        self.inner = inner
        self.inp = inp
        self.out = out
        self.args_getter = args_getter or self._default_args_getter
        self.callback = callback

    def _default_args_getter(self, **kwargs):
        assert self.inp is not None, "input keys are not specified"
        return [kwargs[k] for k in self.inp]

    def forward(
        self,
        **kwargs: typing.Union[
            torch.Tensor, typing.List[torch.Tensor], typing.Dict[str, torch.Tensor]
        ],
    ) -> typing.Dict[str, torch.Tensor]:
        args = self.args_getter(**kwargs)

        if isinstance(args, (list, tuple)):
            res = self.inner(*args)
        elif isinstance(args, dict):
            res = self.inner(**args)
        else:
            res = self.inner(args)

        if self.callback is not None:
            res = self.callback(res, **kwargs)
        kwargs[self.out] = res

        return kwargs


class PassThroughLayerSpec(LayerSpec):
    def __init__(
        self,
        typename,
        out: str,
        inp: typing.Optional[typing.List[str]] = None,
        callback: typing.Optional[typing.Callable] = None,
        args_getter: typing.Optional[typing.Callable] = None,
        *module_args,
        **module_kwargs,
    ):
        super().__init__(typename, *module_args, **module_kwargs)
        self.out = out
        self.inp = inp
        self.args_getter = args_getter
        self.callback = callback

    def build(self, log=False):
        return PassThroughLayer(
            super().build(log),
            self.out,
            self.inp,
            self.callback,
            self.args_getter,
        )
