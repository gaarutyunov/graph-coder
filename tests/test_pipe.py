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
import random

import torch

from graph_coder.pipe import Layers, pipe_wrap, PipeModule
from torch import nn
from torch.utils.data import DataLoader


def collate_fn(items):
    a = []
    b = []
    for i in items:
        a.append(i[0].unsqueeze(0))
        b.append(i[1].unsqueeze(0))
    return tuple([torch.cat(a, dim=1), torch.cat(a, dim=1)]), torch.randn(1, 1)


def test_pipe():
    class TestModule(nn.Module):
        def forward(self, *args):
            x = args[0]

            return x * 2, *args[1:]

    class TestModulePipe(PipeModule):
        @pipe_wrap
        def to_layers(self) -> Layers:
            return [TestModule(), TestModule()]

    loader = DataLoader(
        [
            (
                torch.randn(random.randint(1, 10), 8),
                torch.randn(random.randint(1, 10), 8),
            )
            for _ in range(10)
        ],
        batch_size=2,
        collate_fn=collate_fn,
    )
    wrapper = loader
    layers = TestModulePipe().to_layers()

    for i in wrapper:
        inp = i[0]
        assert isinstance(i, tuple)
        assert isinstance(inp, tuple)
        exp = inp[0] * 4
        x = inp
        for j, layer in enumerate(layers):
            if j == len(layers) - 1:
                x = layer(x, i[1])
            else:
                x = layer(x)

        assert torch.all(exp == x[0])
        assert torch.all(inp[1] == x[1])
