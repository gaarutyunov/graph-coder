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
from typing import Callable, Union

from torch import nn


class ConditionalLayer(nn.Module):
    def __init__(self, inner: Union[nn.Module, Callable], condition: Callable) -> None:
        super().__init__()
        self.inner = inner
        self.condition = condition

    def forward(self, *args, **kwargs):
        if not self.condition(*args, **kwargs):
            return kwargs

        return self.inner(*args, **kwargs)
