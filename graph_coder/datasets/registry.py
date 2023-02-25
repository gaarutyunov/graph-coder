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
import functools
from typing import Type, TypeVar, Callable

from .base import BaseDataset

_DATASETS = {}

T = TypeVar("T", bound=Type[BaseDataset])


def register(name: str) -> Callable[[T], T]:
    def wrapper(cls: T) -> T:
        if name in _DATASETS:
            raise ValueError(f"Dataset {name} already registered")
        _DATASETS[name] = cls
        return cls

    return wrapper


def get(name: str) -> T:
    if name not in _DATASETS:
        raise ValueError(f"Dataset {name} not registered")
    return _DATASETS[name]
