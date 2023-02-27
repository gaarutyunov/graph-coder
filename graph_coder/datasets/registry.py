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
from typing import TypeVar, Callable, Generic, Dict, Any

T = TypeVar("T")


class Registry(Generic[T]):
    def __init__(self) -> None:
        super().__init__()
        self._obj: Dict[str, T] = {}

    def register(self, name: str, obj: T):
        if name in self._obj:
            raise ValueError(f"Dataset {name} already registered")
        self._obj[name] = obj

    def get(self, name: str) -> T:
        if name not in self._obj:
            raise ValueError(f"Dataset {name} not registered")
        return self._obj[name]


_DEFAULT_REGISTRY: Registry[Any] = Registry()


def register(name: str) -> Callable[[T], T]:
    def wrapper(cls: T) -> T:
        _DEFAULT_REGISTRY.register(name, cls)
        return cls

    return wrapper


get = _DEFAULT_REGISTRY.get
