#  Copyright 2023 German Arutyunov
#
#     Licensed under the Apache License, Version 2.0 (the "License");
#     you may not use this file except in compliance with the License.
#     You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#     Unless required by applicable law or agreed to in writing, software
#     distributed under the License is distributed on an "AS IS" BASIS,
#     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#     See the License for the specific language governing permissions and
#     limitations under the License.

import dataclasses

from typing import List, Tuple

from .base import BaseExample


@dataclasses.dataclass
class AstData:
    x: List[str]
    edge_attr: List[str]
    edge_index: List[Tuple[int, int]]
    idx: int


@dataclasses.dataclass
class AstExample(BaseExample[AstData]):
    """Example for AST dataset."""

    pass
