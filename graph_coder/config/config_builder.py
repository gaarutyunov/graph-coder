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
from collections import OrderedDict
from os import PathLike
from pathlib import Path
from typing import Union, Optional

from catalyst.contrib.scripts.run import process_configs
from catalyst.registry import REGISTRY


class ConfigBuilder:
    def __init__(
        self,
        root: Union[str, PathLike],
        name: Optional[str] = None,
        size: Optional[str] = None,
        arch: Optional[str] = None,
        common_dir: str = "_common",
    ):
        self.root = Path(root).expanduser()
        self.configs = OrderedDict()

        if self.root.is_file():
            return
        self.name = name
        self.size = size
        self.arch = arch
        self.common_dir = common_dir

        self.dirs = [self.root]

        if name is not None:
            assert (self.root / name).exists(), f"Model {name} does not exist"
            self.dirs.append(self.root / name)

        if size is not None:
            assert (
                self.root / name / size
            ).exists(), f"Model {name} does not have size {size}"
            self.dirs.append(self.root / name / size)

        if arch is not None:
            assert (
                self.root / name / size / arch
            ).exists(), (
                f"Model {name} does not have size {size} and architecture {arch}"
            )
            self.dirs.append(self.root / name / size / arch)

        self.dirs.reverse()

    def load(self) -> "ConfigBuilder":
        if self.root.is_file():
            self._add(self.root)
            return self
        for d in self.dirs:
            if (d / self.common_dir).exists():
                for cfg in (d / self.common_dir).iterdir():
                    self._add(cfg)
            for cfg in d.iterdir():
                self._add(cfg)

        return self

    def _add(self, config: Path):
        if not config.is_file() and not config.suffix == ".yaml":
            return
        elif config.stem in self.configs:
            return

        self.configs[config.stem] = str(config.relative_to(self.root))

    def build(self):
        if self.root.is_file():
            config = process_configs([str(self.root)])
        else:
            config = process_configs(
                (str(self.root / path) for path in self.configs.values())
            )
        experiment_params = REGISTRY.get_from_params(**config)

        return experiment_params
