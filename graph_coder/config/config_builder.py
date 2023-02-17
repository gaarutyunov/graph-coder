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
from typing import Union, Optional, Dict, Any

import yaml
from catalyst.registry import REGISTRY

from .utils import process_configs


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
        self.configs: Dict[str, Any] = OrderedDict()

        if self.root.is_file():
            return
        self.name = name
        self.size = size
        self.arch = arch
        self.common_dir = common_dir

        self.dirs = [self.root]

        if name is not None:
            path = self.root / str(name)
            assert path.exists(), f"Model {name} does not exist"
            self.dirs.append(path)

        if size is not None:
            path = self.root / str(name) / str(size)
            assert path.exists(), f"Model {name} does not have size {size}"
            self.dirs.append(path)

        if arch is not None:
            path = self.root / str(name) / str(size) / str(arch)
            assert (
                path
            ).exists(), (
                f"Model {name} does not have size {size} and architecture {arch}"
            )
            self.dirs.append(path)

        self.dirs.reverse()

    def load(self) -> "ConfigBuilder":
        if self.root.is_file():
            self._add(self.root)
            return self
        for i, d in enumerate(self.dirs):
            if i == 0 and len(self.dirs) < 4:
                common_dir = d / self.common_dir
            else:
                common_dir = d.parent / self.common_dir
            if common_dir.exists():
                for cfg in sorted(common_dir.iterdir()):
                    self._add(cfg, from_common=True)
            for cfg in sorted(d.iterdir()):
                self._add(cfg)

        return self

    def save(self, path: Optional[Union[str, PathLike]] = None):
        if self.root.is_file():
            return
        root: Optional[Path] = None

        if path is not None:
            path = Path(path).expanduser()
            if path.is_dir():
                root = path

        if root is None:
            root = self.root

        if path is None or path.is_dir():
            parts = []
            if self.name is not None:
                parts.append(self.name)
            if self.size is not None:
                parts.append(self.size)
            if self.arch is not None:
                parts.append(self.arch)
            if len(parts) == 0:
                parts.append("config")

            path = root / ("_".join(parts) + ".yaml")

        configs = self._process_configs(ordered=True, load_ordered=False)

        with open(path, "a") as f:
            f.truncate()
            for key, config in configs.items():
                yaml.dump({key: config}, f)

    def build(self):
        config = self._process_configs()
        REGISTRY._vars_dict = {}
        experiment_params = REGISTRY.get_from_params(**config)

        return experiment_params

    def _process_configs(
        self, ordered: bool = True, load_ordered: bool = True
    ) -> Dict[str, Any]:
        if self.root.is_file():
            configs = [str(self.root)]
        else:
            configs = [str(self.root / cfg["path"]) for cfg in self.configs.values()]

        return process_configs(configs, ordered=ordered, load_ordered=load_ordered)

    def _add(self, config: Path, from_common: bool = False):
        if not config.is_file() and not config.suffix == ".yaml":
            return
        elif config.stem in self.configs and (
            not self.configs[config.stem]["from_common"]
            or from_common
            and self.configs[config.stem]["from_common"]
        ):
            return

        self.configs[config.stem] = {
            "path": str(config.relative_to(self.root)),
            "from_common": from_common,
        }
