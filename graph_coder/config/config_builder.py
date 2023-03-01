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
import pathlib
from collections import OrderedDict
from os import PathLike
from pathlib import Path
from typing import Union, Optional, Dict, Any, List

import yaml
from catalyst.registry import REGISTRY
from torchgen.utils import OrderedSet

from .utils import process_configs


class ConfigBuilder:
    def __init__(
        self,
        root: Union[str, PathLike],
        *args: str,
        common_dir: str = "_common",
        order_file: str = "_order",
    ):
        self.root = Path(root).expanduser()
        self.order_file = order_file
        self.configs: Dict[str, Any] = OrderedDict()

        if self.root.is_file():
            return
        self.common_dir = common_dir

        self.dirs: List[Path] = [self.root]

        for arg in args:
            path = self.dirs[-1] / arg
            assert path.exists(), f"Model in {path} does not exist"
            self.dirs.append(path)

        self.dirs.reverse()

    def load(self) -> "ConfigBuilder":
        if self.root.is_file():
            self._add(self.root)
            return self
        processed_dirs = []
        order = OrderedSet()
        order_from_common = False

        def read_order(dr: pathlib.Path, from_common: bool):
            nonlocal order, order_from_common
            order_file = dr / self.order_file
            if (
                not order_file.exists()
                or len(order.storage) != 0
                and (not order_from_common or from_common and order_from_common)
            ):
                return
            with open(order_file, mode="r") as f:
                order_from_common = from_common
                order.update(OrderedSet(f.read().splitlines(False)))

        for i, d in enumerate(self.dirs):
            read_order(d, False)
            for cfg in sorted(d.iterdir()):
                self._add(cfg)
            common_dir = d / self.common_dir
            if not common_dir.exists():
                common_dir = d.parent / self.common_dir
            if common_dir.exists() and str(common_dir) not in processed_dirs:
                read_order(common_dir, True)
                for cfg in sorted(common_dir.iterdir()):
                    self._add(cfg, from_common=True)
                processed_dirs.append(str(common_dir))

        new_configs = OrderedDict()

        order.update(OrderedSet(self.configs.keys()))

        for key in order:
            if key not in self.configs:
                continue
            new_configs[key] = self.configs[key]

        self.configs = new_configs

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
            parts: List[str] = []
            parts.extend([dr.stem for dr in reversed(self.dirs[:-1])])
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
        if not config.is_file() or not config.suffix == ".yaml":
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
