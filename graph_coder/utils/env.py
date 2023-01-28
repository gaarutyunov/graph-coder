#  Copyright 2023 German Arutyunov
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import sys
from pathlib import Path
from typing import Optional

from catalyst.contrib.scripts.run import run_from_params
from catalyst.registry import REGISTRY

from graph_coder.config import ConfigBuilder


def check_ipython() -> bool:
    try:
        get_ipython = sys.modules["IPython"].get_ipython

        if get_ipython is None:
            return False
        else:
            _ = str(get_ipython())
            return True
    except:
        return False


def run_async(func):
    import asyncio

    if check_ipython():
        import nest_asyncio

        nest_asyncio.apply()

    loop = asyncio.get_event_loop()
    loop.run_until_complete(func)


def _add_all_to_registry():
    """Add all graph-coder modules to registry."""
    import graph_coder.data
    import graph_coder.datasets
    import graph_coder.models
    import graph_coder.modules
    import graph_coder.runners
    import graph_coder.utils

    REGISTRY.add_from_module(graph_coder.data)
    REGISTRY.add_from_module(graph_coder.datasets)
    REGISTRY.add_from_module(graph_coder.models)
    REGISTRY.add_from_module(graph_coder.modules)
    REGISTRY.add_from_module(graph_coder.runners)
    REGISTRY.add_from_module(graph_coder.utils)


def run_model(
    root: str,
    name: Optional[str] = None,
    size: Optional[str] = None,
    arch: Optional[str] = None,
):
    """Run a model from a config directory with the specified name, size and arch.

    If root is a path to a file, it will be used as the config file."""
    _add_all_to_registry()
    configs_path = Path(root)
    experiment_params = None

    if configs_path.exists():
        parts = [name, size, arch]
        parts = [part for part in parts if part is not None]
        config_path = configs_path / ("_".join(parts) + ".yaml")
        if config_path.exists():
            experiment_params = ConfigBuilder(config_path).load().build()

    if experiment_params is None:
        experiment_params = ConfigBuilder(root, name, size, arch).load().build()

    run_from_params(experiment_params)
