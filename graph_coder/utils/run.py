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

from typing import Optional

from catalyst.contrib.scripts.run import run_from_params

from graph_coder.config import ConfigBuilder


def run_model(
    root: str,
    name: Optional[str] = None,
    size: Optional[str] = None,
    arch: Optional[str] = None,
):
    """Run a model from a config directory with the specified name, size and arch.

    If root is a path to a file, it will be used as the config file."""
    experiment_params = ConfigBuilder(root, name, size, arch).load().build()

    run_from_params(experiment_params)
