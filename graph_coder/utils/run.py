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
from pathlib import Path

from catalyst.contrib.scripts.run import run_from_params

from graph_coder.config import ConfigBuilder


def run_model(root: str, *args: str):
    """Run a model from a config `root` directory with and path parts specified by `args`.

    If `root` is a path to a file, it will be used as the config file."""
    builder = ConfigBuilder(root, *args)
    experiment_params = builder.load().build()

    if "log_path" in experiment_params:
        builder.save(Path(experiment_params["log_path"]).expanduser() / "config.yaml")

    run_from_params(experiment_params)
