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
import click
from catalyst.contrib.scripts.run import run_from_config
from catalyst.registry import REGISTRY

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


@click.command()
@click.option("--config", default="configs/small.yaml", help="Path to config file")
def main(config: str):
    run_from_config(configs=[config])


if __name__ == "__main__":
    main()
