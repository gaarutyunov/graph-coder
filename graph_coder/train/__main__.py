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
import warnings

import click

from typing import Optional

from graph_coder.utils import run_model


@click.command()
@click.option("--root", default="configs", help="Config directory")
@click.option("--name", default="generator", help="Model name")
@click.option("--size", default="small", help="Model size")
@click.option("--arch", default="performer", help="Model architecture")
def main(
    root: str,
    name: Optional[str] = None,
    size: Optional[str] = None,
    arch: Optional[str] = None,
):
    run_model(root, name, size, arch)


if __name__ == "__main__":
    warnings.filterwarnings("ignore", message=".*is a private function and will be deprecated.*", category=UserWarning)
    main()
