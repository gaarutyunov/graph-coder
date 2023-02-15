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

import click

from graph_coder.config import ConfigBuilder


@click.command()
@click.option("--root", default="configs", help="Config directory")
@click.option("--out", default="configs", help="Output directory")
@click.option("--name", default="generator", help="Model name")
@click.option("--size", default="small", help="Model size")
@click.option("--arch", default=None, help="Model architecture")
def main(
    root: str,
    out: str,
    name: Optional[str] = None,
    size: Optional[str] = None,
    arch: Optional[str] = None,
):
    ConfigBuilder(root, name, size, arch).load().save(out)


if __name__ == "__main__":
    main()
