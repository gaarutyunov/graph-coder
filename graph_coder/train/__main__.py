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

from typing import List

import click

from graph_coder.utils import run_model


@click.command()
@click.option(
    "--root",
    default="configs",
    help="Config directory",
)
@click.argument("path", nargs=-1)
def main(
    root: str,
    path: List[str],
):
    """Run a model from a config `root` directory with and path parts specified by `PATH`.
    If `root` is a path to a file, it will be used as the config file."""
    run_model(root, *path)


if __name__ == "__main__":
    warnings.filterwarnings(
        "ignore",
        message=".*is a private function and will be deprecated.*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=".*the loss during ``runner.handle_batch``?.*",
        category=UserWarning,
    )
    main()
