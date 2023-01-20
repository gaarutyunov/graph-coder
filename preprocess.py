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

from graph_coder.datasets import AstDataset
from graph_coder.utils import get_pretrained_tokenizer


@click.command()
@click.option("--root", default="~/git-py/raw/python", help="Root data directory")
def main(root: str):
    tokenizer = get_pretrained_tokenizer("EleutherAI/gpt-neox-20b")
    AstDataset(root=root, introspect=True, tokenizer=tokenizer)


if __name__ == "__main__":
    main()
