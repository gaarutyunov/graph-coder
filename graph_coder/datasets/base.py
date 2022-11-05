import ast
import json
import pathlib
import typing
from dataclasses import asdict
from functools import lru_cache
from lib2to3.refactor import RefactoringTool, get_fixers_from_package
from typing import Callable, Optional, Union, List, Tuple

import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.utils import from_networkx
from tqdm.auto import tqdm

from graph_coder.ast import Context, code_to_graph, Vocabulary
from graph_coder.ast.transformers.docstring_remover import DocstringRemover
from graph_coder.data.features import data_to_text

refactor = RefactoringTool(fixer_names=get_fixers_from_package("lib2to3.fixes"))
docstring_remover = DocstringRemover()


def pre_transform(
    data: str,
) -> typing.List[Context]:
    return code_to_graph(data)


def to_file_name(path: pathlib.Path) -> str:
    return path.name


def to_rel_path(root: typing.Union[pathlib.Path, str]) -> Callable[[pathlib.Path], str]:
    def to_rel_path_(path: pathlib.Path) -> str:
        return str(path.relative_to(pathlib.Path(root)))

    return to_rel_path_


class GraphCoderDatasetBase(Dataset):
    def __init__(self, root: Optional[str] = None):
        super().__init__(
            pathlib.Path(root).expanduser().__str__(), None, pre_transform, None
        )

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        return list(
            map(to_rel_path(self.raw_dir), pathlib.Path(self.raw_dir).rglob("*.py"))
        )

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        return list(
            map(
                to_rel_path(self.processed_dir),
                pathlib.Path(self.processed_dir).glob("**/data.pt"),
            )
        )

    def process(self):
        i = 0
        desc = "Processing dataset ({}/{} graphs)"
        p_bar = tqdm(self.raw_paths)

        for path in p_bar:
            p_bar.set_description(desc.format("?", "?"))

            file_stat = pathlib.Path(path).stat()
            if file_stat.st_size == 0:
                p_bar.update(1)
                continue

            with open(path) as f:
                source = f.read()
                if len(source.strip()) == 0:
                    p_bar.update(1)
                    continue
            try:
                source = str(refactor.refactor_string(source + "\n", path))
            except Exception as e:
                with open(pathlib.Path(self.processed_dir, "log.txt"), "a") as ff:
                    print("Failed to refactor %s: %s\n" % (path, e), file=ff)

            try:
                ctxs: typing.List[Context] = self.pre_transform(source)
                ctxs = [ctx for ctx in ctxs if ctx.g.number_of_nodes() > 9]
                if len(ctxs) == 0:
                    p_bar.update(1)
                    continue
            except Exception as e:
                with open(pathlib.Path(self.processed_dir, "log.txt"), "a") as ff:
                    print("Failed to parse %s: %s\n" % (path, e), file=ff)
                    p_bar.update(1)
                continue
            j = 1

            for ctx in ctxs:
                vocab = ctx.v
                data = from_networkx(ctx.g, all, all)
                src = ast.unparse(docstring_remover.visit(ast.parse(ctx.src)))

                p_bar.set_description(desc=desc.format(j, len(ctxs)))
                (pathlib.Path(self.processed_dir) / str(i)).mkdir(
                    parents=True, exist_ok=True
                )
                torch.save(data, pathlib.Path(self.processed_dir, str(i), f"data.pt"))
                with open(
                    pathlib.Path(self.processed_dir, str(i), f"vocab.json"), "w"
                ) as ff:
                    json.dump(asdict(vocab), ff)
                with open(
                    pathlib.Path(self.processed_dir, str(i), f"source.py"), "w"
                ) as ff:
                    ff.write(src)
                with open(pathlib.Path(self.processed_dir, "ft_data"), "a") as ff:
                    ff.write(data_to_text(data, vocab.value_key.keys(), src) + "\n")
                i += 1
                j += 1

            p_bar.update(1)

    def len(self) -> int:
        return len(self.processed_file_names)

    @lru_cache(16)
    def get(self, idx: int) -> Data:
        return torch.load(pathlib.Path(self.processed_dir, str(idx), f"data.pt"))

    @lru_cache(16)
    def get_transformed(self, idx: int) -> Data:
        return torch.load(
            pathlib.Path(self.processed_dir, str(idx), f"data_transformed.pt")
        )

    def save_transformed(self, idx: int, data: Data):
        torch.save(
            data, pathlib.Path(self.processed_dir, str(idx), f"data_transformed.pt")
        )

    @lru_cache(16)
    def vocab(self, idx: int) -> Vocabulary:
        with open(pathlib.Path(self.processed_dir, str(idx), f"vocab.json")) as f:
            return Vocabulary(**json.load(f))

    @lru_cache(16)
    def source(self, idx: int) -> str:
        with open(pathlib.Path(self.processed_dir, str(idx), f"source.py")) as f:
            return f.read()
