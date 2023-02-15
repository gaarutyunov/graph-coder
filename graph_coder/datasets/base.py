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

import abc
import os
import pickle
import typing
from pathlib import Path

import aiofiles
import torch
from torch.utils.data import DataLoader, random_split, Dataset
from typing import Dict

from tqdm.auto import tqdm, trange

from graph_coder.logger import ILogger
from graph_coder.utils import run_async

T = typing.TypeVar("T")


class BaseDataset(Dataset, abc.ABC, typing.Generic[T]):
    def __init__(
        self,
        logger: ILogger,
        collate_fn: typing.Optional[typing.Callable] = None,
        random_seed: typing.Optional[int] = None,
        test_size: float = 0.2,
        val_size: float = 0.2,
        batch_size: int = 1,
        in_memory: bool = False,
        preprocess: bool = True,
    ):
        self._logger = logger
        self._loaders: Dict[str, DataLoader] = {}
        self.batch_size = batch_size
        self.collate_fn = collate_fn
        self.random_seed = random_seed
        self.val_size = val_size
        self.test_size = test_size
        self.in_memory = in_memory
        self.preprocess = preprocess
        self._cache: Dict[int, T] = {}
        self._is_processed: typing.Optional[bool] = None

    @abc.abstractmethod
    def __len__(self):
        pass

    @abc.abstractmethod
    def __getitem__(self, index: int) -> T:
        pass

    @property
    @abc.abstractmethod
    def processed_dir(self) -> typing.Union[os.PathLike, str]:
        pass

    @property
    def is_processed(self) -> bool:
        if self._is_processed is None:
            try:
                _ = next(iter(Path(self.processed_dir).iterdir()))
                self._is_processed = self._last_processed_idx == len(self) - 1
            except:
                self._is_processed = False

        return self._is_processed

    @property
    def is_loaded(self) -> bool:
        return len(self._cache) == len(self)

    @property
    def loaders(self) -> Dict[str, DataLoader]:
        return self._loaders

    @property
    def _last_processed_idx(self) -> int:
        return int(
            max(Path(self.processed_dir).iterdir(), key=lambda x: int(x.stem)).stem
        )

    @property
    def _last_loaded_idx(self) -> int:
        return len(self._cache) - 1

    def split(self):
        """Splits the dataset into train, test and validation sets using `random_seed`"""
        train_size = 1.0 - self.test_size - self.val_size
        datasets = random_split(
            self,
            [train_size, self.val_size, self.test_size],
            generator=torch.Generator().manual_seed(self.random_seed)
            if self.random_seed
            else None,
        )

        self._loaders = dict(
            zip(
                ["train", "val", "test"],
                [
                    DataLoader(
                        dataset, collate_fn=self.collate_fn, batch_size=self.batch_size
                    )
                    for dataset in datasets
                ],
            )
        )

    def process(self):
        """Preprocesses the dataset by getting each item and saving it into `processed_dir`"""
        if not self.is_processed:
            run_async(self._process())
            self._is_processed = True

    def load(self):
        """Loads the dataset into memory"""
        assert (
            self.is_processed
        ), "Dataset is not processed. Please run `process` first."
        if not self.is_loaded:
            run_async(self._load())

    def _get_item(self, index: int) -> typing.Optional[T]:
        if self.in_memory:
            assert self.is_loaded, "Dataset is not loaded yet, call .load() first"
            return self._get_from_cache(index)
        if self.is_processed:
            return self._get_processed(index)
        return None

    def _get_processed(self, idx: int) -> T:
        with open(Path(self.processed_dir) / str(idx), "rb") as f:
            return pickle.load(f)

    def _get_from_cache(self, idx: int) -> T:
        return self._cache[idx]

    async def _get_processed_async(self, idx: int) -> T:
        async with aiofiles.open(Path(self.processed_dir) / str(idx), "rb") as f:
            return pickle.loads(await f.read())

    async def _load(self):
        for i in trange(len(self), desc="Loading dataset", unit="files"):
            self._cache[i] = await self._get_processed_async(i)

    async def _process(self):
        try:
            _ = next(iter(Path(self.processed_dir).iterdir()))
            i = self._last_processed_idx + 1
        except:
            i = 0
        p_bar = tqdm(
            total=len(self), desc="Processing dataset", unit="files", initial=i
        )
        while i < len(self):
            try:
                item = self[i]
                async with aiofiles.open(
                    os.path.join(self.processed_dir, str(i)), "wb"
                ) as f:
                    await f.write(pickle.dumps(item))
            except Exception as e:
                await self._logger.error(f"Processing item {i}: {e}")
            finally:
                i += 1
                p_bar.update(1)
