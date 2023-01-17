import abc
import typing

import torch
from torch.utils.data import DataLoader, random_split, Dataset


class BaseDataset(Dataset, abc.ABC):
    def __init__(
        self,
        collate_fn: typing.Callable,
        random_seed: typing.Optional[int] = None,
        test_size: float = 0.2,
        val_size: float = 0.2,
        batch_size: int = 1,
    ):
        self._loaders = {}
        self.batch_size = batch_size
        self.collate_fn = collate_fn
        self.random_seed = random_seed
        self.val_size = val_size
        self.test_size = test_size

    @property
    def loaders(self) -> dict[str, DataLoader]:
        return self._loaders

    def split(self):
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
