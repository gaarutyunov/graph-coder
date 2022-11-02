import typing

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch_geometric.data import Dataset


def train_test_val_split(
    dataset: Dataset,
    num_data: int,
    test_size: float = 0.1,
    val_size: float = 0.05,
    seed: int = None,
) -> typing.Tuple[Dataset, typing.Optional[Dataset], typing.Optional[Dataset]]:
    train_dataset = None
    test_dataset = None
    val_dataset = None
    test_size_ = num_data // (test_size * 100)
    val_size_ = num_data // (val_size * 100)

    if test_size_ + val_size_ == 0:
        return dataset, None, None

    if test_size > 0:
        train_valid_idx, test_idx = train_test_split(
            np.arange(num_data),
            test_size=test_size,
            random_state=seed,
        )

    if test_size_ > 0 and val_size_ > 0:
        train_idx, valid_idx = train_test_split(
            train_valid_idx, test_size=val_size, random_state=seed
        )
        train_idx = torch.from_numpy(train_idx)
        valid_idx = torch.from_numpy(valid_idx)
        test_idx = torch.from_numpy(test_idx)
        train_dataset = dataset.index_select(train_idx)
        test_dataset = dataset.index_select(test_idx)
        val_dataset = dataset.index_select(valid_idx)
    elif test_size > 0:
        train_idx = torch.from_numpy(train_valid_idx)
        test_idx = torch.from_numpy(test_idx)
        train_dataset = dataset.index_select(train_idx)
        test_dataset = dataset.index_select(test_idx)

    return train_dataset, test_dataset, val_dataset
