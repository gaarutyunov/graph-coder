from functools import partial
from typing import Optional

from torch.utils.data import DataLoader
from torch_geometric.data import Dataset, Data, LightningDataset

from graph_coder.data.collator import collator
from graph_coder.data.wrapper import preprocess_item
from graph_coder.datasets.base import GraphCoderDatasetBase
from graph_coder.utils.split import train_test_val_split


class GraphCoderDataset(GraphCoderDatasetBase):
    def __init__(self, root: Optional[str] = None, copy: bool = True):
        super().__init__(root)
        self.copy = copy

    def get(self, idx: int) -> Data:
        data = super().get_transformed(idx)
        data.idx = idx

        return preprocess_item(data)


class GraphCoderLightningDataset(LightningDataset):
    def __init__(
        self,
        root: str,
        test_size: float = 0.1,
        val_size: float = 0.05,
        batch_size: int = 16,
        num_workers: int = 0,
        num_nodes: int = 512,
        num_edges: int = 2048,
        seed: int = None,
        **kwargs,
    ):
        assert test_size + val_size < 1, "test_size + val_size must be less than 1"
        dataset = GraphCoderDataset(root)
        num_data = dataset.len()
        train, test, val = train_test_val_split(
            dataset, num_data, test_size, val_size, seed
        )
        self.num_nodes = num_nodes
        self.num_edges = num_edges
        super().__init__(
            train,
            test,
            val,
            batch_size,
            num_workers,
            **kwargs,
        )

    def dataloader(self, dataset: Dataset, shuffle: bool = False):
        return DataLoader(
            dataset,
            shuffle=shuffle,
            collate_fn=partial(
                collator, num_nodes=self.num_nodes, num_edges=self.num_edges
            ),
            **self.kwargs,
        )
