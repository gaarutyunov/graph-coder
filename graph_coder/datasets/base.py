import abc

from torch.utils.data import DataLoader


class BaseDataset(abc.ABC):
    @property
    @abc.abstractmethod
    def loaders(self) -> dict[str, DataLoader]:
        return {}


