import numpy as np
from torch.utils.data import Dataset


class SimpleIterator:
    def __init__(self, data, batch_size, shuffle=False):
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.reset()

    def reset(self):
        self.data_size = len(self.data)
        self.index = 0
        if self.shuffle:
            self.data = np.random.permutation(self.data)

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= self.data_size:
            raise StopIteration
        batch = self.data[self.index : self.index + self.batch_size]
        self.index += self.batch_size
        return batch


class SimpleDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.data)
