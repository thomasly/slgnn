from random import shuffle

from torch_geometric.data import DataLoader
from torch.utils.data import ConcatDataset


class DataSplitter:
    def __init__(self, dataset, ratio=[0.8, 0.1, 0.1], shuffle=True, batch_size=32):
        self._dataset = dataset
        self._ratio = ratio
        self._shuffle = shuffle
        self._batch_size = batch_size
        self._train_loader = None
        self._val_loader = None
        self._test_loader = None

    @property
    def dataset(self):
        return self._dataset

    @property
    def ratio(self):
        return self._ratio

    @property
    def shuffle(self):
        return self._shuffle

    @shuffle.setter
    def shuffle(self, value):
        assert value in (True, False)
        if value != self._shuffle:
            self._shuffle = value
            self._split_dataset()

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value):
        assert value > 0
        if value != self._batch_size:
            self._batch_size = value
            self._split_dataset()

    @property
    def train_loader(self):
        if self._train_loader is None:
            self._split_dataset()
        return self._train_loader

    @property
    def val_loader(self):
        if self._val_loader is None:
            self._split_dataset()
        return self._val_loader

    @property
    def test_loader(self):
        if self._test_loader is None:
            if self.ratio[-1] == 0:
                print("Testing ratio is set to 0.")
                return
            else:
                self._split_dataset()
        return self._test_loader

    def _split_dataset(self):
        assert sum(self.ratio) == 1
        if isinstance(self.dataset, list):
            self._split_from_list()
        else:
            if self.shuffle:
                indices = list(range(len(self.dataset)))
                shuffle(indices)
                dataset = self.dataset[indices]
            sep1 = int(len(dataset) * self.ratio[0])
            sep2 = min(int(len(dataset) * self.ratio[1]) + sep1, len(dataset))
            self._train_loader = DataLoader(
                dataset[:sep1], batch_size=self.batch_size, shuffle=self.shuffle
            )
            if self.ratio[-1] == 0:
                self._val_loader = DataLoader(
                    dataset[sep1:], batch_size=self.batch_size, shuffle=self.shuffle
                )
            else:
                self._val_loader = DataLoader(
                    dataset[sep1:sep2], batch_size=self.batch_size, shuffle=self.shuffle
                )
                self._test_loader = DataLoader(
                    dataset[sep2:], batch_size=self.batch_size, shuffle=self.shuffle
                )

    def _split_from_list(self):
        datasets = self.dataset.copy()
        if self.shuffle:
            for i, dataset in enumerate(datasets):
                indices = list(range(len(dataset)))
                shuffle(indices)
                datasets[i] = dataset[indices]
        train_l, val_l, test_l = list(), list(), list()
        for dataset in datasets:
            sep1 = int(len(dataset) * self.ratio[0])
            sep2 = min(int(len(dataset) * self.ratio[1]) + sep1, len(dataset))
            train_l.append(dataset[:sep1])
            if self.ratio[-1] == 0:
                val_l.append(dataset[sep1:])
            else:
                val_l.append(dataset[sep1:sep2])
                test_l.append(dataset[sep2:])
        self._train_loader = DataLoader(
            ConcatDataset(train_l), batch_size=self.batch_size, shuffle=self.shuffle
        )
        self._val_loader = DataLoader(
            ConcatDataset(val_l), batch_size=self.batch_size, shuffle=self.shuffle
        )
        if self.ratio[-1] != 0:
            self._test_loader = DataLoader(
                ConcatDataset(test_l), batch_size=self.batch_size, shuffle=self.shuffle
            )
