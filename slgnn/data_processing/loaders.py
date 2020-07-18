from random import shuffle
from abc import abstractmethod, ABC

import torch
from torch_geometric.data import DataLoader
from torch.utils.data import ConcatDataset
from rdkit.Chem.Scaffolds import MurckoScaffold


class BaseSplitter(ABC):
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

    @abstractmethod
    def _split_dataset(self):
        ...


class DataSplitter(BaseSplitter):
    """ Split dataset with given ratio.
    """

    def __init__(self, dataset, ratio=[0.8, 0.1, 0.1], shuffle=True, batch_size=32):
        super().__init__(dataset, ratio, shuffle, batch_size)

    def _split_dataset(self):
        assert sum(self.ratio) == 1
        if isinstance(self.dataset, list):
            self._split_from_list()
        else:
            if hasattr(self.dataset, "train_loader"):
                self.dataset.set_shuffle(self.shuffle)
                self.dataset.set_batch_size(self.batch_size)
                self._train_loader = self.dataset.train_loader()
                self._val_loader = self.dataset.val_loader()
                return
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
        train_l, val_l, test_l = list(), list(), list()
        for dataset in datasets:
            if hasattr(dataset, "train_loader"):
                train_l.append(dataset.train_data)
                val_l.append(dataset.val_data)
                continue
            if self.shuffle:
                indices = list(range(len(dataset)))
                shuffle(indices)
                dataset = dataset[indices]
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


class ScaffoldSplitter(BaseSplitter):
    """ Split dataset based on scaffold similarity.
    """

    def __init__(self, dataset, ratio=[0.8, 0.1, 0.1], shuffle=True, batch_size=32):
        super().__init__(dataset, ratio, shuffle, batch_size)

    def _generate_scaffold(self, smiles, include_chirality=False):
        """
        Obtain Bemis-Murcko scaffold from smiles
        :param smiles:
        :param include_chirality:
        :return: smiles of scaffold
        """
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(
            smiles=smiles, includeChirality=include_chirality
        )
        return scaffold

    def _split_dataset(self):
        # create dict of the form {scaffold_i: [idx1, idx....]}
        all_scaffolds = {}
        for i, smiles in enumerate(self.dataset._get_smiles()):
            scaffold = self._generate_scaffold(smiles, include_chirality=True)
            if scaffold not in all_scaffolds:
                all_scaffolds[scaffold] = [i]
            else:
                all_scaffolds[scaffold].append(i)

        # sort from largest to smallest sets
        all_scaffolds = {key: sorted(value) for key, value in all_scaffolds.items()}
        all_scaffold_sets = [
            scaffold_set
            for (scaffold, scaffold_set) in sorted(
                all_scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True
            )
        ]

        # get train, valid test indices
        train_cutoff = self.ratio[0] * len(self.dataset)
        valid_cutoff = (self.ratio[0] + self.ratio[1]) * len(self.dataset)
        train_idx, valid_idx, test_idx = [], [], []
        for scaffold_set in all_scaffold_sets:
            if len(train_idx) + len(scaffold_set) > train_cutoff:
                if len(train_idx) + len(valid_idx) + len(scaffold_set) > valid_cutoff:
                    test_idx.extend(scaffold_set)
                else:
                    valid_idx.extend(scaffold_set)
            else:
                train_idx.extend(scaffold_set)

        assert len(set(train_idx).intersection(set(valid_idx))) == 0
        assert len(set(test_idx).intersection(set(valid_idx))) == 0

        train_dataset = self.dataset[torch.tensor(train_idx)]
        valid_dataset = self.dataset[torch.tensor(valid_idx)]
        test_dataset = self.dataset[torch.tensor(test_idx)]

        self._train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=self.shuffle
        )
        self._val_loader = DataLoader(
            valid_dataset, batch_size=self.batch_size, shuffle=self.shuffle
        )
        if self.ratio[-1] != 0:
            self._test_loader = DataLoader(
                test_dataset, batch_size=self.batch_size, shuffle=self.shuffle
            )


class FixedSplitter(BaseSplitter):
    """ Fix the test set based the given ratio.
    """

