from random import shuffle, sample, choices, seed
from abc import abstractmethod, ABC

import torch
from torch_geometric.data import DataLoader
from torch.utils.data import ConcatDataset
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import MolFromSmiles


def _myShuffle(x, *s):
    x[slice(*s)] = sample(x[slice(*s)], len(x[slice(*s)]))


class BaseSplitter(ABC):
    """Base class for dataloaders.

    Args:
        dataset (torch_geometric.data.DataSet): the dataset to be used in training.
        ratio (list): list of floats indicating the splitting ratios. The numbers
            represent fractions of training, validating, and testing sets, respectively.
        shuffle (bool): Whether shuffle the dataset.
        batch_size (int): size of every mini batches.

    Attributes:
        dataset: the underhood dataset. Immutable.
        ratio: the fractions of training, validating, and testing sets. Immutable.
        shuffle: whether shuffle the dataset while loading. Mutable.
        batch_size: size of mini batches. Mutable.
        train_loader: the training set data loader.
        val_loader: the validation set data loader.
        test_loader: the testing set data loader. None if the testing ratio is 0.
    """

    def __init__(
        self,
        dataset,
        ratio=[0.8, 0.1, 0.1],
        shuffle=True,
        batch_size=32,
        dataloader=DataLoader,
    ):
        self._dataset = dataset
        self._ratio = ratio
        self._shuffle = shuffle
        self._batch_size = batch_size
        self._dataloader = dataloader
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
        """The method splitting the dataset. Need to be implemented by subclasses."""
        ...


class DataSplitter(BaseSplitter):
    """Split dataset with given ratio."""

    def __init__(
        self,
        dataset,
        ratio=[0.8, 0.1, 0.1],
        shuffle=True,
        batch_size=32,
        dataloader=DataLoader,
    ):
        super().__init__(dataset, ratio, shuffle, batch_size, dataloader)

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
            self._train_loader = self._dataloader(
                dataset[:sep1], batch_size=self.batch_size, shuffle=self.shuffle
            )
            if self.ratio[-1] == 0:
                self._val_loader = self._dataloader(
                    dataset[sep1:], batch_size=self.batch_size, shuffle=self.shuffle
                )
            else:
                self._val_loader = self._dataloader(
                    dataset[sep1:sep2], batch_size=self.batch_size, shuffle=self.shuffle
                )
                self._test_loader = self._dataloader(
                    dataset[sep2:], batch_size=self.batch_size, shuffle=self.shuffle
                )

    def _split_from_list(self):
        """Split and combine multiple datasets."""
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
        self._train_loader = self._dataloader(
            ConcatDataset(train_l), batch_size=self.batch_size, shuffle=self.shuffle
        )
        self._val_loader = self._dataloader(
            ConcatDataset(val_l), batch_size=self.batch_size, shuffle=self.shuffle
        )
        if self.ratio[-1] != 0:
            self._test_loader = self._dataloader(
                ConcatDataset(test_l), batch_size=self.batch_size, shuffle=self.shuffle
            )


class ScaffoldSplitter(BaseSplitter):
    """Split dataset based on scaffold similarity.

    Args:
        dataset (torch_geometric.data.DataSet): the dataset to be used in training.
        ratio (list): list of floats indicating the splitting ratios. The numbers
            represent fractions of training, validating, and testing sets, respectively.
        shuffle (bool): Whether shuffle the dataset.
        batch_size (int): size of every mini batches.
    """

    def __init__(
        self,
        dataset,
        ratio=[0.8, 0.1, 0.1],
        shuffle=True,
        batch_size=32,
        dataloader=DataLoader,
    ):
        super().__init__(dataset, ratio, shuffle, batch_size, dataloader=dataloader)

    def _generate_scaffold(self, smiles, include_chirality=False):
        """
        Obtain Bemis-Murcko scaffold from smiles

        Args:
            smiles (str): SMILES string
            include_chirality (bool): Whether taking chirality into consideration when
                generating MurckoScaffolds.
        Returns:
            SMILES of scaffold
        """
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(
            smiles=smiles, includeChirality=include_chirality
        )
        return scaffold

    def _split_dataset(self):
        # create dict of the form {scaffold_i: [idx1, idx....]}
        all_scaffolds = {}
        i = 0
        for smiles in self.dataset._get_smiles():
            if MolFromSmiles(smiles) is None:
                continue
            scaffold = self._generate_scaffold(smiles, include_chirality=True)
            if scaffold not in all_scaffolds:
                all_scaffolds[scaffold] = [i]
            else:
                all_scaffolds[scaffold].append(i)
            i += 1

        # sort from largest to smallest sets
        all_scaffolds = {key: sorted(value) for key, value in all_scaffolds.items()}
        all_scaffold_sets = [
            scaffold_set
            for (scaffold, scaffold_set) in sorted(
                all_scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True
            )
        ]

        # shuffle the order of the sets that have less than 5 members
        # make label distribution more even
        for i, scaffold_set in enumerate(all_scaffold_sets):
            if len(scaffold_set) <= 5:
                start_point = i
                break
        seed(0)
        _myShuffle(all_scaffold_sets, start_point, None)

        # get train, valid, and test indices
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

        self._train_loader = self._dataloader(
            train_dataset, batch_size=self.batch_size, shuffle=self.shuffle
        )
        self._val_loader = self._dataloader(
            valid_dataset, batch_size=self.batch_size, shuffle=self.shuffle
        )
        if self.ratio[-1] != 0:
            self._test_loader = self._dataloader(
                test_dataset, batch_size=self.batch_size, shuffle=self.shuffle
            )


class FixedSplitter(BaseSplitter):
    """Fix the test set based on the given ratio. The test set has the same
    data distribution with the training set, especially when the dataset labels are very
    imbalanced.

    Args:
        dataset (torch_geometric.data.DataSet): the dataset to be used in training.
        ratio (list): list of floats indicating the splitting ratios. The numbers
            represent fractions of training, validating, and testing sets, respectively.
        shuffle (bool): Whether shuffle the dataset.
        batch_size (int): size of every mini batches.
    """

    def __init__(
        self,
        dataset,
        ratio=[0.8, 0.1, 0.1],
        shuffle=True,
        batch_size=32,
        dataloader=DataLoader,
        random_seed=0,
    ):
        self.random_seed = random_seed
        super().__init__(dataset, ratio, shuffle, batch_size, dataloader=dataloader)

    def _split_dataset(self):
        seed(self.random_seed)

        mask = self.dataset.data.y == 1
        positive_dataset = self.dataset[mask]
        negative_dataset = self.dataset[~mask]
        positive_indices = list(range(len(positive_dataset)))
        negative_indices = list(range(len(negative_dataset)))
        if self.shuffle:
            shuffle(positive_indices)
            shuffle(negative_indices)

        pos_sep1 = int(len(positive_dataset) * self.ratio[0])
        pos_sep2 = int(len(positive_dataset) * (self.ratio[0] + self.ratio[1]))
        pos_sep3 = int(len(positive_dataset) * self.ratio[2])
        neg_sep1 = int(len(negative_dataset) * self.ratio[0])
        neg_sep2 = int(len(negative_dataset) * (self.ratio[0] + self.ratio[1]))
        neg_sep3 = int(len(negative_dataset) * self.ratio[2])

        positive_train_indices = positive_indices[:pos_sep1]
        positive_val_indices = positive_indices[pos_sep1:pos_sep2]
        positive_test_indices = positive_indices[-pos_sep3:]
        negative_train_indices = negative_indices[:neg_sep1]
        negative_val_indices = negative_indices[neg_sep1:neg_sep2]
        negative_test_indices = negative_indices[-neg_sep3:]

        self._train_loader = self._dataloader(
            ConcatDataset(
                [
                    positive_dataset[positive_train_indices],
                    negative_dataset[negative_train_indices],
                ]
            ),
            batch_size=self.batch_size,
            shuffle=self.shuffle,
        )
        self._val_loader = self._dataloader(
            ConcatDataset(
                [
                    positive_dataset[positive_val_indices],
                    negative_dataset[negative_val_indices],
                ]
            ),
            batch_size=self.batch_size,
            shuffle=self.shuffle,
        )
        self._test_loader = self._dataloader(
            ConcatDataset(
                [
                    positive_dataset[positive_test_indices],
                    negative_dataset[negative_test_indices],
                ]
            ),
            batch_size=self.batch_size,
            shuffle=self.shuffle,
        )


class OversamplingSplitter(BaseSplitter):
    """Oversampling in the training set. Make the number of negative and postive
    samples even.

    Args:
        dataset (torch_geometric.data.DataSet): the dataset to be used in training.
        ratio (list): list of floats indicating the splitting ratios. The numbers
            represent fractions of training, validating, and testing sets, respectively.
        shuffle (bool): Whether shuffle the dataset.
        batch_size (int): size of every mini batches.
        dataloader (torch_geometric.data.DataLoader): The DataLoader class used to load
            graph data. Default is DataLoader.
        random_seed (int): The random seed used in traning and testing set splitting.
            Default is 0.
    """

    def __init__(
        self,
        dataset,
        ratio=[0.8, 0.1, 0.1],
        shuffle=True,
        batch_size=32,
        dataloader=DataLoader,
        random_seed=0,
    ):
        self.random_seed = random_seed
        super().__init__(dataset, ratio, shuffle, batch_size, dataloader=dataloader)

    def _split_dataset(self):
        seed(self.random_seed)

        mask = self.dataset.data.y == 1
        positive_dataset = self.dataset[mask]
        negative_dataset = self.dataset[~mask]
        positive_indices = list(range(len(positive_dataset)))
        negative_indices = list(range(len(negative_dataset)))
        if self.shuffle:
            shuffle(positive_indices)
            shuffle(negative_indices)

        pos_sep1 = int(len(positive_dataset) * self.ratio[0])
        pos_sep2 = int(len(positive_dataset) * (self.ratio[0] + self.ratio[1]))
        pos_sep3 = int(len(positive_dataset) * self.ratio[2])
        neg_sep1 = int(len(negative_dataset) * self.ratio[0])
        neg_sep2 = int(len(negative_dataset) * (self.ratio[0] + self.ratio[1]))
        neg_sep3 = int(len(negative_dataset) * self.ratio[2])

        positive_train_indices = positive_indices[:pos_sep1]
        positive_val_indices = positive_indices[pos_sep1:pos_sep2]
        positive_test_indices = positive_indices[-pos_sep3:]
        negative_train_indices = negative_indices[:neg_sep1]
        negative_val_indices = negative_indices[neg_sep1:neg_sep2]
        negative_test_indices = negative_indices[-neg_sep3:]

        # Over sampling
        len_pos_train = len(positive_train_indices)
        len_neg_train = len(negative_train_indices)
        n_samples = abs(len_pos_train - len_neg_train)
        if len_pos_train > len_neg_train:
            negative_train_indices = negative_train_indices + choices(
                negative_train_indices, k=n_samples
            )
        else:
            positive_train_indices = positive_train_indices + choices(
                positive_train_indices, k=n_samples
            )

        self._train_loader = self._dataloader(
            ConcatDataset(
                [
                    positive_dataset[positive_train_indices],
                    negative_dataset[negative_train_indices],
                ]
            ),
            batch_size=self.batch_size,
            shuffle=self.shuffle,
        )
        self._val_loader = self._dataloader(
            ConcatDataset(
                [
                    positive_dataset[positive_val_indices],
                    negative_dataset[negative_val_indices],
                ]
            ),
            batch_size=self.batch_size,
            shuffle=self.shuffle,
        )
        self._test_loader = self._dataloader(
            ConcatDataset(
                [
                    positive_dataset[positive_test_indices],
                    negative_dataset[negative_test_indices],
                ]
            ),
            batch_size=self.batch_size,
            shuffle=self.shuffle,
        )
