""" **TU formatted datasets and JAK datases.**
"""

import os
import os.path as osp
import glob
from abc import ABCMeta, abstractmethod

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_sparse import coalesce
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.datasets import TUDataset
from torch_geometric.io import read_txt_array
from torch_geometric.utils import remove_self_loops
from torch_geometric.data import DataLoader
from chemreader.readers import Smiles
from tqdm import tqdm

from slgnn.models.gcn.utils import get_filtered_fingerprint
from slgnn.data_processing.deepchem_datasets import DeepchemDataset


class ZINCDataset(TUDataset):
    """ The base class for TU formatted datasets.
    """

    @staticmethod
    def read_file(folder, prefix, name, dtype=None):
        path = osp.join(folder, "{}_{}.txt".format(prefix, name))
        return read_txt_array(path, sep=",", dtype=dtype)

    @staticmethod
    def cat(seq):
        seq = [item for item in seq if item is not None]
        seq = [item.unsqueeze(-1) if item.dim() == 1 else item for item in seq]
        return torch.cat(seq, dim=1) if len(seq) > 0 else None

    @staticmethod
    def split(data, batch):
        node_slice = torch.cumsum(torch.from_numpy(np.bincount(batch)), 0)
        node_slice = torch.cat([torch.tensor([0]), node_slice])

        row, _ = data.edge_index
        edge_slice = torch.cumsum(torch.from_numpy(np.bincount(batch[row])), 0)
        edge_slice = torch.cat([torch.tensor([0]), edge_slice])

        # Edge indices should start at zero for every graph.
        data.edge_index -= node_slice[batch[row]].unsqueeze(0)
        data.__num_nodes__ = torch.bincount(batch).tolist()

        slices = {"edge_index": edge_slice}
        if data.x is not None:
            slices["x"] = node_slice
        if data.edge_attr is not None:
            slices["edge_attr"] = edge_slice
        if data.y is not None:
            if data.y.size(0) == batch.size(0):
                slices["y"] = node_slice
            else:
                slices["y"] = torch.arange(0, batch[-1] + 2, dtype=torch.long)

        return data, slices

    def read_tu_data(self, folder, prefix):
        files = glob.glob(osp.join(folder, "{}_*.txt".format(prefix)))
        names = [f.split(os.sep)[-1][len(prefix) + 1 : -4] for f in files]

        edge_index = self.read_file(folder, prefix, "A", torch.long).t() - 1
        batch = self.read_file(folder, prefix, "graph_indicator", torch.long) - 1

        node_attributes, node_labels = None, None
        if "node_attributes" in names:
            node_attributes = self.read_file(folder, prefix, "node_attributes")
        if "node_labels" in names:
            node_labels = self.read_file(folder, prefix, "node_labels", torch.long)
            if node_labels.dim() == 1:
                node_labels = node_labels.unsqueeze(-1)
            node_labels = node_labels - node_labels.min(dim=0)[0]
            node_labels = node_labels.unbind(dim=-1)
            node_labels = [F.one_hot(x, num_classes=-1) for x in node_labels]
            node_labels = torch.cat(node_labels, dim=-1).to(torch.float)
        x = self.cat([node_attributes, node_labels])

        edge_attributes, edge_labels = None, None
        if "edge_attributes" in names:
            edge_attributes = self.read_file(folder, prefix, "edge_attributes")
        if "edge_labels" in names:
            edge_labels = self.read_file(folder, prefix, "edge_labels", torch.long)
            if edge_labels.dim() == 1:
                edge_labels = edge_labels.unsqueeze(-1)
            edge_labels = edge_labels - edge_labels.min(dim=0)[0]
            edge_labels = edge_labels.unbind(dim=-1)
            edge_labels = [F.one_hot(e, num_classes=-1) for e in edge_labels]
            edge_labels = torch.cat(edge_labels, dim=-1).to(torch.float)
        edge_attr = self.cat([edge_attributes, edge_labels])

        y = None
        if "graph_attributes" in names:  # Regression problem.
            y = self.read_file(folder, prefix, "graph_attributes")
        elif "graph_labels" in names:  # Classification problem.
            y = self.read_file(folder, prefix, "graph_labels", torch.long)
            _, y = y.unique(sorted=True, return_inverse=True)

        num_nodes = edge_index.max().item() + 1 if x is None else x.size(0)
        edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
        edge_index, edge_attr = coalesce(edge_index, edge_attr, num_nodes, num_nodes)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        data, slices = self.split(data, batch)

        return data, slices

    @property
    def raw_dir(self):
        name = "raw{}".format("_cleaned" if self.cleaned else "")
        return os.path.join(self.root, self.name, name)

    @property
    def raw_file_names(self):
        names = [
            "_A.txt",
            "_graph_indicator.txt",
            "_node_attributes.txt",
            "_graph_labels.txt",
        ]
        return [self.name + name for name in names]

    @property
    def processed_dir(self):
        name = "processed{}".format("_cleaned" if self.cleaned else "")
        return os.path.join(self.root, self.name, name)

    @property
    def processed_file_names(self):
        return "data.pt"

    def download(self):
        pass

    def process(self):
        self.data, self.slices = self.read_tu_data(self.raw_dir, self.name)

        if self.pre_filter is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [data for data in data_list if self.pre_filter(data)]
            self.data, self.slices = self.collate(data_list)

        if self.pre_transform is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [self.pre_transform(data) for data in data_list]
            self.data, self.slices = self.collate(data_list)

        torch.save((self.data, self.slices), self.processed_paths[0])


class JAKDataset(TUDataset):
    """ The base class of JAK datasets.
    """

    @property
    def raw_file_names(self):
        names = [
            "_A.txt",
            "_graph_indicator.txt",
            "_node_attributes.txt",
            "_graph_labels.txt",
        ]
        return [self.name + name for name in names]

    @property
    def processed_file_names(self):
        return "data.pt"

    def download(self):
        pass

    def process(self):
        super().process()


class ZINC1k(ZINCDataset):
    """ ZINC1k dataset. Contains 1k chemicals sampled from ZINC dataset. The chemicals
    are checked with Tanimoto similarity in order to make them dissimilar with each
    other.

    Args:
        root (str): path to the root directory.
    """

    def __init__(self, root=None):
        if root is None:
            root = osp.join("data", "ZINC", "graphs")
        name = "ZINC1k"
        super().__init__(root=root, name=name, use_node_attr=True)

    def process(self):
        super().process()

    def __str__(self):
        return "ZINC1k"


class ZINC10k(ZINCDataset):
    """ ZINC10k dataset. Contains 10k chemicals sampled from ZINC dataset. The chemicals
    are checked with Tanimoto similarity in order to make them dissimilar with each
    other.

    Args:
        root (str): path to the root directory.
    """

    def __init__(self):
        root = osp.join("data", "ZINC", "graphs")
        name = "ZINC10k"
        super().__init__(root=root, name=name, use_node_attr=True)

    def process(self):
        super().process()

    def __str__(self):
        return "ZINC10k"


class ZINC100k(ZINCDataset):
    """ ZINC100k dataset. Contains 100k chemicals sampled from ZINC dataset. The
    chemicals are checked with Tanimoto similarity in order to make them dissimilar with
    each other.

    Args:
        root (str): path to the root directory.
    """

    def __init__(self):
        root = osp.join("data", "ZINC", "graphs")
        name = "ZINC100k"
        super().__init__(root=root, name=name, use_node_attr=True)

    def process(self):
        super().process()

    def __str__(self):
        return "ZINC100k"


class JAK1(DeepchemDataset):
    """ Chemicals that bind to JAK1 kinase.

    Args:
        root (str): path to the root directory.
        name (str): name of the file prefix. Default is "filtered_JAK1".
    """

    def __init__(self, root=None, name="filtered_JAK1"):
        if root is None:
            root = osp.join("data", "JAK", "graphs", "JAK1")
        super().__init__(root=root, name=name)

    def _get_smiles(self):
        df = pd.read_csv(self.raw_paths[0])
        smiles = iter(df["Smiles"])
        return smiles

    def _get_labels(self):
        df = pd.read_csv(self.raw_paths[0])
        for lb in df["Activity"]:
            if lb == -1:
                lb = 0
            yield torch.tensor([lb], dtype=torch.long)

    def process(self, verbose=0):
        super().process(verbose)

    def __str__(self):
        return "JAK1"


class JAK2(DeepchemDataset):
    """ Chemicals that bind to JAK2 kinase.

    Args:
        root (str): path to the root directory.
        name (str): name of the file prefix. Default is "filtered_JAK2".
    """

    def __init__(self, root=None, name="filtered_JAK2"):
        if root is None:
            root = osp.join("data", "JAK", "graphs", "JAK2")
        super().__init__(root=root, name=name)

    def _get_smiles(self):
        df = pd.read_csv(self.raw_paths[0])
        smiles = iter(df["Smiles"])
        return smiles

    def _get_labels(self):
        df = pd.read_csv(self.raw_paths[0])
        for lb in df["Activity"]:
            if lb == -1:
                lb = 0
            yield torch.tensor([lb], dtype=torch.long)

    def process(self, verbose=0):
        super().process(verbose)

    def __str__(self):
        return "JAK2"


class JAK3(DeepchemDataset):
    """ Chemicals that bind to JAK3 kinase.

    Args:
        root (str): path to the root directory.
        name (str): name of the file prefix. Default is "filtered_JAK3".
    """

    def __init__(self, root=None, name="filtered_JAK3"):
        if root is None:
            root = osp.join("data", "JAK", "graphs", "JAK3")
        super().__init__(root=root, name=name)

    def _get_smiles(self):
        df = pd.read_csv(self.raw_paths[0])
        smiles = iter(df["Smiles"])
        return smiles

    def _get_labels(self):
        df = pd.read_csv(self.raw_paths[0])
        for lb in df["Activity"]:
            if lb == -1:
                lb = 0
            yield torch.tensor([lb], dtype=torch.long)

    def process(self, verbose=0):
        super().process(verbose)

    def __str__(self):
        return "JAK3"


class JAK1FP(JAKDataset):
    """ Chemicals that bind to JAK1 kinase with fingerprints as labels.
    """

    def __init__(self):
        root = osp.join("data", "JAK", "graphs")
        name = "JAK1FP"
        super().__init__(root=root, name=name, use_node_attr=True)

    def process(self):
        super().process()

    def __str__(self):
        return "JAK1FP"


class JAK2FP(JAKDataset):
    """ Chemicals that bind to JAK2 kinase with fingerprints as labels.
    """

    def __init__(self):
        root = osp.join("data", "JAK", "graphs")
        name = "JAK2FP"
        super().__init__(root=root, name=name, use_node_attr=True)

    def process(self):
        super().process()

    def __str__(self):
        return "JAK2FP"


class JAK3FP(JAKDataset):
    """ Chemicals that bind to JAK3 kinase with fingerprints as labels.
    """

    def __init__(self):
        root = osp.join("data", "JAK", "graphs")
        name = "JAK3FP"
        super().__init__(root=root, name=name, use_node_attr=True)

    def process(self):
        super().process()

    def __str__(self):
        return "JAK3FP"


class JAK1Train(JAKDataset):
    """ The presplitted JAK1 training dataset. The presplitting process was done by
    clustering chemicals based on Tanimoto similarity and then splitting the clusters
    to make sure training set and testing set are dissimilar. While splitting, the
    ratio of positive and negative samples was retained.
    """

    def __init__(self):
        root = osp.join("data", "JAK", "graphs")
        name = "JAK1_train"
        super().__init__(root=root, name=name, use_node_attr=True)

    def process(self):
        super().process()

    def __str__(self):
        return "JAK1Train"


class JAK1Test(JAKDataset):
    """ The presplitted JAK1 testing dataset corresponding to the training set above.
    """

    def __init__(self):
        root = osp.join("data", "JAK", "graphs")
        name = "JAK1_test"
        super().__init__(root=root, name=name, use_node_attr=True)

    def process(self):
        super().process()

    def __str__(self):
        return "JAK1Test"


class JAK2Train(JAKDataset):
    """ The presplitted JAK2 training dataset. The presplitting method is the same as
    JAK1Train and JAK1Test datasets.
    """

    def __init__(self):
        root = osp.join("data", "JAK", "graphs")
        name = "JAK2_train"
        super().__init__(root=root, name=name, use_node_attr=True)

    def process(self):
        super().process()

    def __str__(self):
        return "JAK2Train"


class JAK2Test(JAKDataset):
    """ The presplitted JAK2 testing dataset corresponding to the training set above.
    """

    def __init__(self):
        root = osp.join("data", "JAK", "graphs")
        name = "JAK2_test"
        super().__init__(root=root, name=name, use_node_attr=True)

    def process(self):
        super().process()

    def __str__(self):
        return "JAK2Test"


class JAK3Train(JAKDataset):
    """ The presplitted JAK3 training dataset. The presplitting method is the same as
    JAK1Train and JAK1Test datasets.
    """

    def __init__(self):
        root = osp.join("data", "JAK", "graphs")
        name = "JAK3_train"
        super().__init__(root=root, name=name, use_node_attr=True)

    def process(self):
        super().process()

    def __str__(self):
        return "JAK3Train"


class JAK3Test(JAKDataset):
    """ The presplitted JAK3 testing dataset corresponding to the training set above.
    """

    def __init__(self):
        root = osp.join("data", "JAK", "graphs")
        name = "JAK3_test"
        super().__init__(root=root, name=name, use_node_attr=True)

    def process(self):
        super().process()

    def __str__(self):
        return "JAK3Test"


class PresplittedBase:
    """ The base class can be used by data loader to load presplitted datasets.

    Args:
        shuffle_ (bool): Whether shuffling the dataset.
        batch_size (int): The size of mini batches.

    Attributes:
        train_data: the training dataset.
        val_data: the validation dataset.
        num_classes: number of labeled classes.
    """

    def __init__(self, shuffle_=None, batch_size=None):
        self._shuffle = shuffle_
        self._batch_size = batch_size

    def set_shuffle(self, value):
        self._shuffle = value

    def set_batch_size(self, value):
        self._batch_size = value

    @property
    def train_data(self):
        return self._train_data

    @property
    def val_data(self):
        return self._val_data

    @property
    def num_classes(self):
        return self.train_data.num_classes

    def train_loader(self):
        """ Must be implemented by subclasses.

        Returns:
            DataLoader: a instance of pytorch-geometric DataLoader class with training
                dataset underhood.
        """
        raise NotImplementedError

    def val_loader(self):
        """ Must be implemented by subclasses.

        Returns:
            DataLoader: a instance of pytorch-geometric DataLoader class with validation
                dataset underhood.
        """
        raise NotImplementedError

    def __str__(self):
        return self._str()

    def _str(self):
        """ Must be implemented by subclasses.

        Returns:
            str: Name or description of the class.
        """
        raise NotImplementedError


class JAK1Presplitted(PresplittedBase):
    """ The class can be used by dataloaders for JAK1 presplitted dataset.

    Args:
        shuffle_ (bool): Whether shuffling the dataset.
        batch_size (int): The size of mini batches.
    """

    def __init__(self, shuffle_=None, batch_size=None):
        super().__init__(shuffle_, batch_size)
        self._train_data = JAK1Train()
        self._val_data = JAK1Test()

    def train_loader(self):
        return DataLoader(
            self._train_data, shuffle=self._shuffle, batch_size=self._batch_size
        )

    def val_loader(self):
        return DataLoader(
            self._val_data, shuffle=self._shuffle, batch_size=self._batch_size
        )

    def _str(self):
        return "JAK1Presplitted"


class JAK2Presplitted(PresplittedBase):
    """ The class can be used by dataloaders for JAK2 presplitted dataset.

    Args:
        shuffle_ (bool): Whether shuffling the dataset.
        batch_size (int): The size of mini batches.
    """

    def __init__(self, shuffle_=None, batch_size=None):
        super().__init__(shuffle_, batch_size)
        self._train_data = JAK2Train()
        self._val_data = JAK2Test()

    def train_loader(self):
        return DataLoader(
            self._train_data, shuffle=self._shuffle, batch_size=self._batch_size
        )

    def val_loader(self):
        return DataLoader(
            self._val_data, shuffle=self._shuffle, batch_size=self._batch_size
        )

    def _str(self):
        return "JAK2Presplitted"


class JAK3Presplitted(PresplittedBase):
    """ The class can be used by dataloaders for JAK3 presplitted dataset.

    Args:
        shuffle_ (bool): Whether shuffling the dataset.
        batch_size (int): The size of mini batches.
    """

    def __init__(self, shuffle_=None, batch_size=None):
        super().__init__(shuffle_, batch_size)
        self._train_data = JAK3Train()
        self._val_data = JAK3Test()

    def train_loader(self):
        return DataLoader(
            self._train_data, shuffle=self._shuffle, batch_size=self._batch_size
        )

    def val_loader(self):
        return DataLoader(
            self._val_data, shuffle=self._shuffle, batch_size=self._batch_size
        )

    def _str(self):
        return "JAK3Presplitted"


class DudeDecoys(InMemoryDataset, metaclass=ABCMeta):
    """ The base class handling decoys from DUD*E server.

    Args:
        root (str): root path.
        transform: optional. The transformer transforms data on the fly.
        pre_transform: optional. The transformer transforms data while converting raw
            data to graphs.
        pre_filter: optional. The filter function filters data while converting raw
            data to graphs.
    """

    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @abstractmethod
    def raw_file_names(self):
        """ Must be implemented by subclasses.

        Returns:
            list: decoys filenames. Does not need the full list of all files. One
                filename to make the class constuctor skip download is enough.
        """
        raw_file_name = "decoys.P10000001.picked"
        return [raw_file_name]

    @property
    def processed_file_names(self):
        return ["data.pt"]

    def download(self):
        pass

    def process(self, verbose=0):
        """ Mothed processes decoys from DUD*E.

        Args:
            verbose (bool): Whether showing the process bar.
        """
        smiles = self._get_smiles(verbose)
        fingerprints = self._get_fingerprints(smiles, verbose)
        data_list = list()
        pb = (
            tqdm(zip(smiles, fingerprints), desc="Generating graphs", total=len(smiles))
            if verbose
            else zip(smiles, fingerprints)
        )
        for smi, y in pb:
            try:
                x, edge_idx = self._graph_helper(smi)
                y = torch.LongTensor([y])
                data_list.append(Data(x=x, edge_index=edge_idx, y=y))
            except AttributeError:  # SMILES invalid
                continue
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def _graph_helper(self, smi):
        graph = Smiles(smi).to_graph(sparse=True)
        x = torch.tensor(graph["atom_features"], dtype=torch.float)
        edge_idx = graph["adjacency"].tocoo()
        edge_idx = torch.tensor([edge_idx.row, edge_idx.col], dtype=torch.long)
        return x, edge_idx

    def _get_smiles(self, verbose):
        smiles = set()
        picked_files = os.scandir(self.raw_dir)
        it = (
            tqdm(list(picked_files), desc="Loading SMILES") if verbose else picked_files
        )
        for picked in it:
            if not picked.name.endswith(".picked"):
                continue
            with open(picked.path, "r") as f:
                contents = f.readlines()[1:]
            for line in contents:
                smiles.add(line.split()[0])
        return list(smiles)

    def _get_fingerprints(self, smiles, verbose):
        it = tqdm(smiles, desc="Generating fingerprints") if verbose else smiles
        return [get_filtered_fingerprint(sm) for sm in it]


class JAK1Dude(DudeDecoys):
    """ DUD*E decoys generated with JAK1 dataset.

    Args:
        root (str): root path to the dataset directory.
    """

    def __init__(self, root="data/JAK/DUDE_decoys/JAK1"):
        super().__init__(root=root)

    def raw_file_names(self):
        return ["decoys.P10000001.picked"]

    def process(self):
        super().process(verbose=1)

    def _str(self):
        return "JAK1Dude"


class JAK2Dude(DudeDecoys):
    """ DUD*E decoys generated with JAK2 dataset.

    Args:
        root (str): root path to the dataset directory.
    """

    def __init__(self, root="data/JAK/DUDE_decoys/JAK2"):
        super().__init__(root=root)

    def raw_file_names(self):
        return ["decoys.P10000001.picked"]

    def process(self):
        super().process(verbose=1)

    def _str(self):
        return "JAK2Dude"


class JAK3Dude(DudeDecoys):
    """ DUD*E decoys generated with JAK3 dataset.

    Args:
        root (str): root path to the dataset directory.
    """

    def __init__(self, root="data/JAK/DUDE_decoys/JAK3"):
        super().__init__(root=root)

    def raw_file_names(self):
        return ["decoys.P10000002.picked"]

    def process(self):
        super().process(verbose=1)

    def _str(self):
        return "JAK3Dude"
