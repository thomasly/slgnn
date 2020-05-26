import os
import os.path as osp
import glob

import numpy as np
import torch
import torch.nn.functional as F
from torch_sparse import coalesce
from torch_geometric.data import Data
from torch_geometric.datasets import TUDataset
from torch_geometric.io import read_txt_array
from torch_geometric.utils import remove_self_loops
from torch_geometric.data import DataLoader


class ZINCDataset(TUDataset):
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
    def __init__(self):
        root = osp.join("data", "ZINC", "graphs")
        name = "ZINC10k"
        super().__init__(root=root, name=name, use_node_attr=True)

    def process(self):
        super().process()

    def __str__(self):
        return "ZINC10k"


class ZINC100k(ZINCDataset):
    def __init__(self):
        root = osp.join("data", "ZINC", "graphs")
        name = "ZINC100k"
        super().__init__(root=root, name=name, use_node_attr=True)

    def process(self):
        super().process()

    def __str__(self):
        return "ZINC100k"


class JAK1(JAKDataset):
    def __init__(self):
        root = osp.join("data", "JAK", "graphs")
        name = "JAK1"
        super().__init__(root=root, name=name, use_node_attr=True)

    def process(self):
        super().process()

    def __str__(self):
        return "JAK1"


class JAK2(JAKDataset):
    def __init__(self):
        root = osp.join("data", "JAK", "graphs")
        name = "JAK2"
        super().__init__(root=root, name=name, use_node_attr=True)

    def process(self):
        super().process()

    def __str__(self):
        return "JAK2"


class JAK3(JAKDataset):
    def __init__(self):
        root = osp.join("data", "JAK", "graphs")
        name = "JAK3"
        super().__init__(root=root, name=name, use_node_attr=True)

    def process(self):
        super().process()

    def __str__(self):
        return "JAK3"


class JAK1FP(JAKDataset):
    def __init__(self):
        root = osp.join("data", "JAK", "graphs")
        name = "JAK1FP"
        super().__init__(root=root, name=name, use_node_attr=True)

    def process(self):
        super().process()

    def __str__(self):
        return "JAK1FP"


class JAK2FP(JAKDataset):
    def __init__(self):
        root = osp.join("data", "JAK", "graphs")
        name = "JAK2FP"
        super().__init__(root=root, name=name, use_node_attr=True)

    def process(self):
        super().process()

    def __str__(self):
        return "JAK2FP"


class JAK3FP(JAKDataset):
    def __init__(self):
        root = osp.join("data", "JAK", "graphs")
        name = "JAK3FP"
        super().__init__(root=root, name=name, use_node_attr=True)

    def process(self):
        super().process()

    def __str__(self):
        return "JAK3FP"


class JAK1Train(JAKDataset):
    def __init__(self):
        root = osp.join("data", "JAK", "graphs")
        name = "JAK1_train"
        super().__init__(root=root, name=name, use_node_attr=True)

    def process(self):
        super().process()

    def __str__(self):
        return "JAK1Train"


class JAK1Test(JAKDataset):
    def __init__(self):
        root = osp.join("data", "JAK", "graphs")
        name = "JAK1_test"
        super().__init__(root=root, name=name, use_node_attr=True)

    def process(self):
        super().process()

    def __str__(self):
        return "JAK1Test"


class JAK2Train(JAKDataset):
    def __init__(self):
        root = osp.join("data", "JAK", "graphs")
        name = "JAK2_train"
        super().__init__(root=root, name=name, use_node_attr=True)

    def process(self):
        super().process()

    def __str__(self):
        return "JAK2Train"


class JAK2Test(JAKDataset):
    def __init__(self):
        root = osp.join("data", "JAK", "graphs")
        name = "JAK2_test"
        super().__init__(root=root, name=name, use_node_attr=True)

    def process(self):
        super().process()

    def __str__(self):
        return "JAK2Test"


class JAK3Train(JAKDataset):
    def __init__(self):
        root = osp.join("data", "JAK", "graphs")
        name = "JAK3_train"
        super().__init__(root=root, name=name, use_node_attr=True)

    def process(self):
        super().process()

    def __str__(self):
        return "JAK3Train"


class JAK3Test(JAKDataset):
    def __init__(self):
        root = osp.join("data", "JAK", "graphs")
        name = "JAK3_test"
        super().__init__(root=root, name=name, use_node_attr=True)

    def process(self):
        super().process()

    def __str__(self):
        return "JAK3Test"


class PresplittedBase:
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
        raise NotImplementedError

    def val_loader(self):
        raise NotImplementedError

    def __str__(self):
        return self._str()

    def _str(self):
        raise NotImplementedError


class JAK1Presplitted(PresplittedBase):
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
            self._val_data, shuffle=self._shuffle, batch_size=self.batch_size
        )

    def _str(self):
        return "JAK1Presplitted"


class JAK2Presplitted(PresplittedBase):
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
            self._val_data, shuffle=self._shuffle, batch_size=self.batch_size
        )

    def _str(self):
        return "JAK2Presplitted"


class JAK3Presplitted(PresplittedBase):
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
