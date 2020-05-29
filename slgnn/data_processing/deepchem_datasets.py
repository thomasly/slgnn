import os.path as osp
from abc import ABCMeta, abstractmethod

import torch
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
import pandas as pd
from chemreader.readers import Smiles
from tqdm import tqdm

from slgnn.models.gcn.utils import get_filtered_fingerprint


class DeepchemDataset(InMemoryDataset, metaclass=ABCMeta):
    def __init__(self, root, name, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.name = name
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        raw_file_name = self.name + ".csv"
        return [raw_file_name]

    @property
    def processed_file_names(self):
        return ["data.pt"]

    def download(self):
        pass

    def process(self, verbose=0):
        smiles = self._get_smiles()
        labels = self._get_labels()
        data_list = list()
        counter = 0
        pb = tqdm(zip(smiles, labels)) if verbose else zip(smiles, labels)
        for smi, y in pb:
            x, edge_idx = self._graph_helper(smi)
            data_list.append(Data(x=x, edge_index=edge_idx, y=y))
            counter += 1
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def _graph_helper(self, smi):
        graph = Smiles(smi).to_graph(sparse=True)
        x = torch.tensor(graph["atom_features"], dtype=torch.float)
        edge_idx = graph["adjacency"].tocoo()
        edge_idx = torch.tensor([edge_idx.row, edge_idx.col], dtype=torch.long)
        return x, edge_idx

    @abstractmethod
    def _get_smiles(self):
        ...

    @abstractmethod
    def _get_labels(self):
        ...


class Sider(DeepchemDataset):
    def __init__(self, root=None, name=None):
        if root is None:
            root = osp.join("data", "DeepChem", "Sider")
        if name is None:
            name = "sider"
        self.root = root
        self.name = name
        super().__init__(root=root, name=name)

    def _get_smiles(self):
        sider_df = pd.read_csv(self.raw_paths[0])
        smiles = iter(sider_df["smiles"])
        return smiles

    def _get_labels(self):
        sider_df = pd.read_csv(self.raw_paths[0])
        for row in sider_df.iterrows():
            yield torch.tensor(list(row[1][1:]), dtype=torch.long)[None, :]

    def process(self, verbose=0):
        super().process(verbose)


class SiderFP(Sider):
    def __init__(self, root=None, name=None):
        if root is None:
            root = osp.join("data", "DeepChem", "SiderFP")
        if name is None:
            name = "sider"
        self.root = root
        self.name = name
        super().__init__(root=root, name=name)

    def _get_labels(self):
        sider_df = pd.read_csv(self.raw_paths[0])
        for smi in sider_df["smiles"]:
            fp = get_filtered_fingerprint(smi)
            yield torch.tensor(list(fp), dtype=torch.long)[None, :]

    def process(self, verbose=1):
        super().process(verbose)

    # def process(self):
    #     sider_df = pd.read_csv(self.raw_paths[0])
    #     data_list = list()
    #     for row in sider_df.iterrows():
    #         smi = row[1]["smiles"]
    #         graph = Smiles(smi).to_graph(sparse=True)
    #         x = torch.tensor(graph["atom_features"], dtype=torch.float)
    #         edge_idx = graph["adjacency"].tocoo()
    #         edge_idx = torch.tensor([edge_idx.row, edge_idx.col], dtype=torch.long)
    #         fp = get_pubchem_fingerprint(smi)
    #         y = torch.tensor(fp, dtype=torch.long)[None, :]
    #         data_list.append(Data(x=x, edge_index=edge_idx, y=y))
    #     data, slices = self.collate(data_list)
    #     torch.save((data, slices), self.processed_paths[0])
