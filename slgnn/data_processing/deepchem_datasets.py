import os.path as osp

import torch
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
import pandas as pd
from chemreader.readers import Smiles


class DeepchemDataset(InMemoryDataset):

    def __init__(self, root, name, transform=None, pre_transform=None,
                 pre_filter=None):
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

    def process(self):
        pass


class SiderDataset(DeepchemDataset):

    def __init__(self, root=None, name=None):
        if root is None:
            root = osp.join("data", "DeepChem", "Sider")
        if name is None:
            name = "sider"
        self.root = root
        self.name = name
        super().__init__(root=root, name=name)

    def process(self):
        sider_df = pd.read_csv(self.raw_paths[0])
        data_list = list()
        for row in sider_df.iterrows():
            smi = row[1]["smiles"]
            graph = Smiles(smi).to_graph(sparse=True)
            x = torch.tensor(graph["atom_features"],
                             dtype=torch.float)
            edge_idx = graph["adjacency"].tocoo()
            edge_idx = torch.tensor(
                [edge_idx.row, edge_idx.col], dtype=torch.long)
            y = torch.tensor(list(row[1][1:]), dtype=torch.long)[None, :]
            data_list.append(Data(x=x, edge_index=edge_idx, y=y))
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
