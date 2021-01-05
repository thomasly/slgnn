# Repurposing dataset from: https://clue.io/repurposing#download-data

import os

import torch
from torch_geometric.data import Data, InMemoryDataset
from chemreader.readers import Smiles
from tqdm import tqdm
import pandas as pd

from slgnn.models.gcn.utils import get_filtered_fingerprint


class Repurposing(InMemoryDataset):
    def __init__(self, verbose=False, root=None, **kwargs):
        if root is None:
            root = os.path.join("data", "Repurposing")
        self.verbose = verbose
        super().__init__(root, **kwargs)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["repurposing_samples_20200324.txt"]

    @property
    def processed_file_names(self):
        return ["data.pt"]

    @property
    def raw_dir(self):
        return self.root

    @property
    def processed_dir(self):
        return os.path.join(self.root, "graphs")

    def download(self):
        pass

    def process(self):
        """The method converting SMILES and labels to graphs.
        """

        data_list = list()
        pb = tqdm(self._get_smiles()) if self.verbose else self._get_smiles()
        for sm, y in pb:
            try:
                x, edge_idx = self._graph_helper(sm)
            except ValueError:
                continue
            y = torch.tensor(y, dtype=torch.float)
            data_list.append(Data(x=x, edge_index=edge_idx, y=y))

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def _graph_helper(self, smi):
        graph = Smiles(smi).to_graph(sparse=True)
        x = torch.tensor(graph["atom_features"], dtype=torch.float)
        edge_idx = graph["adjacency"].tocoo()
        edge_idx = torch.tensor([edge_idx.row, edge_idx.col], dtype=torch.long)
        return x, edge_idx

    def _get_smiles(self):
        df = pd.read_csv(self.raw_paths[0], sep="\t", header=9)
        for smi in df["smiles"]:
            fp = get_filtered_fingerprint(smi)
            yield smi, fp
