from chemreader.readers import Smiles
import torch
from torch_geometric.data import Data, InMemoryDataset
import pandas as pd


class _Base(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return self.root

    def download(self):
        pass

    def process(self, smiles_col, ecfp_col, label_col=None):
        data_list = list()
        df = pd.read_csv(self.raw_paths[0])
        if label_col is None:
            it = zip(df[smiles_col], df[ecfp_col])
        else:
            it = zip(df[smiles_col], df[ecfp_col], df[label_col])
        for item in it:
            smiles = item[0]
            fp = item[1]
            if label_col is not None:
                label = item[2]
            smi = Smiles(smiles)
            try:
                graph = smi.to_graph(sparse=True)
            except AttributeError:
                continue
            x = torch.tensor(graph["atom_features"], dtype=torch.float)
            edge_idx = graph["adjacency"].tocoo()
            edge_idx = torch.tensor([edge_idx.row, edge_idx.col], dtype=torch.long)
            y = torch.tensor(list(map(int, list(fp.strip()))), dtype=torch.long)[
                None, :
            ]
            if label_col is None:
                data_list.append(Data(x=x, edge_index=edge_idx, y=y))
            else:
                data_list.append(Data(x=x, edge_index=edge_idx, y=y, label=label))

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
