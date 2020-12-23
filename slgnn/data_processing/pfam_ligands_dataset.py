import os
import tarfile

import torch
import torch.nn.functional as F
from torch_geometric.data import Data, InMemoryDataset
from scipy.spatial import distance_matrix
from chemreader.readers.basereader import GraphFromRDKitMol
from rdkit import Chem
import numpy as np
from tqdm import tqdm


class ThreeDimensionOneHot:
    def __init__(self, num_bins):
        """ Convert distance matrix from continuous value to descrete bins

        Args:
            num_bins (int): number of bins to use.
        """
        self.num_bins = num_bins

    def __call__(self, data):
        new_y = torch.round(data.y)
        new_y[new_y >= self.num_bins - 3] = self.num_bins - 2
        new_y[new_y == -1] = self.num_bins - 1
        new_y = (torch.arange(self.num_bins) == new_y[..., None]).type(torch.float)
        data.y = new_y
        return data

    def __repr__(self):
        return "{}(num_bins={})".format(self.__class__.__name__, self.num_bins)


class PfamLigands(InMemoryDataset):
    def __init__(self, verbose=False, root=None, **kwargs):
        if root is None:
            root = os.path.join("data", "pfam")
        self.verbose = verbose
        super().__init__(root, **kwargs)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        raw_file_name = "ligand_ordered.tar.gz"
        return [raw_file_name]

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

        Args:
            verbose (int): Whether show the progress bar. Not showing if verbose == 0,
                show otherwise.
        """

        data_list = list()
        pb = tqdm(self._get_mols()) if self.verbose else self._get_mols()
        for mol, y in pb:
            x, edge_idx = self._graph_helper(mol)
            y = torch.tensor(y, dtype=torch.float)
            pad_len = 70 - y.size(0)
            y = F.pad(input=y, pad=(0, pad_len, 0, pad_len), mode="constant", value=-1)
            data_list.append(Data(x=x, edge_index=edge_idx, y=y))

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def _graph_helper(self, mol):
        graph = GraphFromRDKitMol(mol).to_graph(sparse=True)
        x = torch.tensor(graph["atom_features"], dtype=torch.float)
        edge_idx = graph["adjacency"].tocoo()
        edge_idx = torch.tensor([edge_idx.row, edge_idx.col], dtype=torch.long)
        return x, edge_idx

    def _get_mols(self):
        with tarfile.open(self.raw_paths[0], "r:gz") as tarf:
            for member in tarf.getmembers():
                pdbf = tarf.extractfile(member)
                if pdbf is None:
                    continue
                mol = Chem.MolFromPDBBlock(pdbf.read().decode("utf8"))
                if mol is None:
                    continue
                conformer = mol.GetConformer(0)
                atom_coors = list()
                for atom in range(mol.GetNumAtoms()):
                    pos = conformer.GetAtomPosition(atom)
                    atom_coors.append((pos.x, pos.y, pos.z))
                dis_matrix = distance_matrix(np.array(atom_coors), np.array(atom_coors))
                yield mol, dis_matrix
