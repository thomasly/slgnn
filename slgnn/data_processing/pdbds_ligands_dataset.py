import os
import tarfile

import torch
import torch.nn.functional as F
from torch_geometric.data import Data, InMemoryDataset
from scipy.spatial import distance_matrix
from chem_reader.chemreader.readers.basereader import GraphFromRDKitMol
from chem_reader.chemreader.readers import PDB
from rdkit import Chem
import numpy as np
from tqdm import tqdm
import pandas as pd


class ThreeDimensionOneHot:
    def __init__(self, num_bins):
        """ Convert distance matrix from continuous value to descrete bins

        Args:
            num_bins (int): number of bins to use.
        """
        self.num_bins = num_bins

    def __call__(self, data):
        new_y = torch.round(data.y)
        # capping
        new_y[new_y >= self.num_bins - 3] = self.num_bins - 2
        # assign padding
        new_y[new_y == -1] = self.num_bins - 1
        # to 3d one-hot
        new_y = (torch.arange(self.num_bins) == new_y[..., None]).type(torch.float)
        # create data mask
        data.mask = torch.ones(*data.y.size())
        data.mask[data.y == -1] = 0
        data.mask = torch.stack([data.mask for _ in range(new_y.size(-1))], dim=2)
        data.y = new_y
        return data

    def __repr__(self):
        return "{}(num_bins={})".format(self.__class__.__name__, self.num_bins)


class PDBLigands(InMemoryDataset):
    def __init__(self, verbose=False, root=None, **kwargs):
        if root is None:
            root = os.path.join("data", "PDB_sub", "chemicals")
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


class PDBBsAndSub(InMemoryDataset):
    def __init__(self, verbose=False, root=None, **kwargs):
        """ PDB binding sites together with their substrates. The processed protein and
        substrates graphs are saved seperately. A proper loader is needed to handle
        this.
        
        Args:
            verbose (bool): if to show progress bar when processing the dataset.
            root (str): path to the dataset root directory.
            kwargs: other kwargs for InMemoryDataset class.
        """
        if root is None:
            root = os.path.join("data", "PDB_sub")
        self.verbose = verbose
        super().__init__(root, **kwargs)
        self.data_subs, self.slices_subs = torch.load(self.processed_paths[0])
        self.data_pros, self.slices_pros = torch.load(self.processed_paths[1])

    @property
    def raw_dir(self):
        return self.root

    @property
    def raw_file_names(self):
        return []

    def download(self):
        pass

    @property
    def processed_dir(self):
        return os.path.join(self.root, "bs_and_sub")

    @property
    def processed_file_names(self):
        return ["data_subs.pt", "data_pros.pt"]

    def _graph_helper(self, mol):
        graph = GraphFromRDKitMol(mol).to_graph(sparse=True)
        x = torch.tensor(graph["atom_features"], dtype=torch.float)
        edge_idx = graph["adjacency"].tocoo()
        edge_idx = torch.tensor([edge_idx.row, edge_idx.col], dtype=torch.long)
        return x, edge_idx

    def process(self):
        subs_data_list = list()
        pros_data_list = list()
        pb = self._get_pros_and_subs()
        if self.verbose:
            pb = tqdm(pb)
        for mol, y, pro_graph in pb:
            # substrate
            x, edge_idx = self._graph_helper(mol)
            y = torch.tensor(y, dtype=torch.float)
            pad_len = 70 - y.size(0)
            y = F.pad(input=y, pad=(0, pad_len, 0, pad_len), mode="constant", value=-1)
            subs_data_list.append(Data(x=x, edge_index=edge_idx, y=y))
            # protein
            x_pro = torch.tensor(pro_graph["atom_features"], dtype=torch.float)
            edge_idx_pro = pro_graph["adjacency"].tocoo()
            edge_idx_pro = torch.tensor(
                [edge_idx_pro.row, edge_idx_pro.col], dtype=torch.long
            )
            pros_data_list.append(Data(x=x_pro, edge_index=edge_idx_pro))

        #         if self.pre_filter is not None:
        #             data_list = [data for data in data_list if self.pre_filter(data)]
        #         if self.pre_transform is not None:
        #             data_list = [self.pre_transform(data) for data in data_list]
        data_subs, slices_subs = self.collate(subs_data_list)
        torch.save((data_subs, slices_subs), self.processed_paths[0])
        data_pros, slices_pros = self.collate(pros_data_list)
        torch.save((data_pros, slices_pros), self.processed_paths[1])

    def _sub2pro(self, df):
        """ Generate substrate file to protein file dict from the reference dataframe.
        """
        ref = dict()
        for _, row in df.iterrows():
            ref[row["ligand_pdb_file"]] = row["binding_site_coodinate_file"]
        return ref

    def _get_pros_and_subs(self):
        subs_file = os.path.join(self.root, "chemicals", "ligand_ordered.tar.gz")
        ref_file = os.path.join(self.root, "nometal-pdbtoligand.txt")
        ref_df = pd.read_csv(ref_file, sep="\t")
        ref_dict = self._sub2pro(ref_df)
        pro_file_root = os.path.join(self.root, "protein_binding_pockets", "bs-pdbs")
        with tarfile.open(subs_file, "r:gz") as tarf:
            for member in tarf.getmembers():
                if not member.name.endswith(".pdb"):
                    continue
                pdbf = tarf.extractfile(member)
                if pdbf is None:
                    continue
                ligand_pdb_file = member.name.split("/")[1]
                try:
                    pro_pdb_file = ref_dict[ligand_pdb_file]
                except KeyError:
                    continue

                # get substrate mol and distance_matrix
                mol = Chem.MolFromPDBBlock(pdbf.read().decode("utf8"))
                if mol is None:
                    continue
                try:
                    conformer = mol.GetConformer(0)
                except ValueError:  # Bad Conformer Id
                    continue
                atom_coors = list()
                for atom in range(mol.GetNumAtoms()):
                    pos = conformer.GetAtomPosition(atom)
                    atom_coors.append((pos.x, pos.y, pos.z))
                dis_matrix = distance_matrix(np.array(atom_coors), np.array(atom_coors))

                # get protein graph
                pro_file = os.path.join(pro_file_root, pro_pdb_file)
                try:
                    pro_graph = PDB(pro_file).to_graph(
                        include_coordinates=True, sparse=True
                    )
                except AttributeError:  # PDB file cannot be converted to RDKit Mol
                    continue
                except ValueError:  # molecule does not have conformer
                    continue

                yield mol, dis_matrix, pro_graph


class PDBSubstrates(PDBBsAndSub):
    def __init__(self, verbose=False, root=None, **kwargs):
        super().__init__(verbose=verbose, root=root, **kwargs)
        self.data, self.slices = self.data_subs, self.slices_subs

    def process(self):
        super().process()


class PDBProteins(PDBBsAndSub):
    def __init__(self, verbose=False, root=None, **kwargs):
        super().__init__(verbose=verbose, root=root, **kwargs)
        self.data, self.slices = self.data_pros, self.slices_pros

    def process(self):
        super().process()
