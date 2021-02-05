import os
import os.path as osp
from abc import ABCMeta, abstractmethod
import multiprocessing as mp

import torch
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
import pandas as pd
from chem_reader.chemreader.readers import Smiles
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem

from slgnn.models.gcn.utils import get_filtered_fingerprint
from .utils import NumNodesFilter


def _smiles_from_csv(path, column):
    df = pd.read_csv(path)
    return iter(df[column])


class DeepchemDataset(InMemoryDataset, metaclass=ABCMeta):
    """The base class to convert Deepchem dataset formatted data to pytorch-geometric
    Dataset.

    Args:
        root (str): path to the dataset directory. (The parent directory of raw and
            processed directories.)
        name (str): name of the csv file without extension name.
        transform: optional. The transformer transforms data on the fly.
        pre_transform: optional. The transformer transforms data while converting raw
            data to graphs.
        pre_filter: optional. The filter function filters data while converting raw
            data to graphs.
        fragment_label (bool): Include fragment labels in atom features. Default: False.

    Attributes:
        raw_file_names: name of the raw file. If the file is not found in raw directory,
            download method will be called.
        processed_file_names: filename of processed data. If the file is not found in
            processed directory, process method will be called.
    """

    def __init__(
        self,
        root,
        name,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        fragment_label=False,
        fp_type=None,
    ):
        self.root = root
        self.name = name
        self.fragment_label = fragment_label
        if fp_type is not None:
            assert fp_type in ["pubchem", "ecfp"], "fp_type should be pubchem or ecfp"
        self.fp_type = fp_type
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return osp.join(self.root, "raw")

    @property
    def processed_dir(self):
        fg = "fragment_label_" if self.fragment_label else ""
        fp = "" if self.fp_type is None else self.fp_type + "_"
        return osp.join(self.root, "{}{}processed".format(fg, fp))

    @property
    def raw_file_names(self):
        raw_file_name = self.name + ".csv"
        return [raw_file_name]

    @property
    def processed_file_names(self):
        return ["data.pt"]

    def download(self):
        """Get raw data and save to raw directory."""
        pass
                
    def create_graph_data(self, idx, smi, y):
        try:
            x, edge_idx = self._graph_helper(smi)
        except AttributeError:  # SMILES invalid
            return
        if y is None:
            y = self.get_fingerprint(smi)
            if y is None: # fail to create fingerprint for the smi
                return
        return Data(x=x, edge_index=edge_idx, y=y, id=idx)

    def process(self, verbose=1):
        """The method converting SMILES and labels to graphs.

        Args:
            verbose (int): Whether show the progress bar. Not showing if verbose == 0,
                show otherwise.
        """
        data_list = list()

        pb = tqdm(self._get_data(), total=self.n_data) if verbose else self._get_data()
        for smi, y, i in pb:
            data_list.append(self.create_graph_data(i, smi, y))
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def _graph_helper(self, smi):
        graph = Smiles(smi).to_graph(sparse=True, fragment_label=self.fragment_label)
        x = torch.tensor(graph["atom_features"], dtype=torch.float)
        edge_idx = graph["adjacency"].tocoo()
        edge_idx = torch.tensor([edge_idx.row, edge_idx.col], dtype=torch.long)
        return x, edge_idx

    @abstractmethod
    def _get_data(self):
        """Method to yield SMILES and labels from the raw data. Must be implemented by
        all subclasses.
        
        Returns:
            smiles (str), label (tensor)
        """
        ...
        
    def get_fingerprint(self, smiles):
        if self.fp_type == "pubchem":
            try:
                fp = get_filtered_fingerprint(smiles)
            except OSError: # Invalid SMILES
                return
            fp = torch.tensor(list(fp), dtype=torch.long)[None, :]
        else:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 4, nBits=1024)
            fp = torch.tensor(list(fp), dtype=torch.long)[None, :]
        return fp

    
class Sider(DeepchemDataset):
    """Class for Sider dataset."""

    def __init__(self, root=None, name="sider", **kwargs):
        if root is None:
            root = osp.join("data", "DeepChem", "Sider")
        self.n_data = 1427
        super().__init__(root=root, name=name, **kwargs)
        
    def _get_data(self):
        df = pd.read_csv(self.raw_paths[0])
        for i, row in df.iterrows():
            smiles = row["smiles"].strip()
            if self.fp_type is None:
                label = torch.tensor(list(row[1:]), dtype=torch.long)[None, :]
            else:
                label = None
            yield smiles, label, i

    def process(self, verbose=1):
        super().process(verbose)


class SiderFP(Sider):
    """Class of Sider dataset with fingerprints as labels."""

    def __init__(self, root=None, name="sider", fp_type="pubchem", **kwargs):
        super().__init__(root=root, name=name, fp_type=fp_type, **kwargs)
    
    def process(self, verbose=1):
        super().process(verbose)

        
class BACE(DeepchemDataset):
    """Class for BACE dataset"""

    def __init__(self, root=None, name="bace", **kwargs):
        if root is None:
            root = osp.join("data", "DeepChem", "BACE")
        self.n_data = 1513
        super().__init__(root=root, name=name, **kwargs)
        
    def _get_data(self):
        df = pd.read_csv(self.raw_paths[0])
        for i, row in df.iterrows():
            smiles = row["mol"].strip()
            if self.fp_type is None:
                label = torch.tensor([row["Class"]], dtype=torch.long)
            else:
                label = None
            yield smiles, label, i

    def process(self, verbose=1):
        super().process(verbose)
        

class Debugging(BACE):
    
    def __init__(self, root=None, name="bace_debugging", **kwargs):
        if root is None:
            root = osp.join("data", "DeepChem", "BACE")
        self.n_data = 100
        super().__init__(root=root, name=name, **kwargs)
        self.n_data = 100
        
    @property
    def processed_dir(self):
        fg = "fragment_label_" if self.fragment_label else ""
        fp = "" if self.fp_type is None else self.fp_type + "_"
        return osp.join(self.root, "debugging_{}{}processed".format(fg, fp))

    def process(self, verbose=1):
        super().process(verbose)


class BACEFP(BACE):
    """Class of BACE dataset with fingerprints as labels."""
    
    def __init__(self, root=None, name="bace", fp_type="pubchem", **kwargs):
        super().__init__(root=root, name=name, fp_type=fp_type, **kwargs)
    
    def process(self, verbose=1):
        super().process(verbose)


class BBBP(DeepchemDataset):
    """Class of BBBP dataset."""

    def __init__(self, root=None, name="BBBP", **kwargs):
        if root is None:
            root = osp.join("data", "DeepChem", "BBBP")
        self.n_data = 2049
        super().__init__(root=root, name=name, **kwargs)
        
    def _get_data(self):
        df = pd.read_csv(self.raw_paths[0])
        for i, row in df.iterrows():
            smiles = row["smiles"].strip()
            if self.fp_type is None:
                label = torch.tensor([row["p_np"]], dtype=torch.long)
            else:
                label = None
            yield smiles, label, i

    def process(self, verbose=1):
        super().process(verbose)


class BBBPFP(BBBP):
    """Class of BBBP dataset with fingerprints as labels."""
    
    def __init__(self, root=None, name="BBBP", fp_type="pubchem", **kwargs):
        super().__init__(root=root, name=name, fp_type=fp_type, **kwargs)
    
    def process(self, verbose=1):
        super().process(verbose)


class ClinTox(DeepchemDataset):
    """Class of ClinTox dataset."""
    
    def __init__(self, root=None, name="clintox", **kwargs):
        if root is None:
            root = osp.join("data", "DeepChem", "ClinTox")
        self.n_data = 1483
        super().__init__(root=root, name=name, **kwargs)
        
    def _get_data(self):
        df = pd.read_csv(self.raw_paths[0])
        for i, row in df.iterrows():
            smiles = row["smiles"].strip()
            if self.fp_type is None:
                label = torch.tensor(list(row[1:]), dtype=torch.long)[None, :]
            else:
                label = None
            yield smiles, label, i

    def process(self, verbose=1):
        super().process(verbose)


class ClinToxFP(ClinTox):
    """Class of ClinTox dataset with fingerprints as labels."""
    
    def __init__(self, root=None, name="clintox", fp_type="pubchem", **kwargs):
        super().__init__(root=root, name=name, fp_type=fp_type, **kwargs)
    
    def process(self, verbose=1):
        super().process(verbose)


class ClinToxBalanced(ClinTox):
    """Balanced ClinTox dataset. The amounts of positive and negative samples are
    equivalent.
    """

    def __init__(self, root=None, name="clintox_balanced", **kwargs):
        if root is None:
            root = osp.join("data", "DeepChem", "ClinToxBalanced")
        super().__init__(root=root, name=name, **kwargs)

    def process(self, verbose=1):
        super().process(verbose)


class HIV(DeepchemDataset):
    """Class of HIV dataset."""
    
    def __init__(self, root=None, name="HIV", **kwargs):
        if root is None:
            root = osp.join("data", "DeepChem", "HIV")
        self.n_data = 41127
        super().__init__(root=root, name=name, **kwargs)
        
    def _get_data(self):
        df = pd.read_csv(self.raw_paths[0])
        for i, row in df.iterrows():
            smiles = row["smiles"].strip()
            if self.fp_type is None:
                label = torch.tensor([row["HIV_active"]], dtype=torch.long)
            else:
                label = None
            yield smiles, label, i
            
    def process(self, verbose=1):
        super().process(verbose)


class HIVFP(HIV):
    """Class of HIV dataset with fingerprints as labels."""
    
    def __init__(self, root=None, name="HIV", fp_type="pubchem", **kwargs):
        super().__init__(root=root, name=name, fp_type=fp_type, **kwargs)
    
    def process(self, verbose=1):
        super().process(verbose)


class HIVBalanced(HIV):
    """Balanced HIV dataset. The amounts of positive and negative samples are
    equivalent.
    """

    def __init__(self, root=None, name="hiv_balanced", **kwargs):
        if root is None:
            root = osp.join("data", "DeepChem", "HIVBalanced")
        super().__init__(root=root, name=name, **kwargs)

    def process(self, verbose=1):
        super().process(verbose)


class Tox21(DeepchemDataset):
    """Class of Tox21 dataset. NA labels are filled with 2."""
    
    def __init__(self, root=None, name="tox21", **kwargs):
        if root is None:
            root = osp.join("data", "DeepChem", "Tox21")
        self.n_data = 7830
        super().__init__(root=root, name=name, **kwargs)
        
    def _get_data(self):
        df = pd.read_csv(self.raw_paths[0])
        for i, row in df.iterrows():
            smiles = row["smiles"].strip()
            if self.fp_type is None:
                label = torch.tensor(list(row[:-2].fillna(-1)), dtype=torch.long)[None, :]
            else:
                label = None
            yield smiles, label, i

    def process(self, verbose=1):
        super().process(verbose)


class Tox21FP(Tox21):
    """Class of Tox21 dataset with fingerprints as labels."""
    
    def __init__(self, root=None, name="tox21", fp_type="pubchem", **kwargs):
        super().__init__(root=root, name=name, fp_type=fp_type, **kwargs)
    
    def process(self, verbose=1):
        super().process(verbose)


class ToxCast(DeepchemDataset):
    """Class of ToxCast dataset. NA labels are filled with 2."""
    
    def __init__(self, root=None, name="toxcast_data", **kwargs):
        if root is None:
            root = osp.join("data", "DeepChem", "ToxCast")
        self.n_data = 8596
        super().__init__(root=root, name=name, **kwargs)
        
    def _get_data(self):
        df = pd.read_csv(self.raw_paths[0])
        for i, row in df.iterrows():
            smiles = row["smiles"].strip()
            if self.fp_type is None:
                label = torch.tensor(list(row[1:].fillna(-1)), dtype=torch.long)[None, :]
            else:
                label = None
            yield smiles, label, i

    def process(self, verbose=1):
        super().process(verbose)


class ToxCastFP(ToxCast):
    """Class of ToxCast dataset with fingerprints as labels."""
    
    def __init__(self, root=None, name="toxcast_data", fp_type="pubchem", **kwargs):
        super().__init__(root=root, name=name, fp_type=fp_type, **kwargs)
    
    def process(self, verbose=1):
        super().process(verbose)


class MUV(DeepchemDataset):
    """Class of MUV dataset. NA labels are fiiled with 2."""
    
    def __init__(self, root=None, name="muv", **kwargs):
        if root is None:
            root = osp.join("data", "DeepChem", "MUV")
        self.n_data = 93087
        super().__init__(root=root, name=name, **kwargs)
        
    def _get_data(self):
        df = pd.read_csv(self.raw_paths[0])
        for i, row in df.iterrows():
            smiles = row["smiles"].strip()
            if self.fp_type is None:
                label = torch.tensor(list(row[:-2].fillna(-1)), dtype=torch.long)[None, :]
            else:
                label = None
            yield smiles, label, i

    def process(self, verbose=1):
        super().process(verbose)


class MUVFP(MUV):
    """Class of MUV dataset with fingerprints as labels."""
    
    def __init__(self, root=None, name="muv", fp_type="pubchem", **kwargs):
        super().__init__(root=root, name=name, fp_type=fp_type, **kwargs)
    
    def process(self, verbose=1):
        super().process(verbose)
