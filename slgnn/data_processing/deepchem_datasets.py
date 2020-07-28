import os.path as osp
from abc import ABCMeta, abstractmethod

import torch
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
import pandas as pd
from chemreader.readers import Smiles
from tqdm import tqdm

from slgnn.models.gcn.utils import get_filtered_fingerprint


def _smiles_from_csv(path, column):
    df = pd.read_csv(path)
    return iter(df[column])


class DeepchemDataset(InMemoryDataset, metaclass=ABCMeta):
    """ The base class to convert Deepchem dataset formatted data to pytorch-geometric
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

    Attributes:
        raw_file_names: name of the raw file. If the file is not found in raw directory,
            download method will be called.
        processed_file_names: filename of processed data. If the file is not found in
            processed directory, process method will be called.
    """

    def __init__(self, root, name, transform=None, pre_transform=None, pre_filter=None):
        self.root = root
        self.name = name
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        raw_file_name = self.name + ".csv"
        return [raw_file_name]

    @property
    def processed_file_names(self):
        return ["data.pt"]

    def download(self):
        """ Get raw data and save to raw directory.
        """
        pass

    def process(self, verbose=0):
        """ The method converting SMILES and labels to graphs.

        Args:
            verbose (int): Whether show the progress bar. Not showing if verbose == 0,
                show otherwise.
        """
        smiles = self._get_smiles()
        labels = self._get_labels()
        data_list = list()
        pb = tqdm(zip(smiles, labels)) if verbose else zip(smiles, labels)
        for smi, y in pb:
            try:
                x, edge_idx = self._graph_helper(smi)
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

    @abstractmethod
    def _get_smiles(self):
        """ Method to get SMILES strings from the raw data. Must be implemented by all
        subclasses.
        """
        ...

    @abstractmethod
    def _get_labels(self):
        """ Method to get labels from the raw data. Must be implemented by all
        subclasses.
        """
        ...


class Sider(DeepchemDataset):
    """ Class for Sider dataset.
    """

    def __init__(self, root=None, name="sider"):
        if root is None:
            root = osp.join("data", "DeepChem", "Sider")
        super().__init__(root=root, name=name)

    def _get_smiles(self):
        return _smiles_from_csv(self.raw_paths[0], "smiles")

    def _get_labels(self):
        df = pd.read_csv(self.raw_paths[0])
        for row in df.iterrows():
            yield torch.tensor(list(row[1][1:]), dtype=torch.long)[None, :]

    def process(self, verbose=0):
        super().process(verbose)


class SiderFP(Sider):
    """ Class of Sider dataset with fingerprints as labels.
    """

    def __init__(self, root=None, name="sider"):
        if root is None:
            root = osp.join("data", "DeepChem", "SiderFP")
        super().__init__(root=root, name=name)

    def _get_labels(self):
        df = pd.read_csv(self.raw_paths[0])
        for smi in df["smiles"]:
            fp = get_filtered_fingerprint(smi)
            yield torch.tensor(list(fp), dtype=torch.long)[None, :]

    def process(self, verbose=1):
        super().process(verbose)


class BACE(DeepchemDataset):
    """ Class for BACE dataset
    """

    def __init__(self, root=None, name="bace"):
        if root is None:
            root = osp.join("data", "DeepChem", "BACE")
        super().__init__(root=root, name=name)

    def _get_smiles(self):
        return _smiles_from_csv(self.raw_paths[0], "mol")

    def _get_labels(self):
        df = pd.read_csv(self.raw_paths[0])
        for lb in df["Class"]:
            yield torch.tensor([lb], dtype=torch.long)

    def process(self, verbose=0):
        super().process(verbose)


class BACEFP(BACE):
    """ Class of BACE dataset with fingerprints as labels.
    """

    def __init__(self, root=None, name="bace"):
        if root is None:
            root = osp.join("data", "DeepChem", "BACEFP")
        super().__init__(root=root, name=name)

    def _get_labels(self):
        df = pd.read_csv(self.raw_paths[0])
        for smi in df["mol"]:
            fp = get_filtered_fingerprint(smi)
            yield torch.tensor(list(fp), dtype=torch.long)[None, :]

    def process(self, verbose=1):
        super().process(verbose)


class BBBP(DeepchemDataset):
    """ Class of BBBP dataset.
    """

    def __init__(self, root=None, name="BBBP"):
        if root is None:
            root = osp.join("data", "DeepChem", "BBBP")
        super().__init__(root=root, name=name)

    def _get_smiles(self):
        return _smiles_from_csv(self.raw_paths[0], "smiles")

    def _get_labels(self):
        df = pd.read_csv(self.raw_paths[0])
        for lb in df["p_np"]:
            yield torch.tensor([lb], dtype=torch.long)

    def process(self, verbose=0):
        super().process(verbose)


class BBBPFP(BBBP):
    """ Class of BBBP dataset with fingerprints as labels.
    """

    def __init__(self, root=None, name="BBBP"):
        if root is None:
            root = osp.join("data", "DeepChem", "BBBPFP")
        super().__init__(root=root, name=name)

    def _get_labels(self):
        df = pd.read_csv(self.raw_paths[0])
        for smi in df["smiles"]:
            fp = get_filtered_fingerprint(smi)
            yield torch.tensor(list(fp), dtype=torch.long)[None, :]

    def process(self, verbose=1):
        super().process(verbose)


class ClinTox(DeepchemDataset):
    """ Class of ClinTox dataset.
    """

    def __init__(self, root=None, name="clintox"):
        if root is None:
            root = osp.join("data", "DeepChem", "ClinTox")
        super().__init__(root=root, name=name)

    def _get_smiles(self):
        return _smiles_from_csv(self.raw_paths[0], "smiles")

    def _get_labels(self):
        df = pd.read_csv(self.raw_paths[0])
        # for lb in df["CT_TOX"]:
        #     yield torch.tensor([lb], dtype=torch.long)
        for row in df.iterrows():
            yield torch.tensor(list(row[1][1:]), dtype=torch.long)[None, :]

    def process(self, verbose=0):
        super().process(verbose)


class ClinToxFP(ClinTox):
    """ Class of ClinTox dataset with fingerprints as labels.
    """

    def __init__(self, root=None, name="clintox"):
        if root is None:
            root = osp.join("data", "DeepChem", "ClinToxFP")
        super().__init__(root=root, name=name)

    def _get_labels(self):
        df = pd.read_csv(self.raw_paths[0])
        for smi in df["smiles"]:
            try:
                fp = get_filtered_fingerprint(smi)
            except OSError:  # Invalid SMILES input
                continue
            yield torch.tensor(list(fp), dtype=torch.long)[None, :]

    def process(self, verbose=1):
        super().process(verbose)


class ClinToxBalanced(ClinTox):
    """ Balanced ClinTox dataset. The amounts of positive and negative samples are
    equivalent.
    """

    def __init__(self, root=None, name="clintox_balanced"):
        if root is None:
            root = osp.join("data", "DeepChem", "ClinToxBalanced")
        super().__init__(root=root, name=name)

    def process(self, verbose=1):
        super().process(verbose)


class HIV(DeepchemDataset):
    """ Class of HIV dataset.
    """

    def __init__(self, root=None, name="HIV"):
        if root is None:
            root = osp.join("data", "DeepChem", "HIV")
        super().__init__(root=root, name=name)

    def _get_smiles(self):
        return _smiles_from_csv(self.raw_paths[0], "smiles")

    def _get_labels(self):
        df = pd.read_csv(self.raw_paths[0])
        for lb in df["HIV_active"]:
            yield torch.tensor([lb], dtype=torch.long)

    def process(self, verbose=0):
        super().process(verbose)


class HIVFP(HIV):
    """ Class of HIV dataset with fingerprints as labels.
    """

    def __init__(self, root=None, name="HIV"):
        if root is None:
            root = osp.join("data", "DeepChem", "HIVFP")
        super().__init__(root=root, name=name)

    def _get_labels(self):
        df = pd.read_csv(self.raw_paths[0])
        for smi in df["smiles"]:
            fp = get_filtered_fingerprint(smi)
            yield torch.tensor(list(fp), dtype=torch.long)[None, :]

    def process(self, verbose=1):
        super().process(verbose)


class HIVBalanced(HIV):
    """ Balanced HIV dataset. The amounts of positive and negative samples are
    equivalent.
    """

    def __init__(self, root=None, name="hiv_balanced"):
        if root is None:
            root = osp.join("data", "DeepChem", "HIVBalanced")
        super().__init__(root=root, name=name)

    def process(self, verbose=0):
        super().process(verbose)


class Tox21(DeepchemDataset):
    """ Class of Tox21 dataset. NA labels are filled with 2.
    """

    def __init__(self, root=None, name="tox21"):
        if root is None:
            root = osp.join("data", "DeepChem", "Tox21")
        super().__init__(root=root, name=name)

    def _get_smiles(self):
        return _smiles_from_csv(self.raw_paths[0], "smiles")

    def _get_labels(self):
        df = pd.read_csv(self.raw_paths[0])
        for row in df.iterrows():
            label = row[1][:-2].fillna(2)
            yield torch.tensor(list(label), dtype=torch.long)[None, :]

    def process(self, verbose=0):
        super().process(verbose)


class Tox21FP(Tox21):
    """ Class of Tox21 dataset with fingerprints as labels.
    """

    def __init__(self, root=None, name="tox21"):
        if root is None:
            root = osp.join("data", "DeepChem", "Tox21FP")
        super().__init__(root, name)

    def _get_labels(self):
        df = pd.read_csv(self.raw_paths[0])
        for smi in df["smiles"]:
            fp = get_filtered_fingerprint(smi)
            yield torch.tensor(list(fp), dtype=torch.long)[None, :]

    def process(self, verbose=1):
        super().process(verbose)


class ToxCast(DeepchemDataset):
    """ Class of ToxCast dataset. NA labels are filled with 2.
    """

    def __init__(self, root=None, name="toxcast_data"):
        if root is None:
            root = osp.join("data", "DeepChem", "ToxCast")
        super().__init__(root=root, name=name)

    def _get_smiles(self):
        return _smiles_from_csv(self.raw_paths[0], "smiles")

    def _get_labels(self):
        df = pd.read_csv(self.raw_paths[0])
        for row in df.iterrows():
            label = row[1][1:].fillna(2)
            yield torch.tensor(list(label), dtype=torch.long)[None, :]

    def process(self, verbose=0):
        super().process(verbose)


class ToxCastFP(ToxCast):
    """ Class of ToxCast dataset with fingerprints as labels.
    """

    def __init__(self, root=None, name="toxcast_data"):
        if root is None:
            root = osp.join("data", "DeepChem", "ToxCastFP")
        self._smiles = list()
        self._fps = list()
        super().__init__(root, name)

    def _get_smiles(self):
        df = pd.read_csv(self.raw_paths[0])
        for smi in tqdm(df["smiles"]):
            try:
                fp = get_filtered_fingerprint(smi)
            except OSError:
                continue
            fp = torch.tensor(list(fp), dtype=torch.long)[None, :]
            self._smiles.append(smi)
            self._fps.append(fp)
        return self._smiles

    def _get_labels(self):
        if len(self._fps) == 0:
            self._get_smiles()
        return self._fps

    def process(self, verbose=0):
        super().process(verbose)


class MUV(DeepchemDataset):
    """ Class of MUV dataset. NA labels are fiiled with 2.
    """

    def __init__(self, root=None, name="muv"):
        if root is None:
            root = osp.join("data", "DeepChem", "MUV")
        super().__init__(root=root, name=name)

    def _get_smiles(self):
        return _smiles_from_csv(self.raw_paths[0], "smiles")

    def _get_labels(self):
        df = pd.read_csv(self.raw_paths[0])
        for row in df.iterrows():
            label = row[1][:-2].fillna(2)
            yield torch.tensor(list(label), dtype=torch.long)[None, :]

    def process(self, verbose=1):
        super().process(verbose)


class MUVFP(MUV):
    """ Class of MUV dataset with fingerprints as labels.
    """

    def __init__(self, root=None, name="muv"):
        if root is None:
            root = osp.join("data", "DeepChem", "MUVFP")
        super().__init__(root=root, name=name)

    def _get_smiles(self):
        return _smiles_from_csv(self.raw_paths[0], "smiles")

    def _get_labels(self):
        df = pd.read_csv(self.raw_paths[0])
        for smi in df["smiles"]:
            fp = get_filtered_fingerprint(smi)
            yield torch.tensor(list(fp), dtype=torch.long)[None, :]

    def process(self, verbose=1):
        super().process(verbose)
