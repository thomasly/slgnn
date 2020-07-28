""" **Datasets related to Covid-19.**
"""

import os.path as osp

import torch
from rdkit.Chem import SDMolSupplier, MolToSmiles
import pandas as pd

from .deepchem_datasets import DeepchemDataset
from slgnn.models.gcn.utils import get_filtered_fingerprint


class Amu(DeepchemDataset):
    """ Amu dataset. FDA-approved compounds screened against SARS-CoV-2 in vitro.
    1,484 compounds, 88 hits.
    """

    def __init__(self, root=None, name="amu_sars_cov_2_in_vitro"):
        if root is None:
            root = osp.join("data", "Covid19", "Amu")
        self.root = root
        self.name = name
        super().__init__(root=root, name=name)

    def _get_smiles(self):
        df = pd.read_csv(self.raw_paths[0])
        smiles = iter(df["smiles"])
        return smiles

    def _get_labels(self):
        df = pd.read_csv(self.raw_paths[0])
        for lb in df["fda"]:
            yield torch.tensor([lb], dtype=torch.long)

    def process(self, verbose=0):
        super().process(verbose)


class AmuFP(Amu):
    """ Amu dataset with fingerprints as labels.
    """

    def __init__(self, root=None, name="amu_sars_cov_2_in_vitro"):
        if root is None:
            root = osp.join("data", "Covid19", "AmuFP")
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


class Ellinger(DeepchemDataset):
    """ Ellinger dataset. A large scale drug repurposing collection of compounds
    screened against SARS-CoV-2 in vitro. 5,632 compounds, 67 hits.
    """

    def __init__(self, root=None, name="ellinger"):
        if root is None:
            root = osp.join("data", "Covid19", "Ellinger")
        self.root = root
        self.name = name
        super().__init__(root=root, name=name)

    def _get_smiles(self):
        df = pd.read_csv(self.raw_paths[0])
        smiles = iter(df["Smiles"])
        return smiles

    def _get_labels(self):
        df = pd.read_csv(self.raw_paths[0])
        for lb in df["activity"]:
            yield torch.tensor([lb], dtype=torch.long)

    def process(self, verbose=0):
        super().process(verbose)


class EllingerFP(Ellinger):
    """ Ellinger dataset with fingerprints as labels.
    """

    def __init__(self, root=None, name="ellinger"):
        if root is None:
            root = osp.join("data", "Covid19", "EllingerFP")
        super().__init__(root=root, name=name)

    def _get_labels(self):
        df = pd.read_csv(self.raw_paths[0])
        for smi in df["Smiles"]:
            try:
                fp = get_filtered_fingerprint(smi)
            except OSError:  # Invalid SMILES input
                continue
            yield torch.tensor(list(fp), dtype=torch.long)[None, :]

    def process(self, verbose=1):
        super().process(verbose)


class Mpro(DeepchemDataset):
    """ Mpro dataset. fragments screened for 3CL protease binding using crystallography
    techniques. 880 compounds, 78 hits.
    """

    def __init__(self, root=None, name="mpro_xchem"):
        if root is None:
            root = osp.join("data", "Covid19", "Mpro")
        self.root = root
        self.name = name
        super().__init__(root=root, name=name)

    def _get_smiles(self):
        df = pd.read_csv(self.raw_paths[0])
        smiles = iter(df["smiles"])
        return smiles

    def _get_labels(self):
        df = pd.read_csv(self.raw_paths[0])
        for lb in df["activity"]:
            yield torch.tensor([lb], dtype=torch.long)

    def process(self, verbose=0):
        super().process(verbose)


class MproFP(Mpro):
    """ Mpro dataset with fingerprints as labels.
    """

    def __init__(self, root=None, name="mpro_xchem"):
        if root is None:
            root = osp.join("data", "Covid19", "MproFP")
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


class RepurposingFP(DeepchemDataset):
    """ Drug repurposing dataset from Broad Institue with fingerprints as labels. Can
    be used for pretraining.
    """

    def __init__(self, root=None, name="repurposing_drugs_smiles"):
        if root is None:
            root = osp.join("data", "Covid19", "RepurposingFP")
        self.root = root
        self.name = name
        super().__init__(root=root, name=name)

    @property
    def raw_file_names(self):
        raw_file_name = self.name + ".tsv"
        return [raw_file_name]

    def _get_smiles(self):
        df = pd.read_csv(self.raw_paths[0], sep="\t")
        smiles = iter(df["smiles"])
        return smiles

    def _get_labels(self):
        df = pd.read_csv(self.raw_paths[0], sep="\t")
        for smi in df["smiles"]:
            try:
                fp = get_filtered_fingerprint(smi)
            except OSError:  # Invalid SMILES input
                continue
            yield torch.tensor(list(fp), dtype=torch.long)[None, :]

    def process(self, verbose=1):
        super().process(verbose)


class AntiviralFP(DeepchemDataset):
    """ known anti-viral drugs and related chemical compounds that are structurally
    similar to known antivirals from CAS with fingerprints as labels. Can be used for
    pretraining.
    """

    def __init__(self, root=None, name="antiviral_updated"):
        if root is None:
            root = osp.join("data", "Covid19", "AntiviralFP")
        self.root = root
        self.name = name
        super().__init__(root=root, name=name)

    @property
    def raw_file_names(self):
        raw_file_name = self.name + ".sdf"
        return [raw_file_name]

    def _get_smiles(self):
        supplier = SDMolSupplier(self.raw_paths[0])
        for mol in supplier:
            if mol is None:
                continue
            yield MolToSmiles(mol)

    def _get_labels(self):
        smiles = self._get_smiles
        for smi in smiles():
            fp = get_filtered_fingerprint(smi)
            yield torch.tensor(list(fp), dtype=torch.long)[None, :]

    def process(self, verbose=1):
        super().process(verbose)
