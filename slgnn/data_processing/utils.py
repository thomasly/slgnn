import random
import functools

import torch
from contextPred.chem.util import MaskAtom

# import inspect


class AtomFeaturesOneHotTransformer:
    def __init__(
        self,
        num_atom_types=23,
        num_degree=6,
        num_formal_charge=5,
        num_hybridization=8,
        num_aromatic=2,
        num_chirality=4,
    ):
        self.num_atom_types = num_atom_types
        self.num_degree = num_degree
        self.num_formal_charge = num_formal_charge
        self.num_hybridization = num_hybridization
        self.num_aromatic = num_aromatic
        self.num_charality = num_chirality
        self.feature_len = (
            num_atom_types
            + num_degree
            + num_formal_charge
            + num_hybridization
            + num_aromatic
            + num_chirality
        )

    def __call__(self, data):

        oh_atom_type = torch.zeros((data.x.size(0), self.num_atom_types))
        oh_degree = torch.zeros((data.x.size(0), self.num_degree))
        oh_formal_charge = torch.zeros((data.x.size(0), self.num_formal_charge))
        oh_hybridization = torch.zeros((data.x.size(0), self.num_hybridization))
        oh_aromatic = torch.zeros((data.x.size(0), self.num_aromatic))
        oh_charality = torch.zeros((data.x.size(0), self.num_charality))
        features = [
            oh_atom_type,
            oh_degree,
            oh_formal_charge,
            oh_hybridization,
            oh_aromatic,
            oh_charality,
        ]
        # make formal charge value positive
        data.x[data.x == -1] = 3
        data.x[data.x == -2] = 4
        data.x[:, 1][data.x[:, 1] > 5] = 5
        for dt, feat in zip(data.x.T, features):
            feat.scatter_(1, dt.unsqueeze(1).type(torch.int64), 1)
        data.x = torch.cat(features, 1)
        return data


class MaskOneHot:
    def __init__(
        self,
        num_edge_type,
        mask_rate,
        num_atom_types=23,
        num_degree=6,
        num_formal_charge=5,
        num_hybridization=8,
        num_aromatic=2,
        num_chirality=4,
        mask_edge=True,
    ):
        num_atom_features = (
            num_atom_types
            + num_degree
            + num_formal_charge
            + num_hybridization
            + num_aromatic
            + num_chirality
        )
        self.oh_transformer = AtomFeaturesOneHotTransformer(
            num_atom_types,
            num_degree,
            num_formal_charge,
            num_hybridization,
            num_aromatic,
            num_chirality,
        )
        self.mask_transformer = MaskAtom(
            num_atom_features, num_edge_type, mask_rate, mask_edge
        )

    def __call__(self, data):
        data = self.oh_transformer(data)
        data = self.mask_transformer(data)
        return data


def get_pubchem_fingerprint(smiles):
    """Generate pubchem fingerprint from SMILES.

    Args:
        smiles (str): the SMILES string
    """
    from PyFingerprint.All_Fingerprint import get_fingerprint

    return get_fingerprint(smiles, fp_type="pubchem", output="vector")


class fix_random_seed:
    """Decorator fix the random seed within the span of a function. The random seed is
    revert to None after the function's execution.

    Args:
        seed (int): the random seed.
    """

    def __init__(self, seed=0):
        self._seed = seed

    def __call__(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            random.seed(self._seed)
            ret = func(*args, **kwargs)
            random.seed()
            return ret

        return wrapper
