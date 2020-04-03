import os
import random
import pickle as pk

from chemreader.readers.readmol2 import Mol2
import h5py
from tqdm import tqdm


SEED = 1458907
random.seed(SEED)


class ZincToHdf5:

    def __init__(self, mol2blocks: list, n_mols=None):
        self._mol2s = mol2blocks
        self._n_mols = n_mols

    @classmethod
    def from_files(cls, fpaths: list):
        for f in fpaths:
            if f.split(".")[-1] not in ["mol2", "gz"]:
                raise ValueError("{} is not a valid path.".format(f))
        mol2blocks = list()
        n_mols = 0
        for path in fpaths:
            mol2 = Mol2(path)
            blocks = mol2.mol2_blocks
            mol2blocks.extend(blocks)
            n_mols += mol2.n_mols
        return cls(mol2blocks, n_mols)

    @classmethod
    def random_sample(cls, n_samples, dir_path=None):
        if dir_path is None:
            dir_path = "data"
        files = list(os.scandir(dir_path))
        mol2blocks = set()
        while len(mol2blocks) < n_samples:
            f = random.choice(files)
            while not f.name.split(".")[-1] in ["gz", "mol2"]:
                f = random.choice(files)
            mol2 = Mol2(f.path)
            mol2blocks.add(random.choice(mol2.mol2_blocks))

    def _count_mols(self):
        counter = 0
        for f in tqdm(self._fpaths):
            if not f.split(".")[-1] in ["gz", "mol2"]:
                continue
            counter += Mol2(f).n_mols
        self._n_mols = counter

    @property
    def n_mols(self):
        if self._n_mols is None:
            respond = input(
                "This may take a while. Continue? [Y/n]")
            if respond.strip().upper() in ["", "Y", "YES"]:
                self._count_mols()
            else:
                return
        return self._n_mols

    def __call__(self, out_path: str):
        return self.save_hdf5(out_path)

    def save_hdf5(self, out_path):
        if self._n_mols is None:
            self._count_mols()
        h5f = h5py.File(out_path, "w")
        h5f.create_dataset("atom_features")
        h5f.create_dataset("bond_features")
        h5f.create_dataset("adjacency_matrices")

    @staticmethod
    def indexing(dir_path):
        """ Indexing the files. Speed up hdf5 file creation.

        Args:
            dir_path (str): path to the directory storing all the .mol2 files.

        Output:
            A binary index file under the dir_path.
        """
        files = list(os.scandir(dir_path))
        index = dict()
        indexer = 0
        for f in tqdm(files):
            if not f.name.split(".")[-1] in ["gz", "mol2"]:
                continue
            mol2 = Mol2(f.path)
            n_mols = mol2.n_mols
            for i in range(n_mols):
                index[indexer] = tuple([f.name, i])
                indexer += 1
        index["total"] = indexer
        with open(os.path.join(dir_path, "index"), "wb") as index_f:
            pk.dump(index, index_f)
