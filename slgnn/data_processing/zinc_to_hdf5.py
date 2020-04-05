import os
import random
import pickle as pk
import logging
from collections import defaultdict

from chemreader.readers.readmol2 import Mol2
import h5py
import numpy as np
from scipy import sparse
from tqdm import tqdm


SEED = 1458907
random.seed(SEED)


class ZincToHdf5:

    def __init__(self, mol2blocks: list, n_mols=None):
        self._mol2s = mol2blocks
        if n_mols is None:
            self._n_mols = len(mol2blocks)
        else:
            self._n_mols = n_mols

    @classmethod
    def from_files(cls, fpaths: list, shuffle=False):
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
        if shuffle:
            shuffle(mol2blocks)
        return cls(mol2blocks, n_mols)

    @classmethod
    def _get_index(cls, dir_path, verbose):
        if not os.path.exists(os.path.join(dir_path, "index")):
            logging.info("Generating index...")
            index = cls.indexing(dir_path, verbose=verbose)
        else:
            with open(os.path.join(dir_path, "index"), "rb") as f:
                index = pk.load(f)
        return index

    @classmethod
    def _sample_all(cls, dir_path, mol2blocks, verbose):
        files = os.scandir(dir_path)
        n_mols = 0
        for f in (tqdm(list(files)) if verbose else files):
            if f.name.split(".")[-1] not in ["gz", "mol2"]:
                continue
            mol2 = Mol2(f.path)
            mol2blocks.extend(mol2.mol2_blocks)
            n_mols += mol2.n_mols
        random.shuffle(mol2blocks)
        return n_mols

    @classmethod
    def _group_samples(cls, samples, index):
        groups = defaultdict(list)
        for s in samples:
            groups[index[s][0]].append(index[s][1])
        return groups

    @classmethod
    def random_sample(cls, n_samples, dir_path=None, verbose=True):
        if dir_path is None:
            dir_path = "data"
        index = cls._get_index(dir_path, verbose=verbose)
        # Get Mol2Blocks randomly
        mol2blocks = list()
        # n_samples is larger than the number of samples in the dir_path
        if n_samples > index["total"]:
            logging.warning(
                "{} does not have enough samples. Got {}/{}".format(
                    dir_path, index["total"], n_samples))
            n_mols = cls._sample_all(dir_path, mol2blocks, verbose=verbose)
            return cls(mol2blocks, n_mols)
        # n_samples is smaller than the number of samples in the dir_path
        samples = random.sample(list(range(index["total"])), n_samples)
        groups = cls._group_samples(samples, index)
        it = tqdm(list(groups.items())) if verbose else groups.items()
        for fname, block_indices in it:
            file_path = os.path.join(dir_path, fname)
            mol2 = Mol2(file_path)
            for id in block_indices:
                mol2blocks.append(mol2.mol2_blocks[id])
        return cls(mol2blocks, n_samples)

    @classmethod
    def random_sample_without_index(cls, n_samples, dir_path, verbose=True):
        # Get Mol2Blocks randomly
        mol2blocks = list()
        files = [item for item in os.scandir(dir_path) if
                 item.name.split(".")[-1] in ["gz", "mol2"]]
        if len(files) < n_samples:
            logging.warning(
                "{} does not have enough samples. Got {}/{}".format(
                    dir_path, len(files), n_samples))
            samples = files
        else:
            samples = random.sample(files, n_samples)
        it = tqdm(samples) if verbose else samples
        counter = 0
        for sample in it:
            try:
                mol2blocks.append(Mol2(sample.path).mol2_blocks[0])
            except IndexError:
                logging.warning("{} is an empty file.".format(sample.path))
                continue
            counter += 1
        return cls(mol2blocks, counter)

    @property
    def n_mols(self):
        return self._n_mols

    def save_hdf5(self, out_path):
        h5f = h5py.File(out_path, "w")
        dt = h5py.string_dtype(encoding="utf-8")
        smiles = list()
        a_matrices = list()
        atom_feats = list()
        bond_feats = list()
        for block in self._mol2s:
            smiles.append(block.to_smiles(isomeric=True))
            graph = block.to_graph(sparse=True, pad_atom=70, pad_bond=100)
            a_matrices.append(graph["adjacency"])
            atom_feats.append(graph["atom_features"])
            bond_feats.append(graph["bond_features"])
        h5f.create_dataset("smiles",
                           data=np.array(smiles).astype(bytes),
                           dtype=dt,
                           chunks=True)
        h5f.create_dataset("atom_features",
                           data=np.array(atom_feats),
                           dtype=np.float16,
                           chunks=True)
        h5f.create_dataset("bond_features",
                           data=np.array(bond_feats),
                           dtype=np.int8,
                           chunks=True)
        # Save sparse adjacency matrices
        adj = h5f.create_group("adjacency_matrices")
        for i, mat in enumerate(a_matrices):
            adj.create_dataset("data_{}".format(i), data=mat.data)
            adj.create_dataset("indptr_{}".format(i), data=mat.indptr)
            adj.create_dataset("indices_{}".format(i), data=mat.indices)
            adj.attrs["shape_{}".format(i)] = mat.shape
        h5f.attrs["total"] = self.n_mols
        h5f.close()

    @staticmethod
    def indexing(dir_path, verbose=True):
        """ Indexing the files. Speed up hdf5 file creation.

        Args:
            dir_path (str): path to the directory storing all the .mol2 files.
            verbose (bool): if to show the progress bar.

        Return:
            dict: the generated index dict

        Output:
            A binary index file under the dir_path.
        """
        files = list(os.scandir(dir_path))
        index = dict()
        indexer = 0
        for f in (tqdm(files) if verbose else files):
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
        return index


class Hdf5Loader:

    def __init__(self, path):
        self.path = path

    def load_adjacency_matrices(self, n=None):
        h5f = h5py.File(self.path, "r")
        adj = h5f["adjacency_matrices"]
        matrices = list()
        if n is None:
            n = self.total
        for i in range(n):
            mat = sparse.csr_matrix(
                (adj["data_{}".format(i)][:],
                 adj["indices_{}".format(i)][:],
                 adj["indptr_{}".format(i)][:]),
                adj.attrs["shape_{}".format(i)])
            matrices.append(mat)
        h5f.close()
        return matrices

    def load_atom_features(self, n=None):
        if n is None:
            n = self.total
        with h5py.File(self.path, "r") as h5f:
            atom_features = h5f["atom_features"][:n]
        return atom_features

    def load_bond_features(self, n=None):
        if n is None:
            n = self.total
        with h5py.File(self.path, "r") as h5f:
            bond_features = h5f["bond_features"][:n]
        return bond_features

    @property
    def total(self):
        with h5py.File(self.path, "r") as h5f:
            return h5f.attrs["total"]
