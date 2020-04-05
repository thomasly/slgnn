import unittest
import os
import pickle as pk

from scipy import sparse

from ..data_processing.zinc_to_hdf5 import ZincToHdf5, Hdf5Loader


class TestZincReading(unittest.TestCase):

    def setUp(self):
        self.invalid_gzips = [f.path for f in os.scandir("test_data")]
        self.gzips = [
            f.path for f in os.scandir("test_data") if f.name.endswith("gz")]
        self.test_file_names = [
            "AAAARN.xaa.mol2.gz",
            "AAAARO.xaa.mol2.gz",
            "AAABRN.xaa.mol2.gz",
            # "HHADRO.xaa.mol2.gz",
        ]

    def test_files_from_list(self):
        with self.assertRaisesRegex(ValueError, "^.* is not a valid path.$"):
            ZincToHdf5.from_files(self.invalid_gzips)
        zth = ZincToHdf5.from_files(self.gzips)
        self.assertIsInstance(zth, ZincToHdf5)
        self.assertTrue(hasattr(zth, "_n_mols"))
        self.assertGreater(zth.n_mols, 0)
        self.assertEqual(zth.n_mols, zth._n_mols)

    def test_files_from_n_samples(self):
        if os.path.exists("test_data/index"):
            os.remove("test_data/index")
        zth = ZincToHdf5.random_sample(
            1000, dir_path="test_data", verbose=False)
        self.assertTrue(hasattr(zth, "_mol2s"))
        self.assertEqual(len(zth._mol2s), 1000)
        self.assertTrue(hasattr(zth, "_n_mols"))
        self.assertEqual(zth.n_mols, 1000)
        # number of samples too big
        with self.assertLogs() as cm:
            zth = ZincToHdf5.random_sample(
                10000, dir_path="test_data", verbose=False)
            self.assertLess(zth.n_mols, 10000)
            self.assertGreater(zth.n_mols, 0)
        self.assertIn(
            "does not have enough samples.", cm.output[0])

    def test_files_from_n_samples_wo_index(self):
        zth = ZincToHdf5.random_sample_without_index(
            1000,
            dir_path="test_data/splitted",
            verbose=True
        )
        self.assertTrue(hasattr(zth, "_mol2s"))
        self.assertEqual(len(zth._mol2s), 1000)
        self.assertTrue(hasattr(zth, "_n_mols"))
        self.assertEqual(zth.n_mols, 1000)
        # number of samples too big
        with self.assertLogs() as cm:
            zth = ZincToHdf5.random_sample_without_index(
                10000, dir_path="test_data/splitted", verbose=True)
            self.assertLess(zth.n_mols, 10000)
            self.assertGreater(zth.n_mols, 0)
        self.assertIn(
            "does not have enough samples.", cm.output[0])

    def test_indexing(self):
        self.assertIsInstance(ZincToHdf5.__dict__["indexing"], staticmethod)
        # remove the old testing file
        if os.path.exists("test_data/index"):
            os.remove("test_data/index")
        ZincToHdf5.indexing("test_data", verbose=False)
        self.assertTrue(os.path.exists("test_data/index"))
        with open("test_data/index", "rb") as f:
            index = pk.load(f)
        # assert the total number is there
        index["total"]
        # assert the number of entries is correct
        index[index["total"]-1]
        with self.assertRaises(KeyError):
            index[index["total"]]


class TestHdf5Loader(unittest.TestCase):

    def setUp(self):
        if os.path.exists("test_data/test.hdf5"):
            os.remove("test_data/test.hdf5")
        if os.path.exists("test_data/index"):
            os.remove("test_data/index")
        zth = ZincToHdf5.random_sample(
            1000, dir_path="test_data", verbose=False)
        zth.save_hdf5("test_data/test.hdf5")
        self.assertTrue(os.path.exists("test_data/test.hdf5"))
        self.loader = Hdf5Loader("test_data/test.hdf5")

    def test_attributes(self):
        self.assertEqual(self.loader.total, 1000)

    def test_read_sparse_adjacency_matrices(self):
        matrices = self.loader.load_adjacency_matrices(1000)
        self.assertIsInstance(matrices[0], sparse.csr_matrix)
        self.assertEqual(matrices[0].shape, (70, 70))

    def test_read_atom_bond_features(self):
        atom_feat = self.loader.load_atom_features()
        bond_feat = self.loader.load_bond_features()
        self.assertEqual(atom_feat.shape, (1000, 70, 7))
        self.assertEqual(bond_feat.shape, (1000, 100,))
        atom_feat = self.loader.load_atom_features(10)
        bond_feat = self.loader.load_bond_features(10)
        self.assertEqual(atom_feat.shape, (10, 70, 7))
        self.assertEqual(bond_feat.shape, (10, 100,))
