import unittest
from unittest import mock
import os

from ..data_processing.zinc_to_hdf5 import ZincToHdf5


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

    @mock.patch("builtins.input")
    def test_files_from_list(self, mocked_input):
        with self.assertRaisesRegex(ValueError, "^.* is not a valid path.$"):
            ZincToHdf5(self.invalid_gzips)

        zth = ZincToHdf5(self.gzips)
        self.assertTrue(hasattr(zth, "_fpaths"))
        self.assertEqual(len(zth._fpaths), 3)
        for gzip in self.gzips:
            self.assertIn(os.path.basename(gzip), self.test_file_names)
        self.assertIsNone(zth._n_mols)
        # calculate n_mols
        # positive responses
        user_responses = ["", "Y", "yes"]
        mocked_input.side_effect = user_responses
        for _ in range(len(user_responses)):
            zth = ZincToHdf5(self.gzips)
            zth.n_mols
            self.assertIsNotNone(zth._n_mols)
        # negative responses
        user_responses = ["n", "#$%"]
        mocked_input.side_effect = user_responses
        for _ in range(len(user_responses)):
            zth = ZincToHdf5(self.gzips)
            zth.n_mols
            self.assertIsNone(zth._n_mols)

    def test_files_from_n_samples(self):
        zth = ZincToHdf5.random_sample(1000, dir_path="test_data")
        self.assertTrue(hasattr(zth, "_fpaths"))
        self.assertGreaterEqual(len(zth._fpaths), 1)
        self.assertIsNotNone(zth._n_mols)
        self.assertGreaterEqual(zth._n_mols, 1000)
        # number of samples too big
        with self.assertLogs() as cm:
            zth = ZincToHdf5.random_sample(10000, dir_path="test_data")
            self.assertLess(zth._n_mols, 10000)
        self.assertIn(
            "Target path does not have enough molecules.", cm.output[0])
