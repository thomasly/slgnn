import unittest

import numpy as np
import scipy.sparse as sp
import torch

from slgnn.models.gcn.utils import load_encoder_data, load_classifier_data
from slgnn.config import PAD_ATOM


class TestEncoderDataLoading(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.train, cls.valid = load_encoder_data(
            "test_data/zinc_ghose_1000.hdf5")

    def test_correctly_loaded(self):
        self.train["features"]
        self.train["adj"]
        self.valid["features"]
        self.valid["adj"]

    def test_correctly_splitted(self):
        self.assertEqual(
            len(self.train["features"]), 9 * len(self.valid["features"]))
        self.assertEqual(
            len(self.train["features"]), 9 * len(self.valid["features"]))

    def test_features(self):
        self.assertEqual(self.train["features"].shape[2], 4)


class TestTox21DataLoading(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.train, cls.valid, cls.test, cls.n_classes = load_classifier_data(
            "test_data/tox21.csv.gz",
            training_ratio=0.8,
            label_cols=['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase',
                        'NR-ER', 'NR-ER-LBD', 'NR-PPAR-gamma', 'SR-ARE',
                        'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53'])

    def test_loading(self):
        for data in ["features", "labels", "adj"]:
            self.train[data]
            self.valid[data]
            self.test[data]

    def test_ratio(self):
        self.assertAlmostEqual(len(self.train["adj"]),
                               9 * len(self.valid["adj"]),
                               delta=10)
        len_training_set = len(self.train["adj"]) + len(self.valid["adj"])
        self.assertAlmostEqual(len_training_set / 0.8,
                               len(self.test["adj"]) / 0.2,
                               delta=10)

    def test_features(self):
        self.assertEqual(self.train["features"].shape[1], PAD_ATOM)
        self.assertEqual(self.train["features"].shape[2], 4)
        self.assertIsInstance(self.train["features"], torch.Tensor)

    def test_adj(self):
        self.assertEqual(self.train["adj"][0].shape, (PAD_ATOM, PAD_ATOM))
        self.assertIsInstance(self.train["adj"][0], torch.Tensor)

    def test_labels(self):
        self.assertEqual(self.train["labels"].shape[1], 12)
        self.assertIsInstance(self.train["labels"][0], torch.Tensor)

    def test_n_classes(self):
        self.assertEqual(self.n_classes, 12)


class TestHIVDataset(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.train, cls.valid, cls.test, cls.n_classes = load_classifier_data(
            "test_data/HIV.csv",
            training_ratio=0.8,
            label_cols=["HIV_active"])

    def test_loading(self):
        for data in ["features", "labels", "adj"]:
            self.train[data]
            self.valid[data]
            self.test[data]

    def test_ratio(self):
        self.assertAlmostEqual(len(self.train["adj"]),
                               9 * len(self.valid["adj"]),
                               delta=10)
        len_training_set = len(self.train["adj"]) + len(self.valid["adj"])
        self.assertAlmostEqual(len_training_set / 0.8,
                               len(self.test["adj"]) / 0.2,
                               delta=10)

    def test_features(self):
        self.assertEqual(self.train["features"].shape[1], PAD_ATOM)
        self.assertEqual(self.train["features"].shape[2], 4)
        self.assertIsInstance(self.train["features"], torch.Tensor)

    def test_adj(self):
        self.assertEqual(self.train["adj"][0].shape, (PAD_ATOM, PAD_ATOM))
        self.assertIsInstance(self.train["adj"][0], torch.Tensor)

    def test_labels(self):
        self.assertEqual(self.train["labels"].shape[1], 1)
        self.assertIsInstance(self.train["labels"][0], torch.Tensor)

    def test_n_classes(self):
        self.assertEqual(self.n_classes, 1)
