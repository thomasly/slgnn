import os.path as osp
import unittest
from shutil import rmtree

from slgnn.data_processing.deepchem_datasets import Sider, SiderFP
from slgnn.config import PUBCHEM_FP_LEN


class TestDeepChemDatasets(unittest.TestCase):
    def test_sider_dataset(self):
        # remove processed data
        path = osp.join("data", "DeepChem", "Sider", "processed")
        if osp.exists(path):
            rmtree(path)
        dataset = Sider()
        self.assertEqual(len(dataset), 1427)
        data = dataset[0]
        self.assertEqual(data.x.size()[1], 6)
        self.assertEqual(data.y.size(), (1, 27))
        self.assertEqual(data.edge_index.size()[0], 2)

    def test_sider_fp_dataset(self):
        # remove processed data
        path = osp.join("data", "DeepChem", "SiderFP", "processed")
        if osp.exists(path):
            rmtree(path)
        dataset = SiderFP()
        self.assertEqual(len(dataset), 1427)
        data = dataset[0]
        self.assertEqual(data.x.size()[1], 6)
        self.assertEqual(data.y.size(), (1, PUBCHEM_FP_LEN))
        self.assertEqual(data.edge_index.size()[0], 2)
