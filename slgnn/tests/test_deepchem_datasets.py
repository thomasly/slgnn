import os.path as osp
import unittest
from shutil import rmtree

from slgnn.data_processing.deepchem_datasets import (
    Sider,
    SiderFP,
    BACE,
    BACEFP,
    BBBP,
    BBBPFP,
    ClinTox,
    ClinToxFP,
    HIV,
    HIVFP,
)
from slgnn.config import FILTERED_PUBCHEM_FP_LEN


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
        self.assertEqual(data.y.size(), (1, FILTERED_PUBCHEM_FP_LEN))
        self.assertEqual(data.edge_index.size()[0], 2)

    def test_bace_dataset(self):
        # remove processed data
        path = osp.join("data", "DeepChem", "BACE", "processed")
        if osp.exists(path):
            rmtree(path)
        dataset = BACE()
        self.assertEqual(len(dataset), 1513)
        data = dataset[0]
        self.assertEqual(data.x.size()[1], 6)
        self.assertEqual(data.y.size(), (1, 1))
        self.assertEqual(data.edge_index.size()[0], 2)

    def test_bacefp_dataset(self):
        # remove processed data
        path = osp.join("data", "DeepChem", "BACEFP", "processed")
        if osp.exists(path):
            rmtree(path)
        dataset = BACEFP()
        self.assertEqual(len(dataset), 1513)
        data = dataset[0]
        self.assertEqual(data.x.size()[1], 6)
        self.assertEqual(data.y.size(), (1, FILTERED_PUBCHEM_FP_LEN))
        self.assertEqual(data.edge_index.size()[0], 2)

    def test_bbbp_dataset(self):
        # remove processed data
        path = osp.join("data", "DeepChem", "BBBP", "processed")
        if osp.exists(path):
            rmtree(path)
        dataset = BBBP()
        self.assertEqual(len(dataset), 2039)
        data = dataset[0]
        self.assertEqual(data.x.size()[1], 6)
        self.assertEqual(data.y.size(), (1, 1))
        self.assertEqual(data.edge_index.size()[0], 2)

    def test_bbbpfp_dataset(self):
        # remove processed data
        path = osp.join("data", "DeepChem", "BBBPFP", "processed")
        if osp.exists(path):
            rmtree(path)
        dataset = BBBPFP()
        self.assertEqual(len(dataset), 2039)
        data = dataset[0]
        self.assertEqual(data.x.size()[1], 6)
        self.assertEqual(data.y.size(), (1, FILTERED_PUBCHEM_FP_LEN))
        self.assertEqual(data.edge_index.size()[0], 2)

    def test_clintox_dataset(self):
        # remove processed data
        path = osp.join("data", "DeepChem", "ClinTox", "processed")
        if osp.exists(path):
            rmtree(path)
        dataset = ClinTox()
        self.assertEqual(len(dataset), 1477)
        data = dataset[0]
        self.assertEqual(data.x.size()[1], 6)
        self.assertEqual(data.y.size(), (1, 1))
        self.assertEqual(data.edge_index.size()[0], 2)

    def test_clintoxFP_dataset(self):
        # remove processed data
        path = osp.join("data", "DeepChem", "ClinToxFP", "processed")
        if osp.exists(path):
            rmtree(path)
        dataset = ClinToxFP()
        self.assertEqual(len(dataset), 1474)
        data = dataset[0]
        self.assertEqual(data.x.size()[1], 6)
        self.assertEqual(data.y.size(), (1, FILTERED_PUBCHEM_FP_LEN))
        self.assertEqual(data.edge_index.size()[0], 2)

    def test_HIV_dataset(self):
        # # remove processed data
        # path = osp.join("data", "DeepChem", "HIV", "processed")
        # if osp.exists(path):
        #     rmtree(path)
        dataset = HIV()
        self.assertEqual(len(dataset), 41127)
        data = dataset[0]
        self.assertEqual(data.x.size()[1], 6)
        self.assertEqual(data.y.size(), (1, 1))
        self.assertEqual(data.edge_index.size()[0], 2)

    def test_HIVFP_dataset(self):
        # # remove processed data
        # path = osp.join("data", "DeepChem", "BBBPFP", "processed")
        # if osp.exists(path):
        #     rmtree(path)
        dataset = HIVFP()
        self.assertEqual(len(dataset), 41127)
        data = dataset[0]
        self.assertEqual(data.x.size()[1], 6)
        self.assertEqual(data.y.size(), (1, FILTERED_PUBCHEM_FP_LEN))
        self.assertEqual(data.edge_index.size()[0], 2)
