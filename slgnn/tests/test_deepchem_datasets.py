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
    ClinToxBalanced,
    HIV,
    HIVFP,
    HIVBalanced,
    Amu,
    AmuFP,
    Ellinger,
    EllingerFP,
    Mpro,
    MproFP,
    RepurposingFP,
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
        self.assertEqual(data.y.size(), (1,))
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
        self.assertEqual(data.y.size(), (1,))
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
        self.assertEqual(data.y.size(), (1,))
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

    def test_clintoxbalanced_dataset(self):
        # remove processed data
        path = osp.join("data", "DeepChem", "ClinToxBalanced", "processed")
        if osp.exists(path):
            rmtree(path)
        dataset = ClinToxBalanced()
        self.assertEqual(len(dataset), 336)
        data = dataset[0]
        self.assertEqual(data.x.size()[1], 6)
        self.assertEqual(data.y.size(), (1,))
        self.assertEqual(data.edge_index.size()[0], 2)

    def test_hiv_dataset(self):
        # # remove processed data
        # path = osp.join("data", "DeepChem", "HIV", "processed")
        # if osp.exists(path):
        #     rmtree(path)
        dataset = HIV()
        self.assertEqual(len(dataset), 41127)
        data = dataset[0]
        self.assertEqual(data.x.size()[1], 6)
        self.assertEqual(data.y.size(), (1,))
        self.assertEqual(data.edge_index.size()[0], 2)

    def test_hivfp_dataset(self):
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

    def test_hivbalanced_dataset(self):
        dataset = HIVBalanced()
        self.assertEqual(len(dataset), 4329)
        data = dataset[0]
        self.assertEqual(data.x.size()[1], 6)
        self.assertEqual(data.y.size(), (1,))
        self.assertEqual(data.edge_index.size()[0], 2)


class TestConvid19Datasets(unittest.TestCase):
    def _test_base(
        self, name, dataset, data_len, feat_size=6, label_shape=(1,), edge_index_size=2
    ):
        # path = osp.join("data", "Covid19", name, "processed")
        # if osp.exists(path):
        #     rmtree(path)
        dataset = dataset()
        self.assertEqual(len(dataset), data_len)
        data = dataset[0]
        self.assertEqual(data.x.size()[1], feat_size)
        self.assertEqual(data.y.size(), label_shape)
        self.assertEqual(data.edge_index.size()[0], edge_index_size)

    def test_amu_dataset(self):
        self._test_base("Amu", Amu, 1484)

    def test_amufp_dataset(self):
        self._test_base("AmuFP", AmuFP, 1484, label_shape=(1, FILTERED_PUBCHEM_FP_LEN))

    def test_ellinger_dataset(self):
        self._test_base("Ellinger", Ellinger, 5591)

    def test_ellingerfp_dataset(self):
        self._test_base(
            "EllingerFP", EllingerFP, 5591, label_shape=(1, FILTERED_PUBCHEM_FP_LEN)
        )

    def test_mpro_dataset(self):
        self._test_base("Mpro", Mpro, 880)

    def test_mprofp_dataset(self):
        self._test_base("MproFP", MproFP, 880, label_shape=(1, FILTERED_PUBCHEM_FP_LEN))

    def test_reproposing_dataset(self):
        self._test_base(
            "RepurposingFP",
            RepurposingFP,
            6254,
            label_shape=(1, FILTERED_PUBCHEM_FP_LEN),
        )
