import unittest

from slgnn.data_processing.pyg_datasets import JAK1Dude, JAK2Dude, JAK3Dude
from slgnn.config import FILTERED_PUBCHEM_FP_LEN


class TestDudeDatasets(unittest.TestCase):
    def test_jak1_jak2_jak3(self):
        jak = JAK1Dude()
        data = jak[0]
        self.assertEqual(data.x.size()[1], 6)
        self.assertEqual(data.y.size(), (1, FILTERED_PUBCHEM_FP_LEN))
        self.assertEqual(data.edge_index.size()[0], 2)

        jak = JAK3Dude()
        data = jak[0]
        self.assertEqual(data.x.size()[1], 6)
        self.assertEqual(data.y.size(), (1, FILTERED_PUBCHEM_FP_LEN))
        self.assertEqual(data.edge_index.size()[0], 2)

        jak = JAK2Dude()
        data = jak[0]
        self.assertEqual(data.x.size()[1], 6)
        self.assertEqual(data.y.size(), (1, FILTERED_PUBCHEM_FP_LEN))
        self.assertEqual(data.edge_index.size()[0], 2)
