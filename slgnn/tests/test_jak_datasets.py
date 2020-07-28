import os.path as osp
import unittest
from shutil import rmtree

from slgnn.data_processing.pyg_datasets import JAK1, JAK2, JAK3


class TestJAKDatasets(unittest.TestCase):
    def test_jak_dataset(self):
        # remove processed data
        for name in ["JAK1", "JAK2", "JAK3"]:
            path = osp.join("data", "JAK", "graphs", name, "processed")
            if osp.exists(path):
                rmtree(path)
        for Ds, length in zip([JAK1, JAK2, JAK3], [3717, 5853, 3520]):
            dataset = Ds()
            self.assertEqual(len(dataset), length)
            data = dataset[0]
            self.assertEqual(data.x.size()[1], 6)
            self.assertEqual(data.y.size(), (1,))
            self.assertFalse(any(dataset.data.y == -1))
            self.assertEqual(data.edge_index.size()[0], 2)
