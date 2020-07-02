import os.path as osp
import unittest
from shutil import rmtree

from slgnn.data_processing.pyg_datasets import JAK1Dude, JAK2Dude, JAK3Dude


class TestDudeDatasets(unittest.TestCase):
    def test_jak1_jak2_jak3(self):
        for Dataset in [JAK1Dude, JAK2Dude, JAK3Dude]:
            jak = Dataset()

