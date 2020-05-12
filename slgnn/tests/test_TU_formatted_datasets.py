from unittest import TestCase
import os
import shutil

from torch_geometric.data import DataLoader

from slgnn.data_processing.pyg_datasets import ZINCDataset


class TestTUData(TestCase):

    def setUp(self):
        self.processed = os.path.join(
            "test_data", "TU_formatted", "sampled_smiles_1000", "processed")
        if os.path.exists(self.processed):
            shutil.rmtree(self.processed)
        root = os.path.join(os.path.curdir, "test_data", "TU_formatted")
        self.dataset = ZINCDataset(
            root=root,
            name="sampled_smiles_1000",
            use_node_attr=True,
            pre_filter=None,
            pre_transform=None,
        )

    def tearDown(self):
        if os.path.exists(self.processed):
            shutil.rmtree(self.processed)

    def test_dataset_properties(self):
        dataloader = DataLoader(self.dataset, batch_size=32)
        batch = next(iter(dataloader))
        self.assertEqual(batch.num_node_features, 7)
        self.assertEqual(batch.num_graphs, 32)
        self.assertTrue(batch.is_undirected, True)
        self.assertTrue(batch.contains_self_loops, True)
        self.assertEqual(len(self.dataset), 1000)
