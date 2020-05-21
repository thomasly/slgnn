from unittest import TestCase

from slgnn.data_processing.clustering import Cluster


class TestClustering(TestCase):
    def setUp(self):
        self.smiles = [
            "CC1:C:C:C:C([C@@H]([NH2+]C[C@@H](C)O)C(C)(C)C):C:1",
            "CC1:C:C:C:C([C@@H]([NH2+]C[C@@H](C)O)C(C)(C)C):C:1",
            "CC1:C:C:C:C([C@@H]([NH2+]C[C@@H](C)O)C(C)(C)C):C:1",  # repeat 3
            "CCCNS(=O)(=O)C1:C:C:C(NC(=O)C(=O)N2CCC[C@](C)(O)C2):C:C:1",
            "CCCNS(=O)(=O)C1:C:C:C(NC(=O)C(=O)N2CCC[C@](C)(O)C2):C:C:1",
            "CCCNS(=O)(=O)C1:C:C:C(NC(=O)C(=O)N2CCC[C@](C)(O)C2):C:C:1",
            "CCCNS(=O)(=O)C1:C:C:C(NC(=O)C(=O)N2CCC[C@](C)(O)C2):C:C:1",  # repeat 4
            "CC(C)(C)[C@H]1C[C@@H](C(=O)N[C@@H]2CC[C@H]3C[C@H]3C2)C1",
            "CCC[C@@]1(NC(=O)N[C@@H](C)C2:C:C:C:C:C:2OC(F)F)CCOC1",
            "CC1:N:C:C(C(=O)N[C@H](C)CN[C@H](C)C2:N:N:C(C):S:2):S:1",
            "O=C(CSCC1:C:C:C:C(Cl):C:1)NC1:C:C:C(S(=O)(=O)[N-]C2:N:C:C:C:N:2):C:C:1",
            "CC[C@@H](C(=O)NCC(=O)NC1CCCC1)C1:C:C:C:C(C(F)(F)F):C:1",
            "COCC1:C:C:C(NC(=O)[C@@H]2CCC3:C:C(OC):C:C:C:32):C:C:1OC",
            "COCCN(C(=O)CNC(=O)OCC(F)(F)F)[C@@H]1CCOC1",
            "CNC(=O)CCCCC(=O)N1CCC(OC)(OC)[C@@H](O)C1",
        ]

    def test_correct_clustering(self):
        cluster = Cluster(threshold=0.5)
        cluster.clustering(self.smiles)
        self.assertEqual(len(cluster.clusters), 10)
