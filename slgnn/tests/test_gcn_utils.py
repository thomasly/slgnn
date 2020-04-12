from unittest import TestCase

from slgnn.models.gcn.utils import get_filtered_fingerprint


class TestPubchemFingerprint(TestCase):

    def setUp(self):
        self.sm = "CCC(CC)COC(=O)[C@H](C)N[P@](=O)(OC[C@@H]1[C@H]"\
                  "([C@H]([C@](O1)(C#N)C2=CC=C3N2N=CN=C3N)O)O)OC4=CC=CC=C4"

    def test_fp_len(self):
        filtered_fp = get_filtered_fingerprint(self.sm)
        self.assertEqual(len(filtered_fp), 740)
