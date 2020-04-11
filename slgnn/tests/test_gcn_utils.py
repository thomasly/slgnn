from unittest import TestCase

from slgnn.models.gcn.utils import get_filtered_fingerprint
from PyFingerprint.All_Fingerprint import get_fingerprint


class TestPubchemFingerprint(TestCase):

    def setUp(self):
        self.sm = "CCC(CC)COC(=O)[C@H](C)N[P@](=O)(OC[C@@H]1[C@H]"\
                  "([C@H]([C@](O1)(C#N)C2=CC=C3N2N=CN=C3N)O)O)OC4=CC=CC=C4"

    def test_fp_len(self):
        pubchem_fp = get_fingerprint(
            self.sm, fp_type="pubchem", output='vector')
        filtered_fp, length = get_filtered_fingerprint(self.sm)
        self.assertEqual(len(filtered_fp), length)
        self.assertLess(length, len(pubchem_fp))
        self.assertEqual(length, 740)
