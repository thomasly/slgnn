from unittest import TestCase

from slgnn.models.gcn.utils import get_filtered_fingerprint
from slgnn.models.gcn.utils import load_encoder_data, load_encoder_txt_data
import torch

from slgnn.config import PAD_ATOM


class TestPubchemFingerprint(TestCase):

    def setUp(self):
        self.sm = "CCC(CC)COC(=O)[C@H](C)N[P@](=O)(OC[C@@H]1[C@H]"\
                  "([C@H]([C@](O1)(C#N)C2=CC=C3N2N=CN=C3N)O)O)OC4=CC=CC=C4"

    def test_fp_len(self):
        filtered_fp = get_filtered_fingerprint(self.sm)
        self.assertEqual(len(filtered_fp), 740)


class TestEncoderLoader(TestCase):

    def test_txt_loader(self):
        dpath = "test_data/sampled_smiles_100.txt"
        train, valid, len_fp = load_encoder_txt_data(dpath)
        self.assertEqual(len_fp, 740)
        self.assertIsInstance(train, dict)
        self.assertIsInstance(valid, dict)
        self.assertAlmostEqual(train["features"].shape[0],
                               valid["features"].shape[0]*9,
                               delta=9)
        self.assertEqual(train["features"].shape, (90, PAD_ATOM, 56))
        self.assertIsInstance(train["features"], torch.FloatTensor)
        self.assertIsInstance(train["adj"][0], torch.sparse.FloatTensor)
        self.assertIsInstance(train["labels"], torch.FloatTensor)
        train2, valid2, len_fp2 = load_encoder_data(dpath, type_="txt")
        self.assertEqual(train["features"].shape, train2["features"].shape)
        self.assertEqual(len(train["adj"]), len(train2["adj"]))
        self.assertEqual(valid["features"].shape, valid2["features"].shape)
        self.assertEqual(len(valid["adj"]), len(valid2["adj"]))
        self.assertEqual(len_fp, len_fp2)
