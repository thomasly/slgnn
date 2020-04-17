import unittest

from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from rdkit.DataStructs import FingerprintSimilarity

from slgnn.data_processing.sample_smiles import SmilesSampler


class TestSamplingSmiles(unittest.TestCase):

    def setUp(self):
        self.sampler = SmilesSampler("test_data/zinc_ghose_smiles_test.csv")

    def test_num_of_sample_is_correct(self):
        samples = self.sampler.sample(100, verbose=False)
        self.assertEqual(len(samples), 100)

    def test_keep_similar_samples(self):
        samp = self.sampler.sample(60, filter_similar=False, verbose=False)
        scores = list()
        i, j = 0, 0
        while i < len(samp)-1:
            j = i + 1
            mol1 = Chem.MolFromSmiles(samp[i])
            fp1 = GetMorganFingerprintAsBitVect(mol1, 4, nBits=2048)
            while j < len(samp):
                mol2 = Chem.MolFromSmiles(samp[j])
                fp2 = GetMorganFingerprintAsBitVect(mol2, 4, nBits=2048)
                score = FingerprintSimilarity(fp1, fp2)
                scores.append(score)
                j += 1
            i += 1
        self.assertFalse(all([s < 0.85 for s in scores]))

    def test_correct_filter_similar_samples(self):
        samp = self.sampler.sample(
            60, filter_similar=True, threshold=0.3, verbose=False)
        scores = list()
        i, j = 0, 0
        while i < len(samp)-1:
            j = i + 1
            mol1 = Chem.MolFromSmiles(samp[i])
            fp1 = GetMorganFingerprintAsBitVect(mol1, 4, nBits=2048)
            while j < len(samp):
                mol2 = Chem.MolFromSmiles(samp[j])
                fp2 = GetMorganFingerprintAsBitVect(mol2, 4, nBits=2048)
                score = FingerprintSimilarity(fp1, fp2)
                scores.append(score)
                j += 1
            i += 1
        self.assertTrue(all([s < 0.3 for s in scores]))
