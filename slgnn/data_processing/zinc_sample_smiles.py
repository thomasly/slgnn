import os
import random

from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from rdkit import DataStructs
from tqdm import tqdm

random.seed(182436)


class SmilesSampler:
    """ Sample SMILES from source file. The source file must have one SMILES
    each line. The SMILES string must be at the beggining of the line. If there
    are comments for the SMILES, they should be seperated with the SMIELS with a
    comma.
    """

    def __init__(self, path):
        """ Sampler initializer.

        Args:
            path (str): path to the file contains SMILES.
        """
        if not os.path.isfile(path):
            raise FileNotFoundError("{} is not a file.".format(path))
        self.path = path
        self._samples = list()

    def _sample_wo_filter(self, n_samples, data, verbose):
        count = 0
        pb = tqdm(total=n_samples, ascii=True, desc="Sampling") if verbose else None
        while count < n_samples:
            sample = random.sample(data, 1)[0].split(",")[0]
            if Chem.MolFromSmiles(sample) is None:
                continue
            self._samples.append(sample)
            count += 1
            if verbose:
                pb.update(1)

    def _are_similar(self, fp, fp_list, threshold):
        for fp2 in fp_list:
            score = DataStructs.FingerprintSimilarity(fp, fp2)
            if score > threshold:
                return True
        return False

    def _sample_w_filter(self, n_samples, data, threshold, verbose):
        count = 0
        selected_mols_fp = list()
        if verbose:
            pb = tqdm(total=n_samples, ascii=True, desc="Sampling")
        while count < n_samples:
            idx = random.sample(range(len(data)), 1)[0]
            smiles = data[idx].split(",")[0]
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                data[idx] = data.pop()
                continue
            fp = GetMorganFingerprintAsBitVect(mol, 4, nBits=1024)
            if self._are_similar(fp, selected_mols_fp, threshold):
                data[idx] = data.pop()
                continue
            self._samples.append(smiles)
            selected_mols_fp.append(fp)
            if verbose:
                pb.update(1)
            count += 1

    def sample(
        self, n_samples, filter_similar=True, threshold=0.85, header=True, verbose=True
    ):
        """ Sample SMILES from file.

        Args:
            n_samples (int): number of samples.
            filter_similar (bool): whether filter out similar samples with
                Tanimoto score. Default is True.
            threshold (float): the similarity score threshold used to filter
                out similar samples. The higher the score is, the less likely
                a sample will be filtered out. Default is 0.85.
            header (bool): if the input file has hearder. If true, the first
                line of the input file will be ignored.
            verbose (bool): display progress bar. Default is True.

        Return (list):
            A python list of sampled SMILES.
        """
        with open(self.path, "r") as inf:
            if header:
                header, *data = inf.readlines()
            else:
                data = inf.readlines()
        if not filter_similar:
            self._sample_wo_filter(n_samples, data, verbose)
            return self._samples
        else:
            self._sample_w_filter(n_samples, data, threshold, verbose)
            return self._samples
