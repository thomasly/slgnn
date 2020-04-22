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
    are comments for the SMILES, they should be seperated with the SMIELS by a
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

    def sample(self,
               n_samples,
               filter_similar=True,
               threshold=0.85,
               header=True,
               verbose=True):
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
            count = 0
            samples = list()
            while count < n_samples:
                sample = random.sample(data, 1)[0].split(",")[0]
                if Chem.MolFromSmiles(sample) is None:
                    continue
                samples.append(sample)
                count += 1
            return samples
        count = 0
        samples = list()
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
            fp = GetMorganFingerprintAsBitVect(mol, 4, nBits=2048)
            flag = 0
            for fp2 in selected_mols_fp:
                score = DataStructs.FingerprintSimilarity(fp, fp2)
                if score > threshold:
                    data[idx] = data.pop()
                    flag = 1
                    break
            if flag == 0:
                samples.append(smiles)
                selected_mols_fp.append(fp)
                if verbose:
                    pb.update(1)
                count += 1

        return samples
