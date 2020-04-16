""" Remove similar molecules in the ZINC dataset with Tanimoto score.
"""

from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from rdkit import DataStructs
from tqdm import tqdm


if __name__ == "__main__":
    with open("./test.csv", "r") as f:
        header, *data = f.readlines()
    i = 0
    pb1 = tqdm(total=len(data), ascii=True, desc="Main progress")
    while i < len(data):
        pb2 = tqdm(total=len(data), ascii=True, desc="Look for similar")
        m1 = Chem.MolFromSmiles(data[i].split(",")[0])
        fp1 = GetMorganFingerprintAsBitVect(m1, 4, nBits=2048)
        j = i + 1
        while j < len(data):
            m2 = Chem.MolFromSmiles(data[j].split(",")[0])
            if m2 is None:
                data[j] = data.pop()
                pb2.update(1)
                continue
            fp2 = GetMorganFingerprintAsBitVect(m2, 4, nBits=2048)
            similarity = DataStructs.FingerprintSimilarity(fp1, fp2)
            if similarity > 0.85:
                if j == len(data) - 1:
                    data.pop()
                else:
                    data[j] = data.pop()
            else:
                j += 1
            pb2.update(1)
        pb1.update(1)
        i += 1
    with open("../ghose_filtered_smiles_smilar_removed.csv", "w") as outf:
        outf.write(header)
        outf.write("".join(data))
