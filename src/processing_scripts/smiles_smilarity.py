""" Remove similar molecules in the ZINC dataset with Tanimoto score.
"""

from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from rdkit import DataStructs


if __name__ == "__main__":
    with open("../ghose_filtered_smiles.csv", "r") as f:
        header, *data = f.readlines()
    i = 0
    while i < len(data):
        m1 = Chem.MolFromSmiles(data[i].split(",")[0])
        fp1 = GetMorganFingerprintAsBitVect(m1, 4, nBits=2048)
        j = i + 1
        while j < len(data):
            m2 = Chem.MolFromSmiles(data[j].split(",")[0])
            fp2 = GetMorganFingerprintAsBitVect(m2, 4, nBits=2048)
            similarity = DataStructs.FingerprintSimilarity(fp1, fp2)
            if similarity > 0.85:
                if j == len(data) - 1:
                    data.pop()
                else:
                    data[j] = data.pop()
                continue
            j += 1
        i += 1
    with open("../ghose_filtered_smiles_smilar_removed.csv", "w") as outf:
        outf.write(header)
        outf.write("".join(data))
