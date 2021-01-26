import os

from chemreader.readers import Mol2

if __name__ == "__main__":
    f = open("data/ZINC/smiles.txt", "w")
    paths = os.scandir("/data/dgx/backup/yangliu/ZINC/ghose_filtered")
    for path in paths:
        if not path.name.endswith(".mol2.gz"):
            continue
        mol2 = Mol2(path.path)
        f.write("\n".join(mol2.to_smiles(isomeric=True, verbose=True)))
        f.write("\n")
    f.close()
