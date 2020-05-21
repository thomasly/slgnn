import os
import gzip

from tqdm import tqdm

from mol2_reader import Mol2Reader


if __name__ == "__main__":
    counter = 0
    items = list(os.scandir("../ghose_filtered"))
    for it in tqdm(items):
        try:
           reader = Mol2Reader(it.path)
           counter += reader.n_mols
        except KeyboardInterrupt:
            raise
        except:
            print(it.path)
            raise
    print("=" * 80)
    print("Total number of molecules: {}".format(counter))
