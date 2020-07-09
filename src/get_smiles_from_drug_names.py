import os
import time

from tqdm import tqdm
import pandas as pd

from pypubchem.get_cids_from_name import PubChemREST


if __name__ == "__main__":
    # Read repurposing dataset and get SMILES
    df = pd.read_csv(
        os.path.join("data", "Covid19", "repurposing_drugs_20200324.txt"),
        sep="\t",
        header=9,
    )
    outf = open(os.path.join("data", "Covid19", "repurposing_drugs_smiles.txt"), "w")
    outf.write("drug_name,cid,smiles\n")
    for name in tqdm(df["pert_iname"]):
        rest = PubChemREST(name)
        time.sleep(1)
        try:
            cid = rest.get_cid()
            smiles = rest.get_smiles()
        except TypeError:
            continue
        outf.write(f"{name},{cid},{smiles}\n")
    outf.close()
