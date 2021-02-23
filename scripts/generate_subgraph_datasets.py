from random import shuffle, seed

import pandas as pd
from rdkit import Chem
from slgnn.data_processing.subgraph_datasets import (
    find_subgraph_and_label,
    find_edge_list,
    write_to_subgraphs,
)
from tqdm import tqdm


def prepare_dataset(ser):
    """ Get some dataset parameters for generating subgraphs.

    Args:
        ser (pandas Series): the Pandas Series containing the SMILES.

    Returns:
        n_samples (int): length of the input.
        train_indices (list): indices of the training samples. Splitting ratio: 8-1-1.
        val_indices (list): indices of the validation samples.
        test_indices (list): indices of the testing samples.
    """
    n_samples = len(ser)
    indices = list(range(n_samples))
    shuffle(indices)
    train_indices = indices[: int(n_samples * 0.8)]
    val_indices = indices[int(n_samples * 0.8) : int(n_samples * 0.9)]
    test_indices = indices[int(n_samples * 0.9) :]
    return n_samples, train_indices, val_indices, test_indices


def to_subgraphs(patterns, ser, save_path):
    """ Save subgraphs.
    
    Args:
        patters (list): list of rdkit Mol objects generated from SMARTS.
        ser (pandas Series object): pandas Serties with SMILES.
        save_path (str): path to the saving root.
    """
    n_samples, train_indices, val_indices, _ = prepare_dataset(ser)
    starting_idx = 0
    for i, smi in tqdm(enumerate(ser), total=n_samples):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        mol = Chem.AddHs(mol)
        edgelist = find_edge_list(mol)
        subgraphs = find_subgraph_and_label(mol, patterns)
        if i in train_indices:
            split = "train"
        elif i in val_indices:
            split = "val"
        else:
            split = "test"
        if i == 0:
            method = "w"
        else:
            method = "a"
        write_to_subgraphs(
            edgelist, subgraphs, save_path, starting_idx, split=split, method=method,
        )
        starting_idx += mol.GetNumAtoms()


def main():
    seed(1987)
    # initiate patterns
    patterns_df = pd.read_csv("chemreader/resources/pubchemFPKeys_to_SMARTSpattern.csv")
    patterns = [Chem.MolFromSmarts(sm) for sm in patterns_df.SMARTS]
    # save BACE dataset
    bace_df = pd.read_csv("data/DeepChem/BACE/raw/bace.csv")
    to_subgraphs(
        patterns, bace_df.mol, "data/DeepChem/BACE/subgraphs/",
    )
    del bace_df
    # save ChemBL dataset
    chembl_df = pd.read_csv(
        "/raid/home/public/dataset_ContextPred_0219/ChemBL/smiles.csv",
        header=None,
        names=["smiles"],
    )
    to_subgraphs(
        patterns,
        chembl_df.smiles,
        "/raid/home/public/dataset_ContextPred_0219/ChemBL/subgraphs/",
    )
    del chembl_df


if __name__ == "__main__":
    main()
