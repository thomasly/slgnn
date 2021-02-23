import os

from rdkit import Chem


def _mol_from_smiles(mol):
    if isinstance(mol, str):
        smiles = mol
        mol = Chem.MolFromSmiles(mol)
    if mol is None:
        raise RuntimeError(
            f"The SMILES ({smiles}) can't be converted to rdkit Mol object."
        )
    mol = Chem.AddHs(mol)
    return mol


def find_subgraph_and_label(mol, patterns):
    mol = _mol_from_smiles(mol)
    subgraphs = dict()
    for i, pat in enumerate(patterns):
        matches = mol.GetSubstructMatches(pat)
        if len(matches) > 0:
            subgraphs[i] = matches
    return subgraphs


def find_edge_list(mol):
    mol = _mol_from_smiles(mol)
    edge_list = list()
    for bond in mol.GetBonds():
        start = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        edge_list.append(f"{start} {end}")
    return edge_list


def write_to_subgraphs(
    edgelist, subgraphs, path=None, starting_idx=0, split="train", method="a"
):
    """ Write edgelist to path/edge_list.txt. Write subgraphs to path/subgraphs.pth.

    Args:
        edgelist (iterable): each item has a str with two node numbers seperated by
            space.
        subgraphs (dict): dict with subgraph label as keys and subgraph nodes lists as
            values.
        path (str): root path to save the files. If not set, will save the files to
            current location.
        starting_idx (int): the starting index of the nodes. All node numbers will add
            this index. Default is 0.
        split (str): train, test, or val. Default is train.
        method (str): writing method to the files. 'w': Truncate file to zero length or
            create text file for writing. The stream is positioned at the beginning of
            the file. 'a': Open for writing.  The file is created if it does not exist.
            The stream is positioned at the end of the file.
    """
    assert split in [
        "train",
        "test",
        "val",
    ], f"Value of split must in ['train', 'test', 'val'], now is {split}"
    if path is None:
        path = "."
    else:
        os.makedirs(path, exist_ok=True)
    edge_list_f = open(os.path.join(path, "edge_list.txt"), method)
    for edge in edgelist:
        start, end = map(lambda x: int(x) + starting_idx, edge.split())
        edge_list_f.write(f"{start} {end}\n")
    edge_list_f.close()

    subgraphs_f = open(os.path.join(path, "subgraphs.pth"), method)
    for key, value in subgraphs.items():
        for subgraph in value:
            subgraphs_f.write(
                "-".join(map(lambda x: str(x + starting_idx), subgraph)) + "\t"
            )
            subgraphs_f.write(str(key) + "\t")
            subgraphs_f.write(split + "\n")
    subgraphs_f.close()


def main(args):
    import pandas as pd

    edgelist = find_edge_list(args.smi)
    patterns_df = pd.read_csv("chemreader/resources/pubchemFPKeys_to_SMARTSpattern.csv")
    patterns = [Chem.MolFromSmarts(sm) for sm in patterns_df.SMARTS]
    subgraphs = find_subgraph_and_label(args.smi, patterns)
    write_to_subgraphs(edgelist, subgraphs, "test_subgraphs", 0, "val", "w")

    mol = Chem.MolFromSmiles(args.smi)
    mol = Chem.AddHs(mol)
    edgelist = find_edge_list(mol)
    subgraphs = find_subgraph_and_label(mol, patterns)
    write_to_subgraphs(edgelist, subgraphs, "test_subgraphs", 65, "train", "a")


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument(
        "--smi",
        type=str,
        default="O1CC[C@@H](NC(=O)[C@@H](Cc2cc3cc(ccc3nc2N)-c2ccccc2C)C)CC1(C)C",
    )
    args = parser.parse_args()
    main(args)
