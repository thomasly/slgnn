import os

import pandas as pd
from chemreader.writers import GraphWriter
from chemreader.readers import Smiles
from rdkit.Chem import MolFromSmiles


def _is_active(value):
    if value < 1000:
        return 1
    elif value >= 10000:
        return -1
    else:
        return 0


def filter(path):
    """ Filter JAK dataset
    """
    jak = pd.read_csv(path)
    jak.dropna(subset=["Standard Relation", "Standard Value"], inplace=True)
    not_eq = jak["Standard Relation"] != "'='"
    lt_10um = jak["Standard Value"] < 100000
    filtered = jak.drop(jak.loc[not_eq & lt_10um].index)
    filtered["Activity"] = filtered["Standard Value"].apply(_is_active)
    out_path = os.path.join(
        os.path.dirname(path), "filtered_"+os.path.basename(path))
    filtered[["Smiles", "Activity"]].to_csv(out_path)


def write_graphs(inpath, outpath, prefix=None):
    """ Convert JAK dataset to graphs
    """
    smiles = list()
    labels = list()
    with open(inpath, "r") as inf:
        line = inf.readline()
        while line:
            _, sm, lb = line.strip().split(",")
            if MolFromSmiles(sm) is None:
                line = inf.readline()
                continue
            smiles.append(Smiles(sm))
            labels.append(lb)
            line = inf.readline()
    writer = GraphWriter(smiles)
    writer.write(outpath, prefix=prefix, graph_labels=labels)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", help="Path to the JAK file")
    args = parser.parse_args()
    filter(args.path)
    inpath = os.path.join(
        os.path.dirname(args.path), "filtered_"+os.path.basename(args.path))
    pre = os.path.basename(args.path).split(".")[0]
    write_graphs(inpath,
                 os.path.join(os.path.dirname(args.path), "graphs"),
                 prefix=pre)
