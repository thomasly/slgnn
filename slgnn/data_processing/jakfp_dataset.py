import os

import pandas as pd
from chemreader.writers import GraphWriter
from chemreader.readers import Smiles
from rdkit.Chem import MolFromSmiles
from slgnn.models.gcn.utils import get_filtered_fingerprint
from tqdm import tqdm


def _is_active(value):
    if value < 1000:
        return 1
    elif value >= 10000:
        return -1
    else:
        return 0


def filter_(path):
    """ Filter JAK dataset
    """
    jak = pd.read_csv(path)
    jak.dropna(subset=["Standard Relation", "Standard Value"], inplace=True)
    not_eq = jak["Standard Relation"] != "'='"
    lt_10um = jak["Standard Value"] < 100000
    filtered = jak.drop(jak.loc[not_eq & lt_10um].index)
    gt = jak["Standard Relation"] == "'>'"
    eq_1um = jak["Standard Value"] >= 1000
    add_back = jak.loc[gt & eq_1um]
    filtered = filtered.append(add_back)
    filtered["Activity"] = filtered["Standard Value"].apply(_is_active)
    out_path = os.path.join(
        os.path.dirname(path), "filtered_"+os.path.basename(path))
    filtered[["Smiles", "Activity"]].to_csv(out_path)


def write_graphs(inpath, outpath, prefix=None):
    """ Convert JAK dataset to graphs
    """
    smiles = list()
    fps = list()
    pb = tqdm()
    with open(inpath, "r") as inf:
        line = inf.readline()
        while line:
            _, sm, _ = line.strip().split(",")
            if MolFromSmiles(sm) is None:
                line = inf.readline()
                continue
            smiles.append(Smiles(sm))
            fps.append(",".join(map(str, get_filtered_fingerprint(sm))))
            pb.update(1)
            line = inf.readline()
    writer = GraphWriter(smiles)
    writer.write(outpath, prefix=prefix, graph_labels=fps)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", help="Path to the JAK file")
    args = parser.parse_args()
    filter_(args.path)
    inpath = os.path.join(
        os.path.dirname(args.path), "filtered_"+os.path.basename(args.path))
    pre = os.path.basename(args.path).split(".")[0]+"FP"
    write_graphs(inpath,
                 os.path.join(os.path.dirname(args.path), "graphs"),
                 prefix=pre)
