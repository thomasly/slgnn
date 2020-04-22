import os

from chemreader.writers import GraphWriter
from chemreader.readers import Smiles
from rdkit.Chem import MolFromSmiles


def write_graphs(inpath, outpath, prefix=None):
    """ Convert JAK dataset to graphs
    """
    smiles = list()
    with open(inpath, "r") as inf:
        line = inf.readline()
        while line:
            sm = line.strip()
            if MolFromSmiles(sm) is None:
                line = inf.readline()
                continue
            smiles.append(Smiles(sm))
            line = inf.readline()
    writer = GraphWriter(smiles)
    writer.write(outpath, prefix=prefix, graph_labels=None)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", help="Path to the SMILES file")
    args = parser.parse_args()
    pre = os.path.basename(args.path).split(".")[0]
    write_graphs(args.path,
                 os.path.join(os.path.dirname(args.path), "graphs"),
                 prefix=pre)
