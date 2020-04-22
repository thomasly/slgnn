import os

import pandas as pd


def is_active(value):
    if value < 1000:
        return 1
    elif value >= 10000:
        return -1
    else:
        return 0


def filter(path):
    jak = pd.read_csv(path)
    jak.dropna(subset=["Standard Relation", "Standard Value"], inplace=True)
    not_eq = jak["Standard Relation"] != "'='"
    lt_10um = jak["Standard Value"] < 100000
    filtered = jak.drop(jak.loc[not_eq & lt_10um].index)
    filtered["Activity"] = filtered["Standard Value"].apply(is_active)
    out_path = os.path.join(
        os.path.dirname(path), "filtered_"+os.path.basename(path))
    filtered[["Smiles", "Activity"]].to_csv(out_path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", help="Path to the JAK file")
    args = parser.parse_args()
    filter(args.path)
