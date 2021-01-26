import os
import json

import pandas as pd


root = os.path.join("data", "JAK")
for fn in ["filtered_JAK1.csv", "filtered_JAK2.csv", "filtered_JAK3.csv"]:
    df = pd.read_csv(os.path.join(root, fn))
    smiles = list(df["Smiles"])
    labels = list(df["Activity"])
    d = dict()
    for s, l in zip(smiles, labels):
        d[s] = l
    save_name = fn.split(".")[0] + ".json"
    with open(os.path.join(root, save_name), "w") as f:
        json.dump(d, f)
