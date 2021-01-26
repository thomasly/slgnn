import os
import pickle as pk

import pandas as pd

from slgnn.data_processing.clustering import Cluster


path = os.path.join("data", "JAK")
filenames = ["filtered_JAK1.csv", "filtered_JAK2.csv", "filtered_JAK3.csv"]
cluster = Cluster()

for fn in filenames:
    df = pd.read_csv(os.path.join(path, fn))
    smiles = list(df["Smiles"])
    labels = list(df["Activity"])
    s2l = dict()
    for s, l in zip(smiles, labels):
        s2l[s] = l
    pos_smiles, mod_smiles, neg_smiles = [], [], []
    for s in smiles:
        if s2l[s] == 1:
            pos_smiles.append(s)
        elif s2l[s] == 0:
            mod_smiles.append(s)
        else:
            neg_smiles.append(s)
    pos_clusters = cluster.clustering(pos_smiles, verbose=1)
    mod_clusters = cluster.clustering(mod_smiles, verbose=1)
    neg_clusters = cluster.clustering(neg_smiles, verbose=1)
    clusters_dict = {"pos": pos_clusters, "mod": mod_clusters, "neg": neg_clusters}
    with open(os.path.join(path, fn.split(".")[0] + "_clusters.pk"), "wb") as f:
        pk.dump(clusters_dict, f)
