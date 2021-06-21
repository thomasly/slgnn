import pickle as pk
import pandas as pd
from slgnn.data_processing.clustering import clustering, merge_common


def initial_clustering(threshold, output_name):
    df = pd.read_csv(
        "/raid/home/yangliu/slgnn/data/Repurposing/repurposing_samples_20200324.txt",
        sep="\t",
        header=9,
    )
    smiles_list = list(df.smiles)
    clusters = clustering(smiles_list, threshold=threshold, verbose=True)
    with open(output_name, "wb") as f:
        pk.dump(clusters, f)


def merge(inputf, outputf):
    with open(inputf, "rb") as f:
        clusters = pk.load(f)
    merged = list(merge_common(clusters))
    with open(outputf, "wb") as f:
        pk.dump(merged, f)


if __name__ == "__main__":
    threshold = 0.9
    output_name = "dataset/drugs_clusters_0.9.pk"
    initial_clustering(threshold, output_name)

    input_file = "dataset/drugs_clusters_0.9.pk"
    output_file = "dataset/drugs_clusters_0.9_merged.pk"
    merge(input_file, output_file)
