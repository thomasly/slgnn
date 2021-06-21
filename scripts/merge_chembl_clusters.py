import pickle as pk
from slgnn.data_processing.clustering import merge_common


def main(inputf, outputf):
    with open(inputf, "rb") as f:
        clusters = pk.load(f)
    merged = list(merge_common(clusters))
    with open(outputf, "wb") as f:
        pk.dump(merged, f)


if __name__ == "__main__":
    input_file = "dataset/chembl_clusters_0.5.pk"
    output_file = "dataset/chembl_clusters_0.5_merged.pk"
    main(input_file, output_file)
