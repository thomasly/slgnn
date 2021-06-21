import pickle as pk
from slgnn.data_processing.clustering import clustering


def main(threshold, output_name):
    with open("contextSub/dataset/chembl/processed/smiles.csv", "r") as f:
        smiles_list = [line.strip() for line in f.readlines()]
    clusters = clustering(smiles_list, threshold=threshold, verbose=True)
    with open(output_name, "wb") as f:
        pk.dump(clusters, f)


if __name__ == "__main__":
    # threshold = 0.75
    threshold = 0.5
    output_name = "dataset/chembl_clusters_0.5.pk"
    main(threshold, output_name)
