import os
import pickle as pk

from slgnn.data_processing.deepchem_datasets import BACEFP, ClinToxFP


if __name__ == "__main__":
    Datasets = [BACEFP, ClinToxFP]
    for Dataset in Datasets:
        fp_list = list()
        dataset = Dataset()
        for data in dataset:
            fp_list.append(data.y.numpy().tolist())
        with open(
            os.path.join("data", str(dataset) + "_fingerprints.pk"), "wb"
        ) as outf:
            pk.dump(fp_list, outf)
