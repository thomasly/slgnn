import os
from random import sample
import pickle as pk

from slgnn.data_processing.zinc_dataset import ZINC


if __name__ == "__main__":
    fp_list = list()
    dataset = ZINC()
    sampled_idx = sample(range(len(dataset)), k=10000)
    for idx in sampled_idx:
        fp_list.append(dataset[idx].y.numpy().tolist())
    with open(os.path.join("data", "zinc_random_fingerprints.pk"), "wb") as outf:
        pk.dump(fp_list, outf)
