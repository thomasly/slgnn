import os

import torch

from ._graph_base import _Base


class MolNet(_Base):
    def __init__(self, root, dataset, transform=None, pre_transform=None):
        self.dataset = dataset
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_dir(self):
        return os.path.join(self.root, "graphs", "molnet", self.dataset)

    @property
    def raw_file_names(self):
        return f"{self.dataset}_ecfp.csv"

    @property
    def processed_file_names(self):
        return f"{self.dataset}_graphs.pt"

    def process(self):
        super().process("smiles", "ECFP", "Label")


if __name__ == "__main__":
    for ds in "tox21", "toxcast", "muv":
        dataset = MolNet(root="data/MolNet_ecfp", dataset=ds)
        print(dataset)
