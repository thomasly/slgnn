import os

from ._graph_base import _Base


class CYP450(_Base):
    @property
    def processed_dir(self):
        return os.path.join(self.root, "graphs", "cyp450")

    @property
    def raw_file_names(self):
        return "fromraw_cid_inchi_smiles_fp_labels.csv"

    @property
    def processed_file_names(self):
        return "cyp450_graphs.pt"

    def process(self):
        super().process("isomeric_SMILES", "ECFP", "Label")


if __name__ == "__main__":
    cyp = CYP450(root="data")
    print(cyp)
