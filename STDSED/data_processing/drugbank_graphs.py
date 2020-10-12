import os

from ._graph_base import _Base


class DrugBank(_Base):
    @property
    def processed_dir(self):
        return os.path.join(self.root, "graphs", "drugbank")

    @property
    def raw_file_names(self):
        return "DrugBank_smiles_fp.csv"

    @property
    def processed_file_names(self):
        return "drugbank_graphs.pt"

    def process(self):
        super().process("SMILES", "ECFP")


if __name__ == "__main__":
    db = DrugBank(root="data")
    print(db)
