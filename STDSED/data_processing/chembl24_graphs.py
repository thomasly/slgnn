import os
import multiprocessing as mp

import torch
from torch_geometric.data import Data
from torch_geometric.data import Dataset
from chemreader.readers import Smiles
from tqdm import tqdm


class ChEMBL24(Dataset):
    def __init__(self, n_workers=4, **kwargs):
        self.n_workers = n_workers
        super().__init__(**kwargs)

    @property
    def raw_dir(self):
        return self.root

    @property
    def processed_dir(self):
        return os.path.join(self.root, "graphs", "chembl24")

    @property
    def raw_file_names(self):
        return "ChEMBL24_smiles_fp.csv"

    @property
    def processed_file_names(self):
        return ["chembl_10000.pt"]

    def download(self):
        """Get raw data and save to raw directory."""
        pass

    def save_data(self, q):
        """Save graphs in q to data.pt files."""
        while 1:
            data = q.get()
            if data == "END":
                break
            graph, label, idx = data
            x = torch.tensor(graph["atom_features"], dtype=torch.float)
            edge_idx = graph["adjacency"].tocoo()
            edge_idx = torch.tensor([edge_idx.row, edge_idx.col], dtype=torch.long)
            dt = Data(x=x, edge_index=edge_idx, y=label)
            torch.save(dt, os.path.join(self.processed_dir, "chembl_{}.pt".format(idx)))
            print("graph #{} saved to chembl_{}.pt".format(idx, idx), end="\r")

    def create_graph(self, data, idx, q):
        try:
            graph = Smiles(data[0]).to_graph(sparse=True)
        except AttributeError:
            return
        fp = data[1]
        label = torch.tensor(fp, dtype=torch.long)[None, :]
        q.put((graph, label, idx))

    def process(self):
        """The method converting SMILES and labels to graphs."""
        # init Queue
        manager = mp.Manager()
        q = manager.Queue(maxsize=self.n_workers * 2)
        # init listener
        writer = mp.Process(target=self.save_data, args=[q])
        writer.start()
        # init pool
        pool = mp.Pool(self.n_workers)
        # init SMILES generator
        data = self._get_data()
        pb = tqdm(data, total=self.len(), desc="Load tasks: ")
        # main loop
        for i, data in enumerate(pb):
            pool.apply_async(self.create_graph, args=[data, i, q])
        # finish the tasks
        pool.close()
        pool.join()
        q.put("END")
        writer.join()

    # def _get_len(self):
    #     n = 0
    #     with open(self.raw_paths[0]) as f:
    #         for line in f.readlines():
    #             if len(line) > 5:  # ignore empty line
    #                 n += 1
    #     return n - 1  # minus header

    def len(self):
        return 1739164

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, "chembl_{}.pt".format(idx)))
        return data

    def _get_data(self):
        """Method to get SMILES strings and generate fingerprint from the raw data."""
        with open(self.raw_paths[0]) as f:
            f.readline()  # remove header
            line = f.readline()
            while line:
                line = f.readline()
                items = line.split(",")
                if len(items) == 4:
                    smiles, fp = line.split(",")[2:]
                    yield smiles.strip(), list(map(int, list(fp.strip())))
                else:
                    continue


if __name__ == "__main__":
    n_cpu = os.cpu_count()
    ChEMBL24(n_workers=int(n_cpu / 2), root="data")
