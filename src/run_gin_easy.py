""" A easy script to run GIN model.
"""
import os.path as osp

from slgnn.models.gcn.model import GIN
from slgnn.data_processing.pyg_datasets import ZINCDataset
from torch_geometric.data import DataLoader
from torch.optim import Adam
from torch.nn import BCEWithLogitsLoss


root = osp.join("data", "ZINC", "graphs")
name = "sampled_smiles_1000"
dataset = ZINCDataset(root=root, name=name, use_node_attr=True)
dataloader = DataLoader(dataset, batch_size=32)

model = GIN(7, 740, )
criterion = BCEWithLogitsLoss()
optimizer = Adam(model.parameters(), lr=0.0001)

epochs = 50
for e in range(epochs):
    for i, batch in enumerate(dataloader):
        out = model(batch)
        loss = criterion(out, batch.y)
        optimizer.zero_grad()
        loss.back()
        optimizer.step()
        print("epoch: {}, batch: {}".format(e+1, i+1), end="\r")
