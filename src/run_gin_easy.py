""" A easy script to run GIN model.
"""
import os.path as osp

from torch_geometric.data import DataLoader
from torch.optim import Adam
from torch.nn import BCELoss
import torch
import matplotlib.pyplot as plt

from slgnn.models.gcn.model import GIN_EASY
from slgnn.data_processing.pyg_datasets import ZINCDataset


if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

root = osp.join("data", "ZINC", "graphs")
name = "sampled_smiles_1000"
dataset = ZINCDataset(root=root, name=name, use_node_attr=True)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

dim_target = 740
dim_features = dataset.data.x.size(1)

model = GIN_EASY(dim_features, dim_target).to(device)
criterion = BCELoss()
optimizer = Adam(model.parameters(), lr=0.0001)

epochs = 50
model.train()
for e in range(epochs):
    for i, batch in enumerate(dataloader):
        batch = batch.to(device)
        out = model(batch)
        loss = criterion(out, batch.y.float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print("epoch: {}, batch: {}, loss: {}".format(
            e+1, i+1, loss.item()), end="\r")
print()

model.eval()
label = batch.y[0].to("cpu").detach()
out = model(batch)
out_y = out[0].to("cpu").detach()
fig, axes = plt.subplots(2, 1, figsize=(8., 12.))
ax1, ax2 = axes.flatten()
ax1.bar(list(range(out_y.shape[0])), out_y)
ax1.set_xlabel("Reconstructed Fingerprint")
ax2.bar(list(range(out_y.shape[0])), label)
ax2.set_xlabel("PubChem Fingerprint")
fig.savefig("./rec.png", dpi=300, bbox_inches="tight")

label = batch.y[1].to("cpu").detach()
out = model(batch)
out_y = out[1].to("cpu").detach()
fig, axes = plt.subplots(2, 1, figsize=(8., 12.))
ax1, ax2 = axes.flatten()
ax1.bar(list(range(out_y.shape[0])), out_y)
ax1.set_xlabel("Reconstructed Fingerprint")
ax2.bar(list(range(out_y.shape[0])), label)
ax2.set_xlabel("PubChem Fingerprint")
fig.savefig("./rec_2.png", dpi=300, bbox_inches="tight")
