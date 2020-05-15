""" A easy script to run GIN model.
"""
import os
import os.path as osp
from shutil import copy2
from datetime import datetime

from torch_geometric.data import DataLoader
import torch
import matplotlib.pyplot as plt

from slgnn.configs.base import Grid, Config
from slgnn.configs.arg_parsers import ModelTrainingArgs
from slgnn.models.gcn.model import GIN_EASY
from slgnn.training.utils import plot_train_val_losses


time_stamp = datetime.now().strftime(r"%Y%m%d_%H%M%S")
args = ModelTrainingArgs().parse_args()

config = Config.from_dict(Grid(args.config)[0])

device = torch.device(config["device"])
dataset = config["dataset"]()
log_dir = osp.join("logs", "GIN", str(dataset), time_stamp)
os.makedirs(log_dir)
copy2(args.config, osp.join(log_dir, osp.basename(args.config)))

sep = int(len(dataset) * 0.9)
batch_size = config["batch_size"]
training_dataloader = DataLoader(
    dataset[:sep], batch_size=batch_size, shuffle=True)
validating_dataloader = DataLoader(
    dataset[sep:], batch_size=batch_size, shuffle=False)

dim_target = dataset.data.y.size(1)
dim_features = dataset.data.x.size(1)

model = GIN_EASY(dim_features, dim_target).to(device)
criterion = config["loss"]()
optimizer = config["optimizer"](model.parameters(), lr=config["learning_rate"])
lr_scheduler = config["scheduler"](optimizer)
early_stopper = config["early_stopper"]()
epochs = config["classifier_epochs"]

model.train()
training_losses = list()
validating_losses = list()
for e in range(epochs):
    for i, batch in enumerate(training_dataloader):
        batch = batch.to(device)
        out = model(batch)
        train_loss = criterion(torch.sigmoid(out), batch.y.float())
        training_losses.append(train_loss.item())
        with torch.no_grad():
            counter = 0
            val_batch_losses = list()
            for val_batch in validating_dataloader:
                val_batch = val_batch.to(device)
                val_out = model(val_batch)
                validate_loss = criterion(
                    torch.sigmoid(val_out), val_batch.y.float())
                val_batch_losses.append(validate_loss.item())
                counter += 1
            validating_losses.append(sum(val_batch_losses) / counter)
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        print("epoch: {}, batch: {}, train_loss: {}, val_loss: {}".format(
            e+1, i+1, train_loss.item(), validating_losses[-1]), end="\r")
        lr_scheduler.step()
    print()
    if early_stopper.stop(e, validating_losses[-1]):
        print("Early stopped at epoch {}".format(e+1))
        break


plot_train_val_losses(
    training_losses,
    validating_losses,
    osp.join(log_dir, "train_val_losses.png")
)

model.eval()
label = val_batch.y[0].to("cpu").detach()
out = torch.round(torch.sigmoid(model(val_batch)))
out_y = out[0].to("cpu").detach()
fig, axes = plt.subplots(2, 1, figsize=(8., 12.))
ax1, ax2 = axes.flatten()
ax1.bar(list(range(out_y.shape[0])), out_y)
ax1.set_xlabel("Reconstructed Fingerprint")
ax2.bar(list(range(out_y.shape[0])), label)
ax2.set_xlabel("PubChem Fingerprint")
fig.savefig(osp.join(log_dir, "gin_rec_1.png"), dpi=300, bbox_inches="tight")

label = val_batch.y[1].to("cpu").detach()
out = torch.round(torch.sigmoid(model(val_batch)))
out_y = out[1].to("cpu").detach()
fig, axes = plt.subplots(2, 1, figsize=(8., 12.))
ax1, ax2 = axes.flatten()
ax1.bar(list(range(out_y.shape[0])), out_y)
ax1.set_xlabel("Reconstructed Fingerprint")
ax2.bar(list(range(out_y.shape[0])), label)
ax2.set_xlabel("PubChem Fingerprint")
fig.savefig(osp.join(log_dir, "gin_rec_2.png"), dpi=300, bbox_inches="tight")
