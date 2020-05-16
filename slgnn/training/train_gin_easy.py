""" A easy script to run GIN model.
"""
import os
import os.path as osp
from datetime import datetime

from torch_geometric.data import DataLoader
import torch
import yaml

from slgnn.configs.base import Grid, Config
from slgnn.configs.arg_parsers import ModelTrainingArgs
from slgnn.models.gcn.model import GIN_EASY
from slgnn.training.utils import plot_train_val_losses, plot_reconstruct


def load_data(dataset, batch_size):
    sep = int(len(dataset) * 0.9)
    train_loader = DataLoader(
        dataset[:sep], batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(
        dataset[sep:], batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def train_one_epoch(model, train_loader, val_loader, epoch, criterion,
                    optimizer, device, train_losses, val_losses):
    for i, batch in enumerate(train_loader):
        model.train()
        batch = batch.to(device)
        out = model(batch)
        train_loss = criterion(torch.sigmoid(out), batch.y.float())
        train_losses.append(train_loss.item())
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        print(
            "\repoch: {}, batch: {}, train_loss: {:.4f} ".format(
                epoch+1, i+1, train_loss.item()),
            end=""
        )
    with torch.no_grad():
        model.eval()
        counter = 0
        val_batch_losses = list()
        for val_batch in val_loader:
            val_batch = val_batch.to(device)
            val_out = model(val_batch)
            validate_loss = criterion(
                torch.sigmoid(val_out), val_batch.y.float())
            val_batch_losses.append(validate_loss.item())
            counter += 1
        val_loss = sum(val_batch_losses) / counter
        val_losses.append(val_loss)
    print("val_loss: {:.4f}".format(val_loss))
    return train_loss.item(), val_loss


def loss_before_training(model, train_loader, val_loader, config):
    device = torch.device(config["device"])
    tr_batch = next(iter(train_loader)).to(device)
    val_batch = next(iter(val_loader)).to(device)
    model.eval()
    criterion = config["loss"]()
    train_loss = criterion(torch.sigmoid(model(tr_batch)), tr_batch.y.float())
    val_loss = criterion(torch.sigmoid(model(val_batch)), val_batch.y.float())
    return train_loss, val_loss


def train_model(model, config, log_dir, train_loader, val_loader):
    device = torch.device(config["device"])

    model = model.to(device)
    criterion = config["loss"]()
    optimizer = config["optimizer"](
        model.parameters(), lr=config["learning_rate"])
    lr_scheduler = config["scheduler"](optimizer)
    early_stopper = config["early_stopper"]()
    epochs = config["classifier_epochs"]

    training_losses = list()
    validating_losses = list()
    tr_bf_train, val_bf_train = loss_before_training(
        model, train_loader, val_loader, config)
    training_losses.append(tr_bf_train)
    validating_losses.append(val_bf_train)
    best_loss_logged = False
    for e in range(epochs):
        train_loss, val_loss = train_one_epoch(
            model, train_loader, val_loader, e, criterion,
            optimizer, device, training_losses, validating_losses
        )
        lr_scheduler.step()
        if early_stopper.stop(e, val_loss, train_loss=train_loss):
            print("Early stopped at epoch {}".format(e+1))
            metrics = early_stopper.get_best_vl_metrics()
            print("Best train loss: {:.4f}, best validate loss: {:.4f}".format(
                metrics[0], metrics[2]))
            with open(osp.join(log_dir, "best_losses.txt"), "w") as f:
                f.write("Best train loss: {}, best validate loss: {}".format(
                    metrics[0], metrics[2]))
            best_loss_logged = True
            break
    if not best_loss_logged:
        with open(osp.join(log_dir, "best_losses.txt"), "w") as f:
            best_val_loss = min(validating_losses)
            best_train_loss = training_losses[
                validating_losses.index(best_val_loss)]
            print("Best train loss: {}, best validate loss: {}".format(
                best_train_loss, best_val_loss))
            f.write("Best train loss: {}, best validate loss: {}".format(
                best_train_loss, best_val_loss))
    plot_train_val_losses(
        training_losses,
        validating_losses,
        osp.join(log_dir, "train_val_losses.png")
    )


if __name__ == "__main__":
    args = ModelTrainingArgs().parse_args()
    config_grid = Grid(args.config)
    for config_dict in config_grid:
        time_stamp = datetime.now().strftime(r"%Y%m%d_%H%M%S")
        config = Config.from_dict(config_dict)
        dataset = config["dataset"]()
        log_dir = osp.join("logs", "GIN", str(dataset), time_stamp)
        os.makedirs(log_dir)
        with open(osp.join(log_dir, "configs.yml"), "w") as f:
            f.write(yaml.dump(config_dict))
        dim_target = dataset.data.y.size(1)
        dim_features = dataset.data.x.size(1)
        hidden_units = config["hidden_units"]
        dropout = config["dropout"]
        train_eps = config["train_eps"]
        aggregation = config["aggregation"]
        model = GIN_EASY(
            dim_features=dim_features,
            dim_target=dim_target,
            dropout=dropout,
            train_eps=train_eps,
            hidden_units=hidden_units,
            aggregation=aggregation
        )
        train_loader, val_loader = load_data(dataset, config["batch_size"])
        train_model(model, config, log_dir, train_loader, val_loader)
        data = next(iter(val_loader)).to(config["device"])
        for index in range(5):
            plot_reconstruct(
                model, data,
                index=index,
                output=osp.join(log_dir, "gin_rec_{}.png".format(index))
            )
