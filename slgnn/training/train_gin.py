""" A easy script to run GIN model.
"""
import os
import os.path as osp
from datetime import datetime
from random import shuffle
import random
import pickle as pk

from torch_geometric.data import DataLoader
from torch.utils.data import ConcatDataset
from torch import nn
import torch.nn.functional as F
import torch
import yaml

from slgnn.configs.base import Grid, Config
from slgnn.configs.arg_parsers import ModelTrainingArgs
from slgnn.models.gcn.model import GIN
from slgnn.models.decoder.model import GINDecoder
from slgnn.training.utils import (
    plot_train_val_losses, plot_reconstruct, plot_train_val_acc)


def load_data(dataset, batch_size, shuffle_=True):
    if isinstance(dataset, list):
        return load_data_from_list(dataset, batch_size, shuffle_)
    else:
        if shuffle_:
            indices = list(range(len(dataset)))
            shuffle(indices)
            dataset = dataset[indices]
        sep = int(len(dataset) * 0.9)
        train_loader = DataLoader(
            dataset[:sep], batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(
            dataset[sep:], batch_size=batch_size, shuffle=False)
        return train_loader, val_loader


def load_data_from_list(datasets: list, batch_size, shuffle_=True):
    if shuffle_:
        for i, dataset in enumerate(datasets):
            indices = list(range(len(dataset)))
            shuffle(indices)
            datasets[i] = dataset[indices]
    train_l, val_l = list(), list()
    for dataset in datasets:
        sep = int(len(dataset) * 0.9)
        train_l.append(dataset[:sep])
        val_l.append(dataset[sep:])
    train_loader = DataLoader(
        ConcatDataset(train_l), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(
        ConcatDataset(val_l), batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def train_one_epoch(model, train_loader, val_loader, epoch, criterion,
                    optimizers, device, train_losses, val_losses,
                    return_accuracy=False):
    if return_accuracy:
        train_acc = list()
        val_acc = list()
    for i, batch in enumerate(train_loader):
        model.train()
        batch = batch.to(device)
        out = model(batch)
        # if len(label.size()) == 1:
        #     label = label[:, None]
        try:
            train_loss = criterion(torch.sigmoid(out), batch.y.float())
        except RuntimeError:
            train_loss = criterion(F.log_softmax(out, dim=1), batch.y.long())
        train_losses.append(train_loss.item())
        for opt in optimizers:
            opt.zero_grad()
        train_loss.backward()
        for opt in optimizers:
            opt.step()
        if return_accuracy:
            _, pred = F.log_softmax(out, dim=1).max(dim=1)
            correct = float(pred.eq(batch.y).sum().item())
            acc = correct / batch.num_graphs
            train_acc.append(acc)
            print(
                "\repoch: {}, batch: {}, "
                "train_loss: {:.4f}, train_acc: {:.4f} ".format(
                    epoch+1, i+1, train_loss.item(), acc),
                end=""
            )
        else:
            print(
                "\repoch: {}, batch: {}, train_loss: {:.4f} ".format(
                    epoch+1, i+1, train_loss.item()),
                end=""
            )
    with torch.no_grad():
        model.eval()
        val_batch_losses = list()
        if return_accuracy:
            val_batch_acc = list()
        for val_batch in val_loader:
            val_batch = val_batch.to(device)
            val_out = model(val_batch)
            try:
                validate_loss = criterion(
                    torch.sigmoid(val_out), val_batch.y.float())
            except RuntimeError:
                validate_loss = criterion(
                    F.log_softmax(val_out, dim=1), val_batch.y.long())
            val_batch_losses.append(validate_loss.item())
            if return_accuracy:
                _, pred = F.log_softmax(val_out, dim=1).max(dim=1)
                correct = float(pred.eq(val_batch.y).sum().item())
                acc = correct / val_batch.num_graphs
                val_batch_acc.append(acc)
        val_loss = sum(val_batch_losses) / len(val_batch_losses)
        val_losses.append(val_loss)
        if return_accuracy:
            acc = sum(val_batch_acc) / len(val_batch_acc)
            val_acc.append(acc)
    if return_accuracy:
        print("val_loss: {:.4f}, val_acc: {:.4f}".format(val_loss, acc))
    else:
        print("val_loss: {:.4f}".format(val_loss))
    if return_accuracy:
        return train_loss.item(), val_loss, train_acc, val_acc
    else:
        return train_loss.item(), val_loss


def loss_before_training(model, train_loader, val_loader, criterion, config,
                         return_acc=False):
    device = torch.device(config["device"])
    tr_batch = next(iter(train_loader)).to(device)
    val_batch = next(iter(val_loader)).to(device)
    model.train()
    train_out = model(tr_batch)
    model.eval()
    val_out = model(val_batch)
    try:
        train_loss = criterion(torch.sigmoid(train_out), tr_batch.y.float())
        val_loss = criterion(torch.sigmoid(val_out), val_batch.y.float())
    except RuntimeError:
        train_loss = criterion(F.log_softmax(train_out, dim=1),
                               tr_batch.y.long())
        val_loss = criterion(F.log_softmax(val_out, dim=1),
                             val_batch.y.long())
    if not return_acc:
        return train_loss, val_loss
    else:
        _, pred = F.log_softmax(train_out, dim=1).max(dim=1)
        train_correct = float(pred.eq(tr_batch.y).sum().item())
        train_acc = train_correct / tr_batch.num_graphs
        _, pred = F.log_softmax(val_out, dim=1).max(dim=1)
        val_correct = float(pred.eq(val_batch.y).sum().item())
        val_acc = val_correct / val_batch.num_graphs
        return train_loss, val_loss, train_acc, val_acc


def train_encoder(model, config, log_dir, train_loader, val_loader):
    if config["encoder_epochs"] == 0:
        return
    device = torch.device(config["device"])

    model = model.to(device)
    criterion = config["encoder_loss"]()
    optimizer = config["optimizer"](
        model.parameters(), lr=config["learning_rate"])
    lr_scheduler = config["scheduler"](optimizer)
    early_stopper = config["early_stopper"]()
    epochs = config["encoder_epochs"]

    training_losses = list()
    validating_losses = list()
    tr_bf_train, val_bf_train = loss_before_training(
        model, train_loader, val_loader, criterion, config)
    training_losses.append(tr_bf_train)
    validating_losses.append(val_bf_train)
    best_loss_logged = False
    for e in range(epochs):
        train_loss, val_loss = train_one_epoch(
            model, train_loader, val_loader, e, criterion,
            [optimizer], device, training_losses, validating_losses
        )
        lr_scheduler.step()
        if early_stopper.stop(e, val_loss, train_loss=train_loss):
            print("Early stopped at epoch {}".format(e+1))
            metrics = early_stopper.get_best_vl_metrics()
            print("Best train loss: {:.4f}, best validate loss: {:.4f}".format(
                metrics[0], metrics[2]))
            with open(osp.join(log_dir, "best_losses.txt"), "a") as f:
                f.write("Best train loss: {}, best validate loss: {}".format(
                    metrics[0], metrics[2]))
            best_loss_logged = True
            break
    if not best_loss_logged:
        with open(osp.join(log_dir, "best_losses.txt"), "a") as f:
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
        osp.join(log_dir, "encoder_losses.png")
    )
    with open(osp.join(log_dir, "encoder_losses.pk"), "wb") as f:
        pk.dump(
            {"training_losses": training_losses,
             "validating_losses": validating_losses},
            f
        )


def train_classifier(encoder, classifier, config, log_dir, train_loader,
                     val_loader):
    device = torch.device(config["device"])

    encoder = encoder.to(device)
    classifier = classifier.to(device)
    criterion = config["classifier_loss"]()
    encoder_optimizer = config["optimizer"](
        encoder.parameters(), lr=config["learning_rate"])
    classifier_optimizer = config["optimizer"](
        classifier.parameters(), lr=config["learning_rate"])
    encoder_lr_scheduler = config["scheduler"](encoder_optimizer)
    classifier_lr_scheduler = config["scheduler"](classifier_optimizer)
    early_stopper = config["early_stopper"]()
    epochs = config["classifier_epochs"]

    training_losses = list()
    validating_losses = list()
    training_accs = list()
    validating_accs = list()
    model = nn.Sequential(encoder, classifier)
    tr_bf_train, val_bf_train, tr_acc_bf, val_acc_bf = loss_before_training(
        model, train_loader, val_loader, criterion, config, return_acc=True)
    training_losses.append(tr_bf_train)
    validating_losses.append(val_bf_train)
    training_accs.append(tr_acc_bf)
    validating_accs.append(val_acc_bf)
    best_loss_logged = False
    for e in range(epochs):
        if e < config["frozen_epochs"]:
            train_loss, val_loss, train_acc, val_acc = train_one_epoch(
                model, train_loader, val_loader, e, criterion,
                [classifier_optimizer], device, training_losses,
                validating_losses, return_accuracy=True)
            classifier_lr_scheduler.step()
            training_accs.extend(train_acc)
            validating_accs.extend(val_acc)
        else:
            optimizers = [encoder_optimizer, classifier_optimizer]
            train_loss, val_loss, train_acc, val_acc = train_one_epoch(
                model, train_loader, val_loader, e, criterion,
                optimizers, device, training_losses, validating_losses,
                return_accuracy=True)
            encoder_lr_scheduler.step()
            classifier_lr_scheduler.step()
            training_accs.extend(train_acc)
            validating_accs.extend(val_acc)
            val_acc_avg = sum(val_acc) / len(val_acc)
            train_acc_avg = sum(train_acc) / len(train_acc)
            if early_stopper.stop(e, val_loss, val_acc=val_acc_avg,
                                  train_loss=train_loss,
                                  train_acc=train_acc_avg):
                print("Early stopped at epoch {}".format(e+1))
                metrics = early_stopper.get_best_vl_metrics()
                print("Best train loss: {:.4f}, "
                      "best train acc: {:.4f}, "
                      "best validate loss: {:.4f}, "
                      "best validate acc: {:.4f}".format(
                          metrics[0], metrics[1], metrics[2], metrics[3])
                      )
                with open(osp.join(log_dir, "best_losses.txt"), "w") as f:
                    f.write("Best train loss: {:.4f}, "
                            "best train acc: {:.4f}, "
                            "best validate loss: {:.4f}, "
                            "best validate acc: {:.4f}".format(
                                metrics[0], metrics[1], metrics[2], metrics[3])
                            )
                best_loss_logged = True
                break

    if not best_loss_logged:
        with open(osp.join(log_dir, "best_losses.txt"), "a") as f:
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
        osp.join(log_dir, "classifier_losses.png")
    )

    plot_train_val_acc(
        training_accs,
        validating_accs,
        osp.join(log_dir, "classifier_accuracies.png")
    )
    with open(osp.join(log_dir, "classifier_metrics.pk"), "wb") as f:
        m_dic = {
            "training_losses": training_losses,
            "validating_losses": validating_losses,
            "training_accs": training_accs,
            "validating_accs": validating_accs
        }
        pk.dump(m_dic, f)


if __name__ == "__main__":
    torch.manual_seed(0)
    random.seed(0)
    args = ModelTrainingArgs().parse_args()
    config_grid = Grid(args.config)
    time_stamp = datetime.now().strftime(r"%Y%m%d_%H%M%S")
    for config_idx, config_dict in enumerate(config_grid):
        config = Config.from_dict(config_dict)
        datasets = config["encoder_dataset"]
        log_dir = osp.join("logs", "GIN", time_stamp,
                           config["encoder_dataset_name"], str(config_idx))
        os.makedirs(log_dir)
        with open(osp.join(log_dir, "configs.yml"), "w") as f:
            f.write(yaml.dump(config_dict))

        dim_encoder_target = config["embedding_dim"]
        dim_decoder_target = datasets[0].data.y.size(1)
        dim_features = datasets[0].data.x.size(1)
        hidden_units = config["hidden_units"]
        dropout = config["dropout"]
        train_eps = config["train_eps"]
        aggregation = config["aggregation"]
        encoder = GIN(
            dim_features=dim_features,
            dim_target=dim_encoder_target,
            config=config
        )
        decoder = GINDecoder(dim_encoder_target, dim_decoder_target, dropout)
        model = nn.Sequential(encoder, decoder)
        train_loader, val_loader = load_data(datasets, config["batch_size"])
        train_encoder(
            model, config, log_dir, train_loader, val_loader)
        data = next(iter(val_loader)).to(config["device"])
        if config["encoder_epochs"] != 0:
            for index in range(5):
                plot_reconstruct(
                    model, data,
                    index=index,
                    output=osp.join(log_dir, "gin_rec_{}.png".format(index))
                )

        cls_dataset = config["classifier_dataset"]()
        classifier = GINDecoder(
            dim_encoder_target, cls_dataset.num_classes, dropout)
        train_loader, val_loader = load_data(cls_dataset, config["batch_size"])
        train_classifier(
            encoder, classifier, config, log_dir, train_loader, val_loader)
