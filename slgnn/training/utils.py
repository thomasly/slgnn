import os

import torch
import matplotlib.pyplot as plt


def plot_reconstruct(model, data, index, output):
    model.eval()
    label = data.y[index].to("cpu").detach()
    out = torch.round(torch.sigmoid(model(data)))
    out_y = out[index].to("cpu").detach()
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 12.0))
    ax1, ax2 = axes.flatten()
    ax1.bar(list(range(out_y.shape[0])), out_y)
    ax1.set_xlabel("Reconstructed Fingerprint")
    ax2.bar(list(range(out_y.shape[0])), label)
    ax2.set_xlabel("PubChem Fingerprint")
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close()


def plot_train_val_losses(train_losses: list, val_losses: list, output):
    dif = int((len(train_losses) - 1) / (len(val_losses) - 1))
    fig, axe = plt.subplots(figsize=(8.0, 6.0))
    x = list(range(len(train_losses)))
    axe.plot(x, train_losses, label="train_loss")
    axe.plot(x[::dif], val_losses, label="val_loss")
    axe.set_ylabel("BCE loss")
    axe.set_xlabel("Steps")
    axe.legend()
    os.makedirs(os.path.dirname(output), exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close()


def plot_train_val_acc(train_accs: list, val_accs: list, output):
    dif = int((len(train_accs) - 1) / (len(val_accs) - 1))
    fig, axe = plt.subplots(figsize=(8.0, 6.0))
    x = list(range(len(train_accs)))
    axe.plot(x, train_accs, label="train_acc")
    axe.plot(x[::dif], val_accs, label="val_acc")
    axe.set_ylabel("Accuracy Score")
    axe.set_xlabel("Steps")
    axe.set_ylim(0, 1)
    axe.legend()
    os.makedirs(os.path.dirname(output), exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close()


class EarlyStopper:
    """Base class of Early stoppers.

    Args:
        epoch (int): current epoch number.
        metrics_dict (dict): dict of current metrics.
    """

    def stop(self, epoch, metrics_dict):
        """Decide if the training should stop.

        Returns:
            bool:
        """
        raise NotImplementedError("Implement this method!")

    def get_best_vl_metrics(self):
        """Get the best metrics values.

        Returns:
            dict: the metrics values when the monitored metric reaches the best
                value.
        """
        return self.opt_metrics


class Patience(EarlyStopper):

    """Implement common "patience" technique.

    Args:
        patience (int): number of patience. Default 20.
        monitor (str): name of the metric to monitor. Default "val_loss".
        mode (str): "min" or "max". Minimize or maxiumize the monitored metric. Default
            "min".
    """

    def __init__(self, patience=20, monitor="val_loss", mode="min"):
        assert mode in ["min", "max"]
        self.local_val_optimum = float("inf") if mode == "min" else -float("inf")
        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self.best_epoch = -1
        self.counter = -1
        self.opt_metrics = dict()

    def stop(self, epoch, metrics_dict):
        if self.mode == "min":
            flag = metrics_dict[self.monitor] <= self.local_val_optimum
        else:
            flag = metrics_dict[self.monitor] >= self.local_val_optimum
        if flag:
            self.counter = 0
            self.local_val_optimum = metrics_dict[self.monitor]
            self.best_epoch = epoch
            self.opt_metrics = metrics_dict.copy()
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience

    def best_so_far(self, metrics_dict):
        if self.mode == "min":
            flag = metrics_dict[self.monitor] <= self.local_val_optimum
        else:
            flag = metrics_dict[self.monitor] >= self.local_val_optimum
        if flag:
            self.local_val_optimum = metrics_dict[self.monitor]
        return flag
