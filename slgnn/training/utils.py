import os

import torch
import matplotlib.pyplot as plt


def plot_reconstruct(model, data, index, output):
    model.eval()
    label = data.y[0].to("cpu").detach()
    out = torch.round(torch.sigmoid(model(data)))
    out_y = out[0].to("cpu").detach()
    fig, axes = plt.subplots(2, 1, figsize=(8., 12.))
    ax1, ax2 = axes.flatten()
    ax1.bar(list(range(out_y.shape[0])), out_y)
    ax1.set_xlabel("Reconstructed Fingerprint")
    ax2.bar(list(range(out_y.shape[0])), label)
    ax2.set_xlabel("PubChem Fingerprint")
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close()


def plot_train_val_losses(train_losses: list, val_losses: list, output):
    dif = int((len(train_losses)-1) / (len(val_losses)-1))
    fig, axe = plt.subplots(figsize=(8., 6.))
    x = list(range(len(train_losses)))
    axe.plot(x, train_losses, label="train_loss")
    axe.plot(x[::dif], val_losses, label="val_loss")
    axe.set_ylabel("BCE loss")
    axe.set_xlabel("Steps")
    axe.legend()
    os.makedirs(os.path.dirname(output), exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close()


class EarlyStopper:

    def stop(self, epoch, val_loss, val_acc=None, test_loss=None,
             test_acc=None, train_loss=None, train_acc=None):
        raise NotImplementedError("Implement this method!")

    def get_best_vl_metrics(self):
        return (self.train_loss, self.train_acc, self.val_loss, self.val_acc,
                self.test_loss, self.test_acc, self.best_epoch)


class Patience(EarlyStopper):

    '''
    Implement common "patience" technique
    '''

    def __init__(self, patience=20, use_loss=True):
        self.local_val_optimum = float("inf") if use_loss else -float("inf")
        self.use_loss = use_loss
        self.patience = patience
        self.best_epoch = -1
        self.counter = -1

        self.train_loss, self.train_acc = None, None
        self.val_loss, self.val_acc = None, None
        self.test_loss, self.test_acc = None, None

    def stop(self, epoch, val_loss, val_acc=None, test_loss=None,
             test_acc=None, train_loss=None, train_acc=None):
        if self.use_loss:
            if val_loss <= self.local_val_optimum:
                self.counter = 0
                self.local_val_optimum = val_loss
                self.best_epoch = epoch
                self.train_loss, self.train_acc = train_loss, train_acc
                self.val_loss, self.val_acc = val_loss, val_acc
                self.test_loss, self.test_acc = test_loss, test_acc
                return False
            else:
                self.counter += 1
                return self.counter >= self.patience
        else:
            if val_acc >= self.local_val_optimum:
                self.counter = 0
                self.local_val_optimum = val_acc
                self.best_epoch = epoch
                self.train_loss, self.train_acc = train_loss, train_acc
                self.val_loss, self.val_acc = val_loss, val_acc
                self.test_loss, self.test_acc = test_loss, test_acc
                return False
            else:
                self.counter += 1
                return self.counter >= self.patience
