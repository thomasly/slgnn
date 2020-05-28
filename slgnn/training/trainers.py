import os
from abc import ABC, abstractmethod
import pickle as pk
from statistics import mean

import torch
from torch_geometric.data import Batch
import matplotlib.pyplot as plt

from slgnn.metrics.metrics import accuracy


class BaseTrainer(ABC):
    """ The base class for trainers
    """

    def __init__(self, config, model=None, train_loader=None, val_loader=None):
        self._model = model
        self._train_loader = train_loader
        self._val_loader = val_loader
        self._config = config
        self._parse_config()
        self._tr_losses = list()
        self._val_losses = list()

    def _parse_config(self):
        self._lr = self.config["learning_rate"]
        self._early_stopper = self.config["early_stopper"]()
        self._device = torch.device(self.config["device"])

    @property
    def device(self):
        return self._device

    @property
    def train_loader(self):
        if self._train_loader is None:
            raise AttributeError(
                "train_loader is not initialized. Please initialize the "
                "train_loader when constructing the trainer instance or use "
                "set_train_loader() method after the instance is constructed."
            )
        return self._train_loader

    @train_loader.setter
    def train_loader(self, dataloader):
        self._train_loader = dataloader

    @property
    def val_loader(self):
        if self._val_loader is None:
            raise AttributeError(
                "val_loader is not initialized. Please initialize the "
                "val_loader when constructing the trainer instance or use "
                "set_val_loader() method after the instance is constructed."
            )
        return self._val_loader

    @val_loader.setter
    def val_loader(self, dataloader):
        self._val_loader = dataloader

    @property
    def config(self):
        return self._config

    @config.setter
    def config(self, conf):
        self._config = conf

    @property
    def train_losses(self):
        return self._tr_losses

    @property
    def val_losses(self):
        return self._val_losses

    @property
    def early_stopper(self):
        return self._early_stopper

    @property
    @abstractmethod
    def criterion(self):
        pass


class EncoderDecoderTrainer(BaseTrainer):
    def __init__(
        self, config, encoder=None, decoder=None, train_loader=None, val_loader=None
    ):
        self._encoder = encoder
        self._decoder = decoder
        super().__init__(
            config=config, train_loader=train_loader, val_loader=val_loader
        )

    def _parse_config(self):
        self._lr = self.config["learning_rate"]
        self._device = torch.device(self.config["device"])
        self._early_stopper = self.config["encoder_early_stopper"]()
        self._encoder_optimizer = self.config["optimizer"](
            self._encoder.parameters(), lr=self._lr
        )
        self._decoder_optimizer = self.config["optimizer"](
            self._decoder.parameters(), lr=self._lr
        )
        self._encoder_lr_scheduler = self.config["scheduler"](self._encoder_optimizer)
        self._decoder_lr_scheduler = self.config["scheduler"](self._decoder_optimizer)
        self._criterion = self.config["encoder_loss"]()

    def _setup_models(self, mode="train"):
        self.encoder = self.encoder.to(self.device)
        self.decoder = self.decoder.to(self.device)
        if mode == "train":
            self.encoder.train()
            self.decoder.train()
        else:
            self.encoder.eval()
            self.decoder.eval()

    def load_optimizers(self, *optimizers):
        self._optimizers = optimizers

    @property
    def encoder(self):
        return self._encoder

    @encoder.setter
    def encoder(self, model):
        self._encoder = model

    @property
    def decoder(self):
        return self._decoder

    @decoder.setter
    def decoder(self, model):
        self._decoder = model

    @property
    def use_accuracy(self):
        return self._use_accuracy

    @use_accuracy.setter
    def use_accuracy(self, value):
        self._use_accuracy = value

    @property
    def criterion(self):
        return self._criterion

    @property
    def encoder_lr_scheduler(self):
        return self._encoder_optimizer

    @property
    def decoder_lr_scheduler(self):
        return self._decoder_lr_scheduler

    def train(self):
        self.epoch = 0
        self.log_before_training_status()
        while self.epoch < self.config["encoder_epochs"]:
            # if self.epoch < self.config["frozen_epochs"]:
            #     self.load_optimizers(self._decoder_optimizer)
            # else:
            self.load_optimizers(self._encoder_optimizer, self._decoder_optimizer)

            self.train_one_epoch()
            self.validate()
            stop = self.early_stopper.stop(
                self.epoch, self._cur_val_loss, train_loss=self._cur_train_loss
            )
            if stop:
                break
            self.encoder_lr_scheduler.step()
            self.decoder_lr_scheduler.step()
            self.epoch += 1
        metrics = self.early_stopper.get_best_vl_metrics()
        print(
            f"Best train loss: {metrics[0]:.4f}, "
            f"best validate loss: {metrics[2]:.4f}"
        )

    def log_before_training_status(self):
        with torch.no_grad():
            self._setup_models("train")
            batch_losses = list()
            for batch in self.train_loader:
                batch = batch.to(self.device)
                out = self.decoder(self.encoder(batch))
                batch_losses.append(self.criterion(out, batch.y.float()).item())
                loss = mean(batch_losses)
            self.train_losses.append(loss)
            batch_losses = list()
            for batch in self.val_loader:
                batch = batch.to(self.device)
                out = self.decoder(self.encoder(batch))
                batch_losses.append(self.criterion(out, batch.y.float()).item())
                loss = mean(batch_losses)
            self.val_losses.append(loss)

    def train_one_epoch(self):
        self._setup_models("train")
        batch_losses = list()
        for i, batch in enumerate(self.train_loader):
            batch = batch.to(self.device)
            out = self.decoder(self.encoder(batch))
            train_loss = self.criterion(out, batch.y.float())
            batch_losses.append(train_loss.item())
            self.train_losses.append(train_loss)
            for opt in self._optimizers:
                opt.zero_grad()
            train_loss.backward()
            for opt in self._optimizers:
                opt.step()
            print(
                f"\repoch: {self.epoch+1}, batch: {i+1}, "
                f"train_loss: {train_loss.item():.4f}",
                end=" ",
            )
        self._cur_train_loss = mean(batch_losses)

    def validate(self):
        self._setup_models("eval")
        with torch.no_grad():
            batch_losses = list()
            for batch in self.val_loader:
                batch = batch.to(self.device)
                out = self.decoder(self.encoder(batch))
                batch_losses.append(self.criterion(out, batch.y.float()).item())
        self._cur_val_loss = mean(batch_losses)
        self.val_losses.append(self._cur_val_loss)
        print(f"val_loss: {self._cur_val_loss:.4f}")

    def _rooting(self, path):
        if path is None:
            root = os.path.curdir
        else:
            root = path
        os.makedirs(root, exist_ok=True)
        return root

    def log_results(self, out=None, txt_name=None, pk_name=None):
        root = self._rooting(out)
        if txt_name is None:
            txt_file = os.path.join(root, "encoder_training_metrics.txt")
        else:
            txt_file = os.path.join(root, txt_name)
        if pk_name is None:
            pk_file = os.path.join(root, "encoder_losses.pk")
        else:
            pk_file = os.path.join(root, pk_name)
        with open(txt_file, "w") as f:
            metrics = self.early_stopper.get_best_vl_metrics()
            f.write(
                f"Best train loss: {metrics[0]:.4f}, "
                f"best validate loss: {metrics[2]:.4f}"
            )
        with open(pk_file, "wb") as f:
            loss_dict = (
                {
                    "training_losses": self.train_losses,
                    "validating_losses": self.val_losses,
                },
            )
            pk.dump(loss_dict, f)

    def plot_training_metrics(self, path=None, name=None):
        root = self._rooting(path)
        if name is None:
            filep = os.path.join(root, "encoder_train_val_losses.png")
        else:
            filep = os.path.join(root, name)
        dif = int((len(self.train_losses) - 1) / (len(self.val_losses) - 1))
        fig, axe = plt.subplots(figsize=(8.0, 6.0))
        x = list(range(len(self.train_losses)))
        axe.plot(x, self.train_losses, label="train_loss")
        axe.plot(x[::dif], self.val_losses, label="val_loss")
        axe.set_ylabel("BCE loss")
        axe.set_xlabel("Steps")
        axe.legend()
        fig.savefig(filep, dpi=300, bbox_inches="tight")
        plt.close()

    def plot_reconstructions(self, index=0, path=None, name=None):
        root = self._rooting(path)
        self._setup_models("eval")
        if name is None:
            filep = os.path.join(root, "reconstructions.png")
        else:
            filep = os.path.join(root, name)
        self._setup_models("eval")
        data = self.val_loader.dataset[index]
        batch = Batch.from_data_list([data]).to(self.device)
        label = batch.y[0].to("cpu").detach()
        with torch.no_grad():
            out = self.decoder(self.encoder(batch))
        out = torch.round(torch.sigmoid(out))[0].to("cpu").detach().numpy()
        fig, axes = plt.subplots(2, 1, figsize=(8.0, 12.0))
        ax1, ax2 = axes.flatten()
        ax1.bar(list(range(out.shape[0])), label)
        ax1.set_xlabel("PubChem Fingerprint")
        ax2.bar(list(range(out.shape[0])), out)
        ax2.set_xlabel("Reconstructed Fingerprint")
        fig.savefig(filep, dpi=300, bbox_inches="tight")
        plt.close()


class EncoderClassifierTrainer(EncoderDecoderTrainer):
    def __init__(
        self, config, encoder=None, decoder=None, train_loader=None, val_loader=None
    ):
        super().__init__(config, encoder, decoder, train_loader, val_loader)
        self._tr_accs = list()
        self._val_accs = list()

    def _parse_config(self):
        super()._parse_config()
        self._early_stopper = self.config["early_stopper"]()
        self._criterion = self.config["classifier_loss"]()

    @property
    def train_accs(self):
        return self._tr_accs

    @property
    def validate_accs(self):
        return self._val_accs

    @property
    def criterion(self):
        return self._criterion

    def train(self):
        self.epoch = 0
        self.log_before_training_status()
        while self.epoch < self.config["classifier_epochs"]:
            if self.epoch < self.config["frozen_epochs"]:
                self.load_optimizers(self._decoder_optimizer)
            else:
                self.load_optimizers(self._encoder_optimizer, self._decoder_optimizer)
            self.train_one_epoch()
            self.validate()
            stop = self.early_stopper.stop(
                self.epoch,
                self._cur_val_loss,
                val_acc=self._cur_val_acc,
                train_loss=self._cur_train_loss,
                train_acc=self._cur_train_acc,
            )
            if stop:
                break
            self.encoder_lr_scheduler.step()
            self.decoder_lr_scheduler.step()
            self.epoch += 1
        metrics = self.early_stopper.get_best_vl_metrics()
        print(
            f"Best train loss: {metrics[0]:.4f}, "
            f"best train acc: {metrics[1]:.4f}, "
            f"best validate loss: {metrics[2]:.4f}, "
            f"best validate acc: {metrics[3]:.4f}"
        )

    def log_before_training_status(self):
        with torch.no_grad():
            self._setup_models("train")
            it = zip(
                [self.train_loader, self.val_loader],
                [self.train_losses, self.val_losses],
                [self.train_accs, self.validate_accs],
            )
            for loader, losses, accs in it:
                batch_losses = list()
                batch_accs = list()
                for batch in loader:
                    batch = batch.to(self.device)
                    out = self.decoder(self.encoder(batch))
                    try:
                        batch_losses.append(self.criterion(out, batch.y).item())
                    except RuntimeError:
                        batch_losses.append(self.criterion(out, batch.y.float()).item())
                    batch_accs.append(accuracy(out, batch.y))
                loss = mean(batch_losses)
                losses.append(loss)
                acc = mean(batch_accs)
                accs.append(acc)

    def train_one_epoch(self):
        self._setup_models("train")
        batch_losses = list()
        batch_accs = list()
        for i, batch in enumerate(self.train_loader):
            batch = batch.to(self.device)
            out = self.decoder(self.encoder(batch))
            try:
                train_loss = self.criterion(out, batch.y)
            except RuntimeError:
                train_loss = self.criterion(out, batch.y.float())
            batch_losses.append(train_loss.item())
            for opt in self._optimizers:
                opt.zero_grad()
            train_loss.backward()
            for opt in self._optimizers:
                opt.step()
            acc = accuracy(out, batch.y)
            batch_accs.append(acc)
            print(
                f"\repoch: {self.epoch+1}, batch: {i+1}, "
                f"train_loss: {train_loss.item():.4f}, train_acc: {acc:.4f}",
                end=" ",
            )
        self._cur_train_loss = mean(batch_losses)
        self._cur_train_acc = mean(batch_accs)
        self.train_losses.append(self._cur_train_loss)
        self.train_accs.append(self._cur_train_acc)

    def validate(self):
        self._setup_models("eval")
        with torch.no_grad():
            batch_losses = list()
            batch_acc = list()
            for batch in self.val_loader:
                batch = batch.to(self.device)
                out = self.decoder(self.encoder(batch))
                batch_losses.append(self.criterion(out, batch.y).item())
                acc = accuracy(out, batch.y)
                batch_acc.append(acc)
        self._cur_val_loss = mean(batch_losses)
        self._cur_val_acc = mean(batch_acc)
        self.val_losses.append(self._cur_val_loss)
        self.validate_accs.append(self._cur_val_acc)
        print(f"val_loss: {self._cur_val_loss:.4f}, val_acc: {self._cur_val_acc:.4f}")

    def log_results(self, out=None, txt_name=None, pk_name=None):
        root = self._rooting(out)
        if txt_name is None:
            txt_file = os.path.join(root, "classifier_training_metrics.txt")
        else:
            txt_file = os.path.join(root, txt_name)
        if pk_name is None:
            pk_file = os.path.join(root, "classifier_losses_accs.pk")
        else:
            pk_file = os.path.join(root, pk_name)
        with open(txt_file, "w") as f:
            metrics = self.early_stopper.get_best_vl_metrics()
            f.write(
                f"Best train loss: {metrics[0]:.4f}, "
                f"best train acc: {metrics[1]:.4f}, "
                f"best validate loss: {metrics[2]:.4f}, "
                f"best validate acc: {metrics[3]}"
            )
        with open(pk_file, "wb") as f:
            loss_dict = (
                {
                    "training_losses": self.train_losses,
                    "validating_losses": self.val_losses,
                    "training_accs": self.train_accs,
                    "validating_accs": self.validate_accs,
                },
            )
            pk.dump(loss_dict, f)

    def plot_training_metrics(self, path=None, name=None):
        root = self._rooting(path)
        if name is None:
            filep = os.path.join(root, "classifier_train_val_losses.png")
        else:
            filep = os.path.join(root, name)
        fig, axes = plt.subplots(ncols=2, figsize=(16.0, 6.0))
        x = list(range(len(self.train_losses)))
        axes[0].plot(x, self.train_losses, label="train_loss")
        axes[0].plot(x, self.val_losses, label="val_loss")
        axes[0].set_ylabel("BCE loss")
        axes[0].set_xlabel("Epochs")
        axes[1].set_title("Losses")
        axes[0].legend()
        axes[1].plot(x, self.train_accs, label="train_acc")
        axes[1].plot(x, self.validate_accs, label="val_acc")
        axes[1].set_ylabel("Accuracy")
        axes[1].set_xlabel("Epochs")
        axes[1].set_title("Accuracies")
        axes[1].legend()
        fig.savefig(filep, dpi=300, bbox_inches="tight")
        plt.close()
