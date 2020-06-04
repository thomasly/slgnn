import os
from abc import ABC, abstractmethod
import pickle as pk
from statistics import mean

import torch
from torch_geometric.data import Batch
import wandb
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")


class BaseTrainer(ABC):
    """ The base class for trainers
    """

    def __init__(
        self, config, model=None, train_loader=None, val_loader=None, wandb=None
    ):
        self._model = model
        self._train_loader = train_loader
        self._val_loader = val_loader
        self._config = config
        self._wandb = wandb
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
    def wandb(self):
        return self._wandb

    @wandb.setter
    def wandb(self, value):
        assert isinstance(value, type(wandb))
        self._wandb = value

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
        self,
        config,
        encoder=None,
        decoder=None,
        train_loader=None,
        val_loader=None,
        wandb=None,
    ):
        self._encoder = encoder
        self._decoder = decoder
        self._tr_metrics = list()
        self._val_metrics = list()
        self._freeze = True
        super().__init__(
            config=config, train_loader=train_loader, val_loader=val_loader, wandb=wandb
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
        self._metrics = [m() for m in self.config["metrics"]]

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
    def train_metrics(self):
        return self._tr_metrics

    @property
    def val_metrics(self):
        return self._val_metrics

    @property
    def freeze_encoder(self):
        return self._freeze

    @property
    def epochs(self):
        return self.config["encoder_epochs"]

    @freeze_encoder.setter
    def freeze_encoder(self, value):
        assert isinstance(value, bool)
        self._freeze = value

    @property
    def criterion(self):
        return self._criterion

    @property
    def metrics(self):
        return self._metrics

    @metrics.setter
    def metrics(self, value):
        assert isinstance(value, list)
        self._metrics = value

    @property
    def encoder_lr_scheduler(self):
        return self._encoder_optimizer

    @property
    def decoder_lr_scheduler(self):
        return self._decoder_lr_scheduler

    def _get_metrics_string(self, metrics):
        s = ""
        if metrics[0] is not None:
            s += f"Best train loss: {metrics[0]:.4f} "
        if metrics[1] is not None:
            s += f"best train {self.metrics[0].name}: {metrics[1]:.4f} "
        if metrics[2] is not None:
            s += f"best validate loss: {metrics[2]:.4f} "
        if metrics[3] is not None:
            s += f"best validate {self.metrics[0].name}: {metrics[3]:.4f} "
        if metrics[4] is not None:
            s += f"best test loss: {metrics[4]:.4f} "
        if metrics[5] is not None:
            s += f"best test {self.metrics[0].name}: {metrics[5]:.4f}"
        return s

    def train(self):
        self.epoch = 0
        self.log_before_training_status()
        while self.epoch < self.epochs:
            if self.wandb:
                self.wandb.log({"epoch": self.epoch})
            if self.epoch < self.config["frozen_epochs"] and self.freeze_encoder:
                self.load_optimizers(self._decoder_optimizer)
            else:
                self.load_optimizers(self._encoder_optimizer, self._decoder_optimizer)

            self.train_one_epoch()
            self.validate()
            try:
                stop = self.early_stopper.stop(
                    self.epoch,
                    self._cur_val_loss,
                    val_acc=self._cur_val_metrics[0],
                    train_loss=self._cur_train_loss,
                    train_acc=self._cur_train_metrics[0],
                )
            except IndexError:
                stop = self.early_stopper.stop(
                    self.epoch, self._cur_val_loss, train_loss=self._cur_train_loss,
                )
            if stop:
                break
            self.encoder_lr_scheduler.step()
            self.decoder_lr_scheduler.step()
            self.epoch += 1
        metrics = self.early_stopper.get_best_vl_metrics()
        print(self._get_metrics_string(metrics))

    def log_before_training_status(self):
        with torch.no_grad():
            self._setup_models("train")
            it = zip(
                [self.train_loader, self.val_loader],
                [self.train_losses, self.val_losses],
                [self.train_metrics, self.val_metrics],
                ["train", "val"],
            )
            for loader, losses, metrics, mode in it:
                outputs = list()
                labels = list()
                for batch in loader:
                    batch = batch.to(self.device)
                    outputs.append(self.decoder(self.encoder(batch)))
                    labels.append(batch.y)
                out = torch.cat(outputs, 0)
                y = torch.cat(labels)
                try:
                    loss = self.criterion(out, y).item()
                except RuntimeError:
                    loss = self.criterion(out, y.float()).item()
                metric = [m(out, y) for m in self.metrics]
                losses.append(loss)
                metrics.append(metric)
                if self.wandb:
                    self.wandb.log({mode + "_loss": loss})
                    self.wandb.log(
                        {
                            mode + "_" + key.name: value
                            for key, value in zip(self.metrics, metric)
                        }
                    )

    def train_one_epoch(self):
        self._setup_models("train")
        batch_losses = list()
        batch_metrics = list()
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
            metrics = [m(out, batch.y) for m in self.metrics]
            batch_metrics.append(metrics)
            print(
                f"\repoch: {self.epoch+1}, batch: {i+1}, "
                f"train_loss: {train_loss.item():.4f}",
                end=" ",
            )
            for smet, met in zip(self.metrics, metrics):
                print(f"{smet.name}: {met:.4f}", end=" ")
        self._cur_train_loss = mean(batch_losses)
        self._cur_train_metrics = [mean(m) for m in zip(*batch_metrics)]
        self.train_losses.append(self._cur_train_loss)
        self.train_metrics.append(self._cur_train_loss)
        if self.wandb:
            self.wandb.log({"train_loss": self._cur_train_loss})
            self.wandb.log(
                {
                    "train_" + met.name: val
                    for met, val in zip(self.metrics, self._cur_train_metrics)
                }
            )

    def validate(self):
        self._setup_models("eval")
        with torch.no_grad():
            outputs = list()
            labels = list()
            for batch in self.val_loader:
                batch = batch.to(self.device)
                outputs.append(self.decoder(self.encoder(batch)))
                labels.append(batch.y)
            out = torch.cat(outputs, 0)
            y = torch.cat(labels)
            try:
                self._cur_val_loss = self.criterion(out, y).item()
            except RuntimeError:
                self._cur_val_loss = self.criterion(out, y.float()).item()
            self._cur_val_metrics = [m(out, y) for m in self.metrics]
        self.val_losses.append(self._cur_val_loss)
        self.val_metrics.append(self._cur_val_metrics)
        print(f"val_loss: {self._cur_val_loss:.4f}", end=" ")
        for smet, met in zip(self.metrics, self._cur_val_metrics):
            print(f"{smet.name}: {met:.4f}", end=" ")
        print()
        if self.wandb:
            self.wandb.log({"val_loss": self._cur_val_loss})
            self.wandb.log(
                {
                    "val_" + met.name: val
                    for met, val in zip(self.metrics, self._cur_val_metrics)
                }
            )

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
            txt_file = os.path.join(root, "training_metrics.txt")
        else:
            txt_file = os.path.join(root, txt_name)
        if pk_name is None:
            pk_file = os.path.join(root, "losses.pk")
        else:
            pk_file = os.path.join(root, pk_name)
        with open(txt_file, "w") as f:
            metrics = self.early_stopper.get_best_vl_metrics()
            f.write(self._get_metrics_string(metrics))
        with open(pk_file, "wb") as f:
            metrics_dict = {
                "training_losses": self.train_losses,
                "validating_losses": self.val_losses,
            }
            it = zip(zip(*self.train_metrics), zip(*self.val_metrics))
            for key, (tr, val) in zip(self.metrics, it):
                metrics_dict.update(
                    {f"training_{key.name}": tr, f"validating_{key.name}": val}
                )
            pk.dump(metrics_dict, f)

    def plot_training_metrics(self, path=None, name=None):
        root = self._rooting(path)
        if name is None:
            filep = os.path.join(root, "train_val_losses.png")
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
        self,
        config,
        encoder=None,
        decoder=None,
        train_loader=None,
        val_loader=None,
        wandb=None,
    ):
        super().__init__(
            config=config,
            encoder=encoder,
            decoder=decoder,
            train_loader=train_loader,
            val_loader=val_loader,
            wandb=wandb,
        )

    def _parse_config(self):
        super()._parse_config()
        self._early_stopper = self.config["early_stopper"]()
        self._criterion = self.config["classifier_loss"]()

    @property
    def criterion(self):
        return self._criterion

    @property
    def epochs(self):
        return self.config["classifier_epochs"]

    def plot_training_metrics(self, path=None, name=None):
        root = self._rooting(path)
        if name is None:
            filep = os.path.join(root, "classifier_train_val_losses.png")
        else:
            filep = os.path.join(root, name)
        ncols = len(self.metrics) + 1
        fig, axes = plt.subplots(ncols=ncols, figsize=(ncols * 8.0, 6.0))
        x = list(range(len(self.train_losses)))
        axes[0].plot(x, self.train_losses, label="train_loss")
        axes[0].plot(x, self.val_losses, label="val_loss")
        axes[0].set_ylabel("BCE loss")
        axes[0].set_xlabel("Epochs")
        axes[0].set_title("Losses")
        axes[0].legend()
        it = zip(zip(*self.train_metrics), zip(*self.val_metrics))
        for i, (tr_metric, val_metric) in enumerate(it):
            axes[i + 1].plot(x, tr_metric, label="train_" + self.metrics[i].name)
            axes[i + 1].plot(x, val_metric, label="val_" + self.metrics[i].name)
            axes[i + 1].set_ylabel(self.metrics[i].name)
            axes[i + 1].set_xlabel("Epochs")
            axes[i + 1].set_title(self.metrics[i].name)
            axes[i + 1].legend()
        fig.savefig(filep, dpi=300, bbox_inches="tight")
        plt.close()
