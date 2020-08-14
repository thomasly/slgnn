""" **Trainers used to train encoder and classifier.**
"""

import os
from abc import ABC, abstractmethod
import pickle as pk
from statistics import mean

import torch
from torch_geometric.data import Batch

# import wandb
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")


class BaseTrainer(ABC):
    """ The base class for trainers.

    Args:
        config: an instance of slgnn.configs.base.Config class.
        model: pytorch model.
        dataloader: pytorch-geometric dataloader.
    """

    def __init__(
        self,
        config,
        model=None,
        dataloader=None,
        # wandb=None
    ):
        self._model = model
        self._dataloader = dataloader
        self._config = config
        # self._wandb = wandb
        self._parse_config()
        self._tr_losses = list()
        self._val_losses = list()

    def _parse_config(self):
        self._lr = self.config["learning_rate"]
        self._early_stopper = self.config["early_stopper"]()
        self._device = torch.device(self.config["device"])

    @property
    def device(self):
        """ The device on which to train the model.
        """
        return self._device

    @property
    def dataloader(self):
        """ The dataloader. Mutable.
        """
        if self._dataloader is None:
            raise AttributeError(
                "dataloader is not initialized. Please initialize the "
                "dataloader by passing the dataloader to constructor or use "
                "self.dataloader = ... after the instance is constructed."
            )
        return self._dataloader

    @dataloader.setter
    def dataloader(self, dataloader):
        self._dataloader = dataloader

    @property
    def config(self):
        """ The Config instance. Mutable.
        """
        return self._config

    @config.setter
    def config(self, conf):
        self._config = conf

    # @property
    # def wandb(self):
    #     return self._wandb

    # @wandb.setter
    # def wandb(self, value):
    #     assert isinstance(value, type(wandb))
    #     self._wandb = value

    @property
    def train_losses(self):
        """ Recorded train losses.
        """
        return self._tr_losses

    @property
    def val_losses(self):
        """ Recorded validation losses.
        """
        return self._val_losses

    @property
    def early_stopper(self):
        """ The early stopper used in training.
        """
        return self._early_stopper

    @property
    @abstractmethod
    def criterion(self):
        """ The criterion used to calculate loss.
        """
        pass


class EncoderDecoderTrainer(BaseTrainer):
    """ Trainer class used to pre-train encoder with a decoder.

    Args:
        config: An instance of slgnn.configs.base.Config class.
        encoder: encoder model.
        decoder: decoder model.
        dataloader: the dataloader feeding training, validation, and testing data.
    """

    def __init__(
        self,
        config,
        encoder=None,
        decoder=None,
        dataloader=None,
        # wandb=None,
    ):
        self._encoder = encoder
        self._decoder = decoder
        self._tr_metrics = list()
        self._val_metrics = list()
        self._freeze = True
        super().__init__(
            config=config,
            dataloader=dataloader,
            # wandb=wandb
        )

    def _parse_config(self):
        """ Parse the config instance and initialize neccessary components for training.
        """
        self._lr = self.config["learning_rate"]
        self._device = torch.device(self.config["device"])
        self._early_stopper = self.config["encoder_early_stopper"]()

        encoder_parameters = self._encoder.get_model_parameters_by_layer()
        encoder_parameters = [
            v for _, v in sorted(encoder_parameters.items(), reverse=True)
        ]
        self._encoder_optimizers = [
            self.config["optimizer"](params, lr=self._lr)
            for params in encoder_parameters
        ]
        self._decoder_optimizer = self.config["optimizer"](
            self._decoder.parameters(), lr=self._lr
        )

        self._encoder_lr_schedulers = [
            self.config["scheduler"](opt) for opt in self._encoder_optimizers
        ]
        self._decoder_lr_scheduler = self.config["scheduler"](self._decoder_optimizer)
        self._criterion = self.config["encoder_loss"]()
        self._metrics = [m() for m in self.config["metrics"]]

    def _setup_models(self, mode="train"):
        """ Change model status and move the model to self.device.

        Args:
            mode (str): Set encoder and decoder to train status (enable normalization
                and dropout) if mode == "train". Otherwise set encoder and decoder to
                eval mode.
        """
        self.encoder = self.encoder.to(self.device)
        self.decoder = self.decoder.to(self.device)
        if mode == "train":
            self.encoder.train()
            self.decoder.train()
        else:
            self.encoder.eval()
            self.decoder.eval()

    def load_optimizers(self, *optimizers):
        """ Load the optimizers to be used in training. Encoder and decoder have
        seperated optimizers. Encoder has more than one optimizer for gradient
        parameter unfreeze. Only the loaded optimizers will be used to update model
        parameters.

        Args:
            *optimizers: optimizers to be loaded.
        """
        self._optimizers = optimizers

    def load_schedulers(self, *schedulers):
        """ Load the learning rate schedulers needed to be updated. The scheduler for
        frozen parameters should not be updated before unfreezing.

        Args:
            *schedulers: learning rate schedulers to be loaded.
        """
        self._schedulers = schedulers

    def _get_current_idx(self):
        """ Get the index of unfreezing stage. The index is used to decide which
        optimizers and schedulers should be loaded.

        Returns:
            int: the first index in frozen_epochs that current epoch is less than. If
                current epoch is greater than all of the numbers in frozen_epochs,
                return the length of the frozen_epochs list.

        Example:
            If current epoch is 16. The config["frozen_epochs"] have value
            [5, 10, 15, 20, 25]. 16 is greater than 15 and less than 20, so 3 (index of
            20) is returned. If current epoch is greater than 25, 5 is returned.
        """
        for i, ep in enumerate(self.config["frozen_epochs"]):
            if self.epoch > ep:
                continue
            return i
        return i + 1

    def _determine_optimizers(self):
        """ Load optimizers based on current epoch.
        """
        if self.freeze_encoder:
            idx = self._get_current_idx()
            self.load_optimizers(
                self._decoder_optimizer, *self._encoder_optimizers[0:idx]
            )
        else:
            self.load_optimizers(self._decoder_optimizer, *self._encoder_optimizers)

    def _determine_schedulers(self):
        """ Load schedulers based on current epoch.
        """
        if self.freeze_encoder:
            idx = self._get_current_idx()
            self.load_schedulers(
                self._decoder_lr_scheduler, *self._encoder_lr_schedulers[0:idx]
            )
        else:
            self.load_schedulers(
                self._decoder_lr_scheduler, *self._encoder_lr_schedulers
            )

    @property
    def encoder(self):
        """ The encoder model. Mutable.
        """
        return self._encoder

    @encoder.setter
    def encoder(self, model):
        self._encoder = model

    @property
    def decoder(self):
        """ The decoder model. Mutable.
        """
        return self._decoder

    @decoder.setter
    def decoder(self, model):
        self._decoder = model

    @property
    def train_metrics(self):
        """ The recorded training metrics values.
        """
        return self._tr_metrics

    @property
    def val_metrics(self):
        """ The recorded validating metrics values.
        """
        return self._val_metrics

    @property
    def freeze_encoder(self):
        """ A boolean value. Whether freeze the encoder parameters. Mutable.
        """
        return self._freeze

    @freeze_encoder.setter
    def freeze_encoder(self, value):
        assert isinstance(value, bool)
        self._freeze = value

    @property
    def epochs(self):
        """ Total number of training epochs.
        """
        return self.config["encoder_epochs"]

    @property
    def criterion(self):
        """ The criterion used to calculate loss.
        """
        return self._criterion

    @property
    def metrics(self):
        """ Metrics used for monitoring training. Mutable.
        """
        return self._metrics

    @metrics.setter
    def metrics(self, value):
        assert isinstance(value, list)
        self._metrics = value

    @property
    def encoder_lr_scheduler(self):
        """ The learning rate schedulers for encoder.
        """
        return self._encoder_optimizer

    @property
    def decoder_lr_scheduler(self):
        """ The learning rate scheduler for decoder.
        """
        return self._decoder_lr_scheduler

    def _get_metrics_string(self, metrics):
        """ Create a printable string of the best metrics values returned by
        early_stopper.

        Args:
            metrics (dict): recorded best metrics.

        Returns:
            str: a printable string.
        """
        s = ""
        for k, v in metrics.items():
            s += f"Best {k}: "
            s += f"{v:.4f}, "
        # remove the last comma
        return s[:-2]

    def train(self):
        """ Train encoder and decoder.
        """
        self.epoch = 0
        self.log_before_training_status()
        while self.epoch < self.epochs:
            # if self.wandb:
            #     self.wandb.log({"epoch": self.epoch})
            self._determine_optimizers()
            self._determine_schedulers()
            self.train_one_epoch()
            self.validate()
            metrics_dict = {
                "val_loss": self._cur_val_loss,
                "train_loss": self._cur_train_loss,
            }
            metrics_dict.update(
                {
                    "train_" + m_name: m_value
                    for m_name, m_value in zip(
                        self.config["metrics_name"], self._cur_train_metrics
                    )
                }
            )
            metrics_dict.update(
                {
                    "val_" + m_name: m_value
                    for m_name, m_value in zip(
                        self.config["metrics_name"], self._cur_val_metrics
                    )
                }
            )
            stop = self.early_stopper.stop(self.epoch, metrics_dict)
            if stop:
                break
            for sch in self._schedulers:
                sch.step()
            self.epoch += 1
        print(self._get_metrics_string(self.early_stopper.get_best_vl_metrics()))

    def log_before_training_status(self):
        """ Log the metrics before the first epoch with the randomly initialized model.
        """
        with torch.no_grad():
            self._setup_models("train")
            it = zip(
                [self.dataloader.train_loader, self.dataloader.val_loader],
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

    def train_one_epoch(self):
        """ Train the encoder and decoder one epoch.
        """
        self._setup_models("train")
        batch_losses = list()
        batch_metrics = list()
        for i, batch in enumerate(self.dataloader.train_loader):
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
        self.train_metrics.append(self._cur_train_metrics)

    def validate(self):
        """ Validating the encoder and decoder with validating dataset.
        """
        self._setup_models("eval")
        with torch.no_grad():
            outputs = list()
            labels = list()
            for batch in self.dataloader.val_loader:
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

    def test(self):
        """ Testing the encoder and decoder with test dataset.
        """
        self._setup_models("test")
        with torch.no_grad():
            outputs = list()
            labels = list()
            for batch in self.dataloader.test_loader:
                batch = batch.to(self.device)
                outputs.append(self.decoder(self.encoder(batch)))
                labels.append(batch.y)
            out = torch.cat(outputs, 0)
            y = torch.cat(labels)
            self.test_metrics = [m(out, y) for m in self.metrics]
        print(f"test:", end=" ")
        for smet, met in zip(self.metrics, self.test_metrics):
            print(f"{smet.name}: {met:.4f}", end=" ")
        print()

    def _rooting(self, path):
        if path is None:
            root = os.path.curdir
        else:
            root = path
        os.makedirs(root, exist_ok=True)
        return root

    def log_results(self, out=None, txt_name=None, pk_name=None):
        """ Log training results:

        Args:
            out (str): optional. Path to save the results. Default is the current
                working direcory.
            txt_name (str): optional. Name of the txt file saving the best training
                metrics. Default is "training_metrics.txt".
            pk_name (str): optional. Name of the pickle file saving all metrics values
                during the whole training process. Default is "losses.pk". (Not only
                losses, but all metrics values are saved in the file.)
        """
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
            f.write("Validation:\n")
            f.write(self._get_metrics_string(metrics) + "\n")
            if hasattr(self, "test_metrics"):
                f.write("Testing:\n")
                for smet, met in zip(self.metrics, self.test_metrics):
                    f.write(f"{smet.name}: {met:.4f} ")
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
        """ Plotting the training and validation loss curves.

        Args:
            path (str): optional. Path to save the plottings. Default is the current
                working directory.
            name (str): optional. Filename of the plotting. Default:
                "train_val_losses.png".
        """
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
        """ Plot reconstruction bar chart with validation data.

        Args:
            index (int): optional. The index of the validation data to use. Default is
                0.
            path (str): optional. Path to save the plottings. Default is the current
                working directory.
            name (str): optional. Name of the saved plotting. Default is
                "reconstructions.png".
        """
        root = self._rooting(path)
        self._setup_models("eval")
        if name is None:
            filep = os.path.join(root, "reconstructions.png")
        else:
            filep = os.path.join(root, name)
        self._setup_models("eval")
        data = self.dataloader.val_loader.dataset[index]
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
    """ Trainer for encoder and classifier.

    Args:
        config: An instance of slgnn.configs.base.Config class.
        encoder: encoder model.
        decoder: decoder model.
        dataloader: the dataloader feeding training, validation, and testing data.
    """

    def __init__(
        self,
        config,
        encoder=None,
        decoder=None,
        dataloader=None,
        # wandb=None,
    ):
        super().__init__(
            config=config,
            encoder=encoder,
            decoder=decoder,
            dataloader=dataloader,
            # wandb=wandb,
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
        """ Number of total epoch to train the classifier.
        """
        return self.config["classifier_epochs"]

    def plot_training_metrics(self, path=None, name=None):
        """ Plot classifier training and validation losses.

        Args:
            path (str): optional. Path to save the plottings. Default is the current
                working directory.
            name (str): optional. Name of the plotting. Default is
                "classifier_train_val_losses.png".
        """
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


class MaskedGraphTrainer(EncoderDecoderTrainer):
    def _parse_config(self):
        super()._parse_config()
        self._mask_rate = self.config["mask_rate"]

    def log_before_training_status(self):
        """ Log the metrics before the first epoch with the randomly initialized model.
        """
        with torch.no_grad():
            self._setup_models("train")
            it = zip(
                [self.dataloader.train_loader, self.dataloader.val_loader],
                [self.train_losses, self.val_losses],
                [self.train_metrics, self.val_metrics],
                ["train", "val"],
            )
            for loader, losses, metrics, mode in it:
                outputs = list()
                labels = list()
                for batch in loader:
                    batch = batch.to(self.device)
                    outputs.append(
                        self.decoder(self.encoder(batch)[batch.masked_atom_indices])
                    )
                    labels.append(batch.mask_node_label[:, 0])
                out = torch.cat(outputs, 0)
                y = torch.cat(labels)
                try:
                    loss = self.criterion(out, y).item()
                except RuntimeError:
                    loss = self.criterion(out, y.float()).item()
                metric = [m(out, y) for m in self.metrics]
                losses.append(loss)
                metrics.append(metric)

    def train_one_epoch(self):
        """ Train the encoder and decoder one epoch.
        """
        self._setup_models("train")
        batch_losses = list()
        batch_metrics = list()
        for i, batch in enumerate(self.dataloader.train_loader):
            batch = batch.to(self.device)
            node_rep = self.encoder(batch)
            out = self.decoder(node_rep[batch.masked_atom_indices])
            try:
                train_loss = self.criterion(out, batch.mask_node_label[:, 0])
            except RuntimeError:
                train_loss = self.criterion(out, batch.mask_node_label[:, 0].float())
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
        self.train_metrics.append(self._cur_train_metrics)

    def validate(self):
        """ Validating the encoder and decoder with validating dataset.
        """
        self._setup_models("eval")
        with torch.no_grad():
            outputs = list()
            labels = list()
            for batch in self.dataloader.val_loader:
                batch = batch.to(self.device)
                outputs.append(
                    self.decoder(self.encoder(batch)[batch.masked_atom_indices])
                )
                labels.append(batch.mask_node_label[:, 0])
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
