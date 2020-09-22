""" A easy script to run GIN model.
"""
import os
import os.path as osp
from datetime import datetime
import random
import logging

import torch
import yaml

from slgnn.configs.base import Grid, Config
from slgnn.configs.arg_parsers import ModelTrainingArgs

# from slgnn.data_processing.deepchem_datasets import ClinTox, ClinToxFP
# from slgnn.data_processing.deepchem_datasets import BACE, BACEFP
# from slgnn.data_processing.deepchem_datasets import BBBP, BBBPFP
from slgnn.data_processing.deepchem_datasets import HIV, HIVFP

# from slgnn.data_processing.deepchem_datasets import Sider, SiderFP
from slgnn.data_processing.utils import AtomFeaturesOneHotTransformer

# from slgnn.models.gcn.model import GIN
from slgnn.models.decoder.model import GINDecoder
from .trainers import EncoderDecoderTrainer, EncoderClassifierTrainer


if __name__ == "__main__":
    Dataset = HIV
    FPDataset = HIVFP
    # ZINC pretrained model
    args = ModelTrainingArgs().parse_args()
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    time_stamp = datetime.now().strftime(r"%Y%m%d_%H%M%S")
    config_grid = Grid(args.config)
    for config_idx, config_dict in enumerate(config_grid):
        config = Config.from_dict(config_dict)
        torch.manual_seed(config["random_seed"])
        random.seed(config["random_seed"])
        datasets = FPDataset(transform=AtomFeaturesOneHotTransformer())
        if config["encoder_epochs"] == 0:
            ifencoder = "nosecondpretrain"
        else:
            ifencoder = "secondpretrain"
        try:
            log_name = "_".join(
                [
                    config["encoder_dataset_name"],
                    config["classifier_dataset_name"],
                    config["classifier_data_splitter_name"],
                    "_".join(map(str, config["data_splitting_ratio"])),
                    "embed{}".format(config["embedding_dim"]),
                    config["classifier_loss_name"],
                    str(config["classifier_loss_args"]["gamma"]),
                    str(config["classifier_loss_args"]["alpha"]),
                    # str(config["batch_size"]),
                    # str(config["learning_rate"]),
                    ifencoder,
                    "_".join(map(str, config["frozen_epochs"])),
                    # config["aggregation"],
                    str(config_idx),
                ]
            )
        except AttributeError:
            log_name = "_".join(
                [
                    config["encoder_dataset_name"],
                    config["classifier_dataset_name"],
                    config["classifier_data_splitter_name"],
                    "_".join(map(str, config["data_splitting_ratio"])),
                    "embed{}".format(config["embedding_dim"]),
                    config["classifier_loss_name"],
                    # str(config["batch_size"]),
                    # str(config["learning_rate"]),
                    ifencoder,
                    "_".join(map(str, config["frozen_epochs"])),
                    # config["aggregation"],
                    str(config_idx),
                ]
            )
        log_dir = osp.join("logs", time_stamp, log_name + "_ZINC_pretrained")
        os.makedirs(log_dir)
        with open(osp.join(log_dir, "configs.yml"), "w") as f:
            f.write(yaml.dump(config_dict))

        dim_encoder_target = config["embedding_dim"]
        dim_decoder_target = datasets[0].y.size(1)
        dim_features = datasets[0].x.size(1)
        dropout = config["dropout"]
        Encoder = config["model"]
        encoder = Encoder(
            dim_features=dim_features, dim_target=dim_encoder_target, config=config
        )
        encoder.load_state_dict(
            torch.load(
                r"D:\Documents\repos\slgnn\trained_models"
                r"\pretrained_GIN_with_ZINC_epoch8.pt"
            )
        )
        if config["encoder_epochs"] > 0:
            decoder = GINDecoder(dim_encoder_target, dim_decoder_target, dropout)

            dloader = config["encoder_data_splitter"](datasets)
            encoder_trainer = EncoderDecoderTrainer(config, encoder, decoder, dloader)
            encoder_trainer.metrics = []
            encoder_trainer.freeze_encoder = False
            encoder_trainer.train()
            encoder_trainer.log_results(
                out=log_dir, txt_name="encoder_losses.txt", pk_name="encoder_losses.pk"
            )
            encoder_trainer.plot_training_metrics(log_dir, name="encoder_losses.png")
            for index in range(5):
                encoder_trainer.plot_reconstructions(
                    index, log_dir, f"reconstruction_{index}.png"
                )

        cls_dataset = Dataset(transform=AtomFeaturesOneHotTransformer())
        classifier = GINDecoder(dim_encoder_target, cls_dataset.num_classes, dropout)
        cls_dloader = config["classifier_data_splitter"](cls_dataset)
        cls_trainer = EncoderClassifierTrainer(
            config,
            encoder,
            classifier,
            cls_dloader,
        )
        cls_trainer.train()
        cls_trainer.plot_training_metrics(log_dir)
        cls_trainer.test()
        cls_trainer.log_results(
            out=log_dir,
            txt_name="classifier_metrics.txt",
            pk_name="classifier_metrics.pk",
        )

    # no ZINC pretrained model
    config_grid = Grid(args.config)
    for config_idx, config_dict in enumerate(config_grid):
        config = Config.from_dict(config_dict)
        torch.manual_seed(config["random_seed"])
        random.seed(config["random_seed"])
        datasets = FPDataset(transform=AtomFeaturesOneHotTransformer())
        if config["encoder_epochs"] == 0:
            ifencoder = "nosecondpretrain"
        else:
            ifencoder = "secondpretrain"
        try:
            log_name = "_".join(
                [
                    config["encoder_dataset_name"],
                    config["classifier_dataset_name"],
                    config["classifier_data_splitter_name"],
                    "_".join(map(str, config["data_splitting_ratio"])),
                    "embed{}".format(config["embedding_dim"]),
                    config["classifier_loss_name"],
                    str(config["classifier_loss_args"]["gamma"]),
                    str(config["classifier_loss_args"]["alpha"]),
                    # str(config["batch_size"]),
                    # str(config["learning_rate"]),
                    ifencoder,
                    "_".join(map(str, config["frozen_epochs"])),
                    # config["aggregation"],
                    str(config_idx),
                ]
            )
        except AttributeError:
            log_name = "_".join(
                [
                    config["encoder_dataset_name"],
                    config["classifier_dataset_name"],
                    config["classifier_data_splitter_name"],
                    "_".join(map(str, config["data_splitting_ratio"])),
                    "embed{}".format(config["embedding_dim"]),
                    config["classifier_loss_name"],
                    # str(config["batch_size"]),
                    # str(config["learning_rate"]),
                    ifencoder,
                    "_".join(map(str, config["frozen_epochs"])),
                    # config["aggregation"],
                    str(config_idx),
                ]
            )
        log_dir = osp.join("logs", time_stamp, log_name + "_not_ZINC_pretrained")
        os.makedirs(log_dir)
        with open(osp.join(log_dir, "configs.yml"), "w") as f:
            f.write(yaml.dump(config_dict))

        dim_encoder_target = config["embedding_dim"]
        dim_decoder_target = datasets[0].y.size(1)
        dim_features = datasets[0].x.size(1)
        dropout = config["dropout"]
        Encoder = config["model"]
        encoder = Encoder(
            dim_features=dim_features, dim_target=dim_encoder_target, config=config
        )
        if config["encoder_epochs"] > 0:
            decoder = GINDecoder(dim_encoder_target, dim_decoder_target, dropout)

            dloader = config["encoder_data_splitter"](datasets)
            encoder_trainer = EncoderDecoderTrainer(config, encoder, decoder, dloader)
            encoder_trainer.metrics = []
            encoder_trainer.freeze_encoder = False
            encoder_trainer.train()
            encoder_trainer.log_results(
                out=log_dir, txt_name="encoder_losses.txt", pk_name="encoder_losses.pk"
            )
            encoder_trainer.plot_training_metrics(log_dir, name="encoder_losses.png")
            for index in range(5):
                encoder_trainer.plot_reconstructions(
                    index, log_dir, f"reconstruction_{index}.png"
                )

        cls_dataset = Dataset(transform=AtomFeaturesOneHotTransformer())
        classifier = GINDecoder(dim_encoder_target, cls_dataset.num_classes, dropout)
        cls_dloader = config["classifier_data_splitter"](cls_dataset)
        cls_trainer = EncoderClassifierTrainer(
            config,
            encoder,
            classifier,
            cls_dloader,
        )
        cls_trainer.train()
        cls_trainer.plot_training_metrics(log_dir)
        cls_trainer.test()
        cls_trainer.log_results(
            out=log_dir,
            txt_name="classifier_metrics.txt",
            pk_name="classifier_metrics.pk",
        )
