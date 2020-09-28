""" Train GIN model with or without pretrained parameters.
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
from slgnn.data_processing.deepchem_datasets import ClinTox, ClinToxFP
from slgnn.data_processing.deepchem_datasets import BACE, BACEFP
from slgnn.data_processing.deepchem_datasets import BBBP, BBBPFP
from slgnn.data_processing.deepchem_datasets import HIV, HIVFP
from slgnn.data_processing.deepchem_datasets import Sider, SiderFP
from slgnn.data_processing.pyg_datasets import JAK1, JAK1FP, JAK2, JAK2FP, JAK3, JAK3FP
from slgnn.data_processing.utils import AtomFeaturesOneHotTransformer
from slgnn.models.decoder.model import GINDecoder
from .trainers import EncoderDecoderTrainer, EncoderClassifierTrainer


if __name__ == "__main__":
    args = ModelTrainingArgs().parse_args()
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    time_stamp = datetime.now().strftime(r"%Y%m%d_%H%M%S")

    exp_datasets = {
        "ClinTox": [ClinTox, ClinToxFP],
        "BACE": [BACE, BACEFP],
        "BBBP": [BBBP, BBBPFP],
        "HIV": [HIV, HIVFP],
        "Sider": [Sider, SiderFP],
        "JAK1": [JAK1, JAK1FP],
        "JAK2": [JAK2, JAK2FP],
        "JAK3": [JAK3, JAK3FP],
    }
    random_seeds = [0, 5, 193, 84234, 839574]
    # random_seeds = [0, 1]
    pretrained_model = os.path.join(args.pretrained_model)

    for ds_name, ds in exp_datasets.items():
        Dataset = ds[0]
        FPDataset = ds[1]
        config_grid = Grid(os.path.join("model_configs", ds_name + ".yml"))
        for config_idx, config_dict in enumerate(config_grid):
            for seed_idx, seed in enumerate(random_seeds):
                config = Config.from_dict(config_dict)
                torch.manual_seed(seed)
                random.seed(seed)
                dataset = FPDataset(transform=AtomFeaturesOneHotTransformer())
                if config["encoder_epochs"] == 0:
                    ifencoder = "nosecondpretrain"
                else:
                    ifencoder = "secondpretrain"
                log_name = "_".join(
                    [
                        config["classifier_dataset_name"],
                        config["classifier_data_splitter_name"],
                        "_".join(map(str, config["data_splitting_ratio"])),
                        "embed{}".format(config["embedding_dim"]),
                        config["classifier_loss_name"],
                        "bs" + str(config["batch_size"]),
                        ifencoder,
                        "freeze" + "_".join(map(str, config["frozen_epochs"])),
                        str(config_idx),
                    ]
                )
                log_dir = osp.join(
                    "logs",
                    time_stamp,
                    ds_name,
                    "ZINC_pretrained_" + log_name,
                    str(seed_idx),
                )
                os.makedirs(log_dir)
                # log configs
                with open(osp.join(log_dir, osp.pardir, "configs.yml"), "w") as f:
                    f.write(yaml.dump(config_dict))
                # init encoder
                dim_encoder_target = config["embedding_dim"]
                dim_decoder_target = dataset[0].y.size(1)
                dim_features = dataset[0].x.size(1)
                dropout = config["dropout"]
                Encoder = config["model"]
                encoder = Encoder(
                    dim_features=dim_features,
                    dim_target=dim_encoder_target,
                    config=config,
                )
                encoder.load_state_dict(torch.load(args.pretrained_model))
                # init decoder and pretrain
                if config["encoder_epochs"] > 0:
                    decoder = GINDecoder(
                        dim_encoder_target, dim_decoder_target, dropout
                    )

                    dloader = config["encoder_data_splitter"](dataset)
                    encoder_trainer = EncoderDecoderTrainer(
                        config, encoder, decoder, dloader
                    )
                    encoder_trainer.metrics = []
                    encoder_trainer.freeze_encoder = False
                    encoder_trainer.train()
                    encoder_trainer.log_results(
                        out=log_dir,
                        txt_name="encoder_losses.txt",
                        pk_name="encoder_losses.pk",
                    )
                    encoder_trainer.plot_training_metrics(
                        log_dir, name="encoder_losses.png"
                    )
                    for index in range(5):
                        encoder_trainer.plot_reconstructions(
                            index, log_dir, f"reconstruction_{index}.png"
                        )

                # fine-tune
                cls_dataset = Dataset(transform=AtomFeaturesOneHotTransformer())
                classifier = GINDecoder(
                    dim_encoder_target, cls_dataset.num_classes, dropout
                )
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
        config_grid = Grid(os.path.join("model_configs", ds_name + ".yml"))
        for config_idx, config_dict in enumerate(config_grid):
            for seed_idx, seed in enumerate(random_seeds):
                config = Config.from_dict(config_dict)
                torch.manual_seed(seed)
                random.seed(seed)
                dataset = FPDataset(transform=AtomFeaturesOneHotTransformer())
                if config["encoder_epochs"] == 0:
                    ifencoder = "nosecondpretrain"
                else:
                    ifencoder = "secondpretrain"
                log_name = "_".join(
                    [
                        config["classifier_dataset_name"],
                        config["classifier_data_splitter_name"],
                        "_".join(map(str, config["data_splitting_ratio"])),
                        "embed{}".format(config["embedding_dim"]),
                        config["classifier_loss_name"],
                        "bs" + str(config["batch_size"]),
                        ifencoder,
                        "freeze" + "_".join(map(str, config["frozen_epochs"])),
                        str(config_idx),
                    ]
                )
                log_dir = osp.join(
                    "logs",
                    time_stamp,
                    ds_name,
                    "not_ZINC_pretrained_" + log_name,
                    str(seed_idx),
                )
                os.makedirs(log_dir)
                with open(osp.join(log_dir, osp.pardir, "configs.yml"), "w") as f:
                    f.write(yaml.dump(config_dict))

                dim_encoder_target = config["embedding_dim"]
                dim_decoder_target = dataset[0].y.size(1)
                dim_features = dataset[0].x.size(1)
                dropout = config["dropout"]
                Encoder = config["model"]
                encoder = Encoder(
                    dim_features=dim_features,
                    dim_target=dim_encoder_target,
                    config=config,
                )
                if config["encoder_epochs"] > 0:
                    decoder = GINDecoder(
                        dim_encoder_target, dim_decoder_target, dropout
                    )

                    dloader = config["encoder_data_splitter"](dataset)
                    encoder_trainer = EncoderDecoderTrainer(
                        config, encoder, decoder, dloader
                    )
                    encoder_trainer.metrics = []
                    encoder_trainer.freeze_encoder = False
                    encoder_trainer.train()
                    encoder_trainer.log_results(
                        out=log_dir,
                        txt_name="encoder_losses.txt",
                        pk_name="encoder_losses.pk",
                    )
                    encoder_trainer.plot_training_metrics(
                        log_dir, name="encoder_losses.png"
                    )
                    for index in range(5):
                        encoder_trainer.plot_reconstructions(
                            index, log_dir, f"reconstruction_{index}.png"
                        )

                cls_dataset = Dataset(transform=AtomFeaturesOneHotTransformer())
                classifier = GINDecoder(
                    dim_encoder_target, cls_dataset.num_classes, dropout
                )
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
