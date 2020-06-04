""" A easy script to run GIN model.
"""
import os
import os.path as osp
from datetime import datetime
import random

import torch
import yaml

from slgnn.configs.base import Grid, Config
from slgnn.configs.arg_parsers import ModelTrainingArgs
from slgnn.models.gcn.model import GIN
from slgnn.models.decoder.model import GINDecoder
from slgnn.data_processing.loaders import DataSplitter
from .trainers import EncoderDecoderTrainer, EncoderClassifierTrainer


if __name__ == "__main__":
    import wandb

    wandb.init(project="slgnn")
    wandb = None
    torch.manual_seed(0)
    random.seed(0)
    args = ModelTrainingArgs().parse_args()
    config_grid = Grid(args.config)
    time_stamp = datetime.now().strftime(r"%Y%m%d_%H%M%S")
    for config_idx, config_dict in enumerate(config_grid):
        config = Config.from_dict(config_dict)
        datasets = config["encoder_dataset"]
        log_dir = osp.join(
            "logs", "GIN", time_stamp, config["encoder_dataset_name"], str(config_idx)
        )
        os.makedirs(log_dir)
        with open(osp.join(log_dir, "configs.yml"), "w") as f:
            f.write(yaml.dump(config_dict))

        dim_encoder_target = config["embedding_dim"]
        dim_decoder_target = datasets[0].data.y.size(1)
        dim_features = datasets[0].data.x.size(1)
        dropout = config["dropout"]
        encoder = GIN(
            dim_features=dim_features, dim_target=dim_encoder_target, config=config
        )
        # wandb.watch(encoder)
        if config["encoder_epochs"] > 0:
            decoder = GINDecoder(dim_encoder_target, dim_decoder_target, dropout)

            dloader = DataSplitter(
                datasets, ratio=[0.9, 0.1, 0], batch_size=config["batch_size"]
            )
            encoder_trainer = EncoderDecoderTrainer(
                config,
                encoder,
                decoder,
                dloader.train_loader,
                dloader.val_loader,
                wandb,
            )
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

        cls_dataset = config["classifier_dataset"]()
        classifier = GINDecoder(dim_encoder_target, cls_dataset.num_classes, dropout)
        # wandb.watch(classifier)
        cls_dloader = DataSplitter(
            cls_dataset, ratio=config["data_ratio"], batch_size=config["batch_size"]
        )
        cls_trainer = EncoderClassifierTrainer(
            config,
            encoder,
            classifier,
            cls_dloader.train_loader,
            cls_dloader.val_loader,
            wandb,
        )
        cls_trainer.train()
        cls_trainer.plot_training_metrics(log_dir)
        cls_trainer.log_results(
            out=log_dir,
            txt_name="classifier_metrics.txt",
            pk_name="classifier_metrics.pk",
        )
