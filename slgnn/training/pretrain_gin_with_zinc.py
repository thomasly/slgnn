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
from slgnn.data_processing.zinc_dataset import ZINC
from slgnn.data_processing.utils import AtomFeaturesOneHotTransformer

# from slgnn.models.gcn.model import GIN
from slgnn.models.decoder.model import GINDecoder
from .trainers import EncoderDecoderTrainer


if __name__ == "__main__":
    args = ModelTrainingArgs().parse_args()
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    config_grid = Grid(args.config)
    time_stamp = datetime.now().strftime(r"%Y%m%d_%H%M%S")
    for config_idx, config_dict in enumerate(config_grid):
        config = Config.from_dict(config_dict)
        torch.manual_seed(config["random_seed"])
        random.seed(config["random_seed"])
        dataset = ZINC(transform=AtomFeaturesOneHotTransformer())

        log_name = "_".join(
            [
                config["encoder_dataset_name"],
                "embed{}".format(config["embedding_dim"]),
                "hiddenunits{}".format(config["hidden_units"][0]),
                str(config["batch_size"]),
                str(config["learning_rate"]),
                config["aggregation"],
                str(config_idx),
            ]
        )

        log_dir = osp.join("logs", time_stamp, log_name)
        os.makedirs(log_dir)
        with open(osp.join(log_dir, "configs.yml"), "w") as f:
            f.write(yaml.dump(config_dict))

        dim_encoder_target = config["embedding_dim"]
        dim_decoder_target = dataset[0].y.size(1)
        dim_features = dataset[0].x.size(1)

        dropout = config["dropout"]
        Encoder = config["model"]
        encoder = Encoder(
            dim_features=dim_features, dim_target=dim_encoder_target, config=config
        )

        decoder = GINDecoder(dim_encoder_target, dim_decoder_target, dropout)

        dloader = config["encoder_data_splitter"](dataset)
        # init trainer
        encoder_trainer = EncoderDecoderTrainer(config, encoder, decoder, dloader)
        encoder_trainer.metrics = []
        encoder_trainer.freeze_encoder = False
        encoder_trainer.train(save_model=True, save_path=log_dir)
        # save and plot training results
        encoder_trainer.log_results(
            out=log_dir, txt_name="encoder_losses.txt", pk_name="encoder_losses.pk"
        )
        encoder_trainer.plot_training_metrics(log_dir, name="encoder_losses.png")
        for index in range(5):
            encoder_trainer.plot_reconstructions(
                index, log_dir, f"reconstruction_{index}.png"
            )
