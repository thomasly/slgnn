""" Pretraining with node features random masked
"""

import os
import logging
from datetime import datetime
import random

import torch
import yaml

from contextPred.chem.util import MaskAtom
from slgnn.configs.base import Grid, Config
from slgnn.configs.arg_parsers import ModelTrainingArgs
from slgnn.data_processing.deepchem_datasets import BACE
from slgnn.models.decoder.model import GINDecoder
from .trainers import EncoderDecoderTrainer


# criterion = nn.CrossEntropyLoss()
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
        dataset = BACE(
            transform=MaskAtom(
                num_atom_type=119,
                num_edge_type=5,
                mask_rate=config["mask_rate"],
                mask_edge=False,
            ),
        )

        log_name = "_".join(
            [
                "BACE",
                "embed{}".format(config["embedding_dim"]),
                config["encoder_loss_name"],
                str(config_idx),
            ]
        )
        log_dir = os.path.join("logs", time_stamp, log_name)
        os.makedirs(log_dir)
        with open(os.path.join(log_dir, "configs.yml"), "w") as f:
            f.write(yaml.dump(config_dict))
        dim_encoder_target = config["embedding_dim"]  # 300 in contextPred
        dim_decoder_target = dataset.data.y.size(0)
        dim_features = dataset.data.x.size(1)
        dropout = config["dropout"]
        dloader = config["encoder_data_splitter"](dataset)
        Encoder = config["model"]
        encoder = Encoder(
            dim_features=dim_features, dim_target=dim_encoder_target, config=config
        )

        decoder = GINDecoder(dim_encoder_target, dim_decoder_target, dropout)

        encoder_trainer = EncoderDecoderTrainer(config, encoder, decoder, dloader)
        encoder_trainer.metrics = []
        encoder_trainer.freeze_encoder = False
        encoder_trainer.train()
        encoder_trainer.log_results(
            out=log_dir, txt_name="encoder_losses.txt", pk_name="encoder_losses.pk"
        )
        encoder_trainer.plot_training_metrics(log_dir, name="encoder_losses.png")
        torch.save(
            encoder.state_dict(),
            os.path.join("trained_models", "BACE_pretrained_GIN.pth"),
        )
