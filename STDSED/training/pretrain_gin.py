""" A script to pretrain GIN model with drugbank and CYP450 datasets.
"""
import os
import os.path as osp
from datetime import datetime
import random
import logging

import torch
import yaml

from slgnn.configs.base import Config, read_config_file
from slgnn.configs.arg_parsers import ModelTrainingArgs
from slgnn.data_processing.utils import AtomFeaturesOneHotTransformer
from slgnn.data_processing.loaders import DataSplitter
from slgnn.models.decoder.model import GINDecoder
from slgnn.training.trainers import EncoderDecoderTrainer
from STDSED.data_processing import DrugBank, CYP450


if __name__ == "__main__":
    # init args
    args = ModelTrainingArgs().parse_args()
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    # read config file
    config_dict = read_config_file(args.config)
    time_stamp = datetime.now().strftime(r"%Y%m%d_%H%M%S")
    config = Config.from_dict(config_dict)
    # random seeds
    torch.manual_seed(config["random_seed"])
    random.seed(config["random_seed"])
    # init encoder datasets
    drugbank = DrugBank(
        root="data", transform=AtomFeaturesOneHotTransformer(num_formal_charge=9)
    )
    cyp450 = CYP450(
        root="data", transform=AtomFeaturesOneHotTransformer(num_formal_charge=9)
    )
    datasets = [drugbank, cyp450]
    dloader = DataSplitter(
        datasets, batch_size=config["batch_size"], ratio=[0.9, 0.1, 0.0], shuffle=True
    )
    # make log dir
    log_name = "_".join(
        [
            "drugbank_cyp450" "embed{}".format(config["embedding_dim"]),
            "hiddenunits{}".format(config["hidden_units"][0]),
            str(config["batch_size"]),
            str(config["learning_rate"]),
            config["aggregation"],
        ]
    )
    log_dir = osp.join("logs", time_stamp, log_name)
    os.makedirs(log_dir)
    with open(osp.join(log_dir, "configs.yml"), "w") as f:
        f.write(yaml.dump(config_dict))
    # init encoder and decoder model
    dim_encoder_target = config["embedding_dim"]
    dim_decoder_target = drugbank[0].y.size(1)
    dim_features = drugbank[0].x.size(1)
    dropout = config["dropout"]
    Encoder = config["model"]
    encoder = Encoder(
        dim_features=dim_features, dim_target=dim_encoder_target, config=config
    )
    decoder = GINDecoder(dim_encoder_target, dim_decoder_target, dropout)
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
