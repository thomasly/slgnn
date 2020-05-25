""" A easy script to run GIN model.
"""
import os
import os.path as osp
from datetime import datetime
from random import shuffle
import random

from torch_geometric.data import DataLoader
from torch.utils.data import ConcatDataset
import torch
import yaml

from slgnn.configs.base import Grid, Config
from slgnn.configs.arg_parsers import ModelTrainingArgs
from slgnn.models.gcn.model import GIN
from slgnn.models.decoder.model import GINDecoder
from slgnn.data_processing.loaders import DataSplitter
from .trainers import EncoderDecoderTrainer, EncoderClassifierTrainer


def load_data(dataset, batch_size, shuffle_=True):
    if isinstance(dataset, list):
        return load_data_from_list(dataset, batch_size, shuffle_)
    else:
        if shuffle_:
            indices = list(range(len(dataset)))
            shuffle(indices)
            dataset = dataset[indices]
        sep = int(len(dataset) * 0.9)
        train_loader = DataLoader(dataset[:sep], batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(dataset[sep:], batch_size=batch_size, shuffle=False)
        return train_loader, val_loader


def load_data_from_list(datasets: list, batch_size, shuffle_=True):
    if shuffle_:
        for i, dataset in enumerate(datasets):
            indices = list(range(len(dataset)))
            shuffle(indices)
            datasets[i] = dataset[indices]
    train_l, val_l = list(), list()
    for dataset in datasets:
        sep = int(len(dataset) * 0.9)
        train_l.append(dataset[:sep])
        val_l.append(dataset[sep:])
    train_loader = DataLoader(
        ConcatDataset(train_l), batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(ConcatDataset(val_l), batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


if __name__ == "__main__":
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
        if config["encoder_epochs"] > 0:
            decoder = GINDecoder(dim_encoder_target, dim_decoder_target, dropout)

            dloader = DataSplitter(
                datasets, ratio=[0.9, 0.1, 0], batch_size=config["batch_size"]
            )
            encoder_trainer = EncoderDecoderTrainer(
                config, encoder, decoder, dloader.train_loader, dloader.val_loader
            )
            encoder_trainer.train()
            encoder_trainer.log_results(out=log_dir)
            encoder_trainer.plot_training_metrics(log_dir)
            for index in range(5):
                encoder_trainer.plot_reconstructions(
                    index, log_dir, f"reconstruction_{index}.png"
                )

        cls_dataset = config["classifier_dataset"]()
        classifier = GINDecoder(dim_encoder_target, cls_dataset.num_classes, dropout)
        cls_dloader = DataSplitter(
            cls_dataset, ratio=config["data_ratio"], batch_size=config["batch_size"]
        )
        cls_trainer = EncoderClassifierTrainer(
            config,
            encoder,
            classifier,
            cls_dloader.train_loader,
            cls_dloader.val_loader,
        )
        cls_trainer.train()
        cls_trainer.plot_training_metrics(log_dir)
