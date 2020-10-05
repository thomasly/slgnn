""" A easy script to run DiffPool with GraphSAGE model.
"""
import os
import os.path as osp
from datetime import datetime
import random
import logging

import torch
import yaml
from torch_geometric.data import DenseDataLoader

from slgnn.configs.base import Grid, Config
from slgnn.configs.arg_parsers import ModelTrainingArgs
from slgnn.models.decoder.model import GINDecoder
from slgnn.data_processing.deepchem_datasets import ClinTox, ClinToxFP
from slgnn.data_processing.deepchem_datasets import BACE, BACEFP
from slgnn.data_processing.deepchem_datasets import BBBP, BBBPFP
from slgnn.data_processing.deepchem_datasets import HIV, HIVFP
from slgnn.data_processing.deepchem_datasets import Sider, SiderFP
from slgnn.data_processing.pyg_datasets import JAK1, JAK1FP, JAK2, JAK2FP, JAK3, JAK3FP
from slgnn.data_processing.utils import NumNodesFilter, MyToDense
from .trainers import EncoderDecoderTrainer, EncoderClassifierTrainer


SEED = 0
MAX_NODES = 150

if __name__ == "__main__":
    args = ModelTrainingArgs().parse_args()
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    time_stamp = datetime.now().strftime(r"%Y%m%d_%H%M%S")

    exp_datasets = {
        "Sider": [Sider, SiderFP],
        "JAK1": [JAK1, JAK1FP],
        "JAK2": [JAK2, JAK2FP],
        "JAK3": [JAK3, JAK3FP],
        "BBBP": [BBBP, BBBPFP],
        "BACE": [BACE, BACEFP],
        "Sider": [Sider, SiderFP],
        "HIV": [HIV, HIVFP],
        "ClinTox": [ClinTox, ClinToxFP],
    }
    random_seeds = [0, 5, 193, 84234, 839574]
    for ds_name, ds in exp_datasets.items():
        Dataset = ds[0]
        FPDataset = ds[1]
        config_grid = Grid(os.path.join("model_configs", ds_name + "_DiffPool.yml"))
        for config_idx, config_dict in enumerate(config_grid):
            for seed_idx, seed in enumerate(random_seeds):
                config = Config.from_dict(config_dict)
                torch.manual_seed(SEED)
                random.seed(SEED)
                dataset = FPDataset(transform=MyToDense(MAX_NODES))
                if config["encoder_epochs"] == 0:
                    ifencoder = "noencoder"
                else:
                    ifencoder = "encoder"
                log_name = "_".join(
                    [
                        "DiffPool",
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
                    log_name,
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
                    coarse_scale=config_dict["coarse_scale"],
                    max_nodes=MAX_NODES,
                )
                if config["encoder_epochs"] > 0:
                    decoder = GINDecoder(
                        dim_encoder_target, dim_decoder_target, dropout
                    )

                    dloader = config["encoder_data_splitter"](
                        dataset,
                        dataloader=DenseDataLoader,
                        batch_size=config_dict["batch_size"],
                    )
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

                cls_dataset = Dataset(transform=MyToDense(MAX_NODES))
                classifier = GINDecoder(
                    dim_encoder_target, cls_dataset.num_classes, dropout
                )
                cls_dloader = config["classifier_data_splitter"](
                    cls_dataset,
                    dataloader=DenseDataLoader,
                    batch_size=config_dict["batch_size"],
                )
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
