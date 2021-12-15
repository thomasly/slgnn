import os
import json
from argparse import ArgumentParser

import torch
import pandas as pd
from slgnn.models.gcn.model import GIN
from slgnn.configs.base import read_config_file, Config
from slgnn.data_processing.utils import AtomFeaturesOneHotTransformer
from torch_geometric.data import Data, Batch
from chemreader.readers import Smiles
from tqdm import tqdm


def smiles2graph(smiles):
    try:
        graph = Smiles(smiles).to_graph(sparse=True)
    except AttributeError:
        raise
    x = torch.tensor(graph["atom_features"], dtype=torch.float)
    edge_idx = graph["adjacency"].tocoo()
    edge_idx = torch.tensor([edge_idx.row, edge_idx.col], dtype=torch.long)
    return Data(x=x, edge_index=edge_idx)


def save_cyp450(datapath, encoder, transformer=None, device="cuda:0"):
    cypdf = pd.read_csv(datapath)
    outputs = list()
    with torch.no_grad():
        it = tqdm(
            zip(cypdf["isomeric_SMILES"], cypdf["Label"]),
            desc="Create GIN fingerprints: ",
            total=len(cypdf),
        )
        for smiles, label in it:
            # transform smiles to input data
            try:
                data = smiles2graph(smiles.strip())
            except AttributeError:
                continue
            if transformer is not None:
                data = transformer(data)
            # get GIN fingerprint
            out = encoder(Batch.from_data_list([data]).to(device))
            outputs.append([smiles, out.to("cpu").squeeze().numpy().tolist(), label])
    return outputs


def save_chembl(datapath, encoder, batch_size=2048, transformer=None, device="cuda:0"):
    chembldf = pd.read_csv(datapath)
    outputs = list()
    smiles_list = list()
    data_list = list()
    n = 0
    with torch.no_grad():
        it = tqdm(
            chembldf[" SMILES"],
            desc="Create GIN fingerprints: ",
            total=chembldf.shape[0],
        )
        for smiles in it:
            # transform smiles to input data
            try:
                data = smiles2graph(smiles.strip())
            except AttributeError:
                continue
            if transformer is not None:
                data = transformer(data)
            data_list.append(data)
            smiles_list.append(smiles)
            n += 1
            if n == batch_size:
                output = encoder(Batch.from_data_list(data_list).to(device))
                for smiles, out in zip(smiles_list, output):
                    outputs.append([smiles, out.to("cpu").squeeze().numpy().tolist()])
                n = 0
                smiles_list = []
                data_list = []
    return outputs


def save_tox21(datapath, encoder, transformer=None, device="cuda:0"):
    tox21df = pd.read_csv(datapath)
    outputs = list()
    with torch.no_grad():
        it = tqdm(
            zip(tox21df["smiles"], tox21df["Label"]),
            desc="Create GIN fingerprints: ",
            total=tox21df.shape[0],
        )
        for smiles, label in it:
            # transform smiles to input data
            try:
                data = smiles2graph(smiles.strip())
            except AttributeError:
                continue
            if transformer is not None:
                data = transformer(data)
            out = encoder(Batch.from_data_list([data]).to(device))
            outputs.append([smiles, out.to("cpu").squeeze().numpy().tolist(), label])
    return outputs


if __name__ == "__main__":
    # constants
    DIM_FEATURES = 52
    NUM_FORMAL_CHARGE = 9
    GPU = "cuda:0"
    CPU = "cpu"
    CONFIG_PATH = os.path.join("model_configs", "GIN_pretrain_drugbank_cyp450.yml")
    SAVED_MODEL = os.path.join(
        "logs",
        "20201011_140529",
        "drugbank_cyp450embed300_hiddenunits512_32_0.001_mean",
        "model_31.pt",
    )

    # OUTPUT_PATH = os.path.join("data", "cyp450_smiles_GINfp_labels.json")

    arg_parser = ArgumentParser()
    arg_parser.add_argument(
        "--dataset-name", type=str, help="Name of the dataset to convert."
    )
    arg_parser.add_argument(
        "--output-path", type=str, help="Path to save the output fingerprints."
    )
    args = arg_parser.parse_args()

    # init model
    config_dict = read_config_file(CONFIG_PATH)
    config = Config.from_dict(config_dict)
    dim_encoder_target = config["embedding_dim"]
    dim_features = DIM_FEATURES
    encoder = GIN(
        dim_features=dim_features, dim_target=dim_encoder_target, config=config
    ).to(GPU)
    state_dict = torch.load(SAVED_MODEL)
    encoder.load_state_dict(state_dict)
    encoder.eval()

    # Save the dataset to json
    transformer = AtomFeaturesOneHotTransformer(num_formal_charge=NUM_FORMAL_CHARGE)
    if args.dataset_name == "cyp450":
        data_path = os.path.join("data", "fromraw_cid_inchi_smiles_fp_labels.csv")
        outputs = save_cyp450(data_path, encoder, transformer=transformer)
    elif args.dataset_name == "chembl":
        data_path = os.path.join("data", "ChEMBL24_all_compounds.csv.gz")
        outputs = save_chembl(data_path, encoder, transformer=transformer)
    elif args.dataset_name == "tox21":
        data_path = os.path.join("data", "MolNet_ecfp", "tox21_ecfp.csv")
        outputs = save_tox21(data_path, encoder, transformer=transformer)

    # write smiles, GIN fingerprints, and labels to file
    with open(args.output_path, "w") as outf:
        json.dump(outputs, outf)
