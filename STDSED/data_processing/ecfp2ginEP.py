import os
import json

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


if __name__ == "__main__":
    # constants
    DIM_FEATURES = 52
    NUM_FORMAL_CHARGE = 9
    GPU = "cuda:0"
    CPU = "cpu"
    CONFIG_PATH = os.path.join("model_configs", "GIN_pretrain_drugbank_cyp450.yml")
    DATA_PATH = os.path.join("data", "fromraw_cid_inchi_smiles_fp_labels.csv")
    SAVED_MODEL = os.path.join(
        "logs",
        "20201011_140529",
        "drugbank_cyp450embed300_hiddenunits512_32_0.001_mean",
        "model_31.pt",
    )
    OUTPUT_PATH = os.path.join("data", "cyp450_smiles_GINfp_labels.json")

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
    # data file and transformer
    cypdf = pd.read_csv(DATA_PATH)
    transformer = AtomFeaturesOneHotTransformer(num_formal_charge=NUM_FORMAL_CHARGE)
    # create GIN fingerprints
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
            data = transformer(data)
            # get GIN fingerprint
            out = encoder(Batch.from_data_list([data]).to(GPU))
            outputs.append([smiles, out.to(CPU).squeeze().numpy().tolist(), label])
    # write smiles, GIN fingerprints, and labels to file
    with open(OUTPUT_PATH, "w") as outf:
        json.dump(outputs, outf)
