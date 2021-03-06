import os
import pickle as pk

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from tqdm import tqdm

from slgnn.data_processing.zinc_to_hdf5 import Hdf5Loader
from chemreader.readers.readsmiles import Smiles
from slgnn.config import PAD_ATOM, PAD_BOND


def encode_onehot(labels, len_label=None):
    """ Convert labels to one-hot.

    Args:
        labels (list): the labels to be converted.
        len_label (int): optional. If is None, the max length of the labels will be
            inferred from the provided label list.
            
    Returns:
        numpy.ndarray: the converted one-hot labels.
    """
    classes = set(labels)
    if len_label is None:
        classes_dict = {c: np.identity(len(classes))[i] for i, c in enumerate(classes)}
    else:
        classes_dict = {c: np.identity(len_label)[i] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot


# def normalize(mx):
#     """Row normalize sparse matrix"""
#     rowsum = np.array(mx.sum(1))
#     r_inv = np.power(rowsum, -1).flatten()
#     r_inv[np.isinf(r_inv)] = 0.
#     print("r_inv: {}".format(r_inv))
#     r_mat_inv = np.diag(r_inv)
#     mx = r_mat_inv.dot(mx)
#     return mx


def normalize(matrices):
    """ Normalize matrices to make the summation of every row equal to 1.
    
    Args:
        matrices (list): list of matrices.
        
    Returns:
        list: normalized matrices.
    """
    for i, mx in enumerate(matrices):
        rowsum = np.array(mx.sum(0))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.0
        r_mat_inv = np.diag(r_inv)
        mx = mx.dot(r_mat_inv)
        matrices[i] = mx
    return matrices


def accuracy(output, labels):
    """ Calculate accuracy.
    
    Args:
        output: output of a model.
        labels: ground truth labels.
        
    Returns:
        float: accuracy score.
    """
    if any([s > 1 for s in labels.shape]):
        s = output.round() + labels
        numerator = (s == 2).sum().double() + (s == 0).sum().double()
        denominator = (s == 1).sum()
        # print("output + labels: {}".format(s),
        #       "numerator: {}".format(numerator.item()),
        #       "denominator: {}".format(denominator.item()))
        if denominator == 0:
            return 0.0
        return (numerator / denominator).item()
    else:
        return (output.round() == labels).sum().item() / output.numel()


def sparse_mx_to_torch_spare_tensor(sparse_mx):
    """ Covert numpy spare matrix to pytorch sparse tensor.

    Args:
        sparse_mx: the sparse matrix to be converted.
        
    Returns:
        torch.sparse.FloatTensor:
    """
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def get_filtered_fingerprint(smiles):
    """ Get filtered PubChem fingerprint. The digits related to elements other than C,
    H, O, N, S, F, Cl, and Br are discarded.

    Args:
        smiles (str): SMILES string.

    Return:
        fp (np.ndarray): The filtered PubChem fingerprint as a vector.
        length (int): length of the filtered vector.
    """
    from PyFingerprint.All_Fingerprint import get_fingerprint

    fp = get_fingerprint(smiles, fp_type="pubchem", output="vector")
    del_pos = (
        [
            26,
            27,
            28,
            29,
            30,
            31,
            32,
            41,
            42,
            46,
            47,
            48,
            295,
            296,
            298,
            303,
            304,
            348,
            354,
            369,
            407,
            411,
            415,
            456,
            525,
            627,
        ]
        + list(range(49, 115))
        + list(range(263, 283))
        + list(range(288, 293))
        + list(range(310, 317))
        + list(range(318, 327))
        + list(range(327, 332))
        + list(range(424, 427))
    )
    fp = np.delete(fp, del_pos)
    return fp


def load_encoder_data(path, type_="txt"):
    if type_ == "txt":
        return load_encoder_txt_data(path)
    elif type_ == "hdf5":
        return load_encoder_hdf5_data(path)


def load_encoder_hdf5_data(path):
    loader = Hdf5Loader(path)
    train, valid = dict(), dict()
    sep = int(loader.total * 0.9)
    features = loader.load_atom_features()

    def atom_feat_to_oh(features):
        arr = np.zeros((features.shape[0], features.shape[1], 59))
        for i, feat in enumerate(features):
            for j, f in enumerate(feat):
                arr[i, j, int(f[3])] = 1
                arr[i, j, -6:] = f[-6:]
        return arr

    features = atom_feat_to_oh(features)
    # features = normalize(features)
    train["features"] = torch.FloatTensor(features[:sep])
    valid["features"] = torch.FloatTensor(features[sep:])
    adjs = loader.load_adjacency_matrices()
    train["adj"], valid["adj"] = list(), list()
    for i, adj in enumerate(adjs):
        adj = adj + sp.identity(adj.shape[0])
        # adj = normalize(adj)
        adj = sparse_mx_to_torch_spare_tensor(adj)
        if i < sep:
            train["adj"].append(adj)
        else:
            valid["adj"].append(adj)
    smiles = loader.load_smiles()
    # pubchem_fps = [get_fingerprint(
    #     sm, fp_type='pubchem', output="vector") for sm in smiles]
    pubchem_fps = [get_filtered_fingerprint(sm) for sm in smiles]
    pubchem_fps = np.stack(pubchem_fps)
    train["labels"] = torch.FloatTensor(pubchem_fps[:sep])
    valid["labels"] = torch.FloatTensor(pubchem_fps[sep:])
    return train, valid


def feat_to_oh(features, col_num, label_len):
    """ Convert a specific column of the features to one-hot label.
    Args:
        features (list): list of atom features.
        col_num (int): the feature column number to convert to one-hot label.

    Return (numpy.ndarray):
        Converted feature with one-hot encoding
    """
    features = np.array(features)
    oh = encode_onehot(list(features[:, col_num]), label_len)
    oh_features = np.concatenate(
        [features[:, :col_num], oh, features[:, col_num + 1 :]], axis=1
    )
    return oh_features


def load_encoder_txt_data(path):
    import random

    random.seed(12391)
    saving_path = os.path.join(
        os.path.dirname(path), os.path.basename(path).split(".")[0] + "_fingerprints.pk"
    )
    graph_saving_path = os.path.join(
        os.path.dirname(path), os.path.basename(path).split(".")[0] + "_graphs.pk"
    )

    if os.path.isfile(graph_saving_path):
        with open(graph_saving_path, "rb") as f:
            graphs = pk.load(f)
    else:
        with open(path, "r") as f:
            smiles = [Smiles(line) for line in f.readlines()]
        random.shuffle(smiles)
        pbar = tqdm(smiles, ascii=True)
        pbar.set_description("Generating graphs ")
        graphs = [
            s.to_graph(pad_atom=PAD_ATOM, pad_bond=PAD_BOND, sparse=True)
            for s in pbar
            if s.num_atoms < PAD_ATOM + 1
        ]
        with open(graph_saving_path, "wb") as f:
            pk.dump(graphs, f)

    train, valid = dict(), dict()
    sep_tv = int(len(graphs) * 0.9)  # training/valid

    # feat = np.stack([g["atom_features"] for g in graphs])
    pbar = tqdm(graphs, ascii=True)
    pbar.set_description("Getting atom features ")
    feat = list()
    for g in pbar:
        features = g["atom_features"]
        features = feat_to_oh(features, 5, 5)
        features = feat_to_oh(features, 2, 7)
        features = feat_to_oh(features, 1, 11)
        features = feat_to_oh(features, 0, 53)
        feat.append(features)
    feat = np.stack(feat)
    train["features"] = torch.FloatTensor(feat[:sep_tv])
    valid["features"] = torch.FloatTensor(feat[sep_tv:])

    pbar = tqdm(graphs, ascii=True)
    pbar.set_description("Getting adjacency matrices ")
    adjs = [g["adjacency"] for g in pbar]
    train["adj"] = [
        sparse_mx_to_torch_spare_tensor(adj + sp.identity(adj.shape[0]))
        for adj in adjs[:sep_tv]
    ]
    valid["adj"] = [
        sparse_mx_to_torch_spare_tensor(adj + sp.identity(adj.shape[0]))
        for adj in adjs[sep_tv:]
    ]

    if os.path.isfile(saving_path):
        with open(saving_path, "rb") as f:
            pubchem_fps = pk.load(f)
    else:
        try:
            smiles
        except NameError:
            with open(path, "r") as f:
                smiles = [Smiles(line) for line in f.readlines()]
        pbar = tqdm(smiles, ascii=True)
        pbar.set_description("Generating PubChem Fingerprints ")
        pubchem_fps = [get_filtered_fingerprint(sm.smiles_str) for sm in pbar]
        pubchem_fps = np.stack(pubchem_fps)
        with open(saving_path, "wb") as f:
            pk.dump(pubchem_fps, f)
    len_fp = pubchem_fps.shape[1]
    train["labels"] = torch.FloatTensor(pubchem_fps[:sep_tv])
    valid["labels"] = torch.FloatTensor(pubchem_fps[sep_tv:])

    return train, valid, len_fp


def load_classifier_data(
    path, smiles_col="smiles", label_cols=[], training_ratio=0.7, testing_ratio=None
):
    """ Load classification task data.

    Args:
        path (str): path to the data file.
        training_ratio (float): training data ratio in the whole dataset. The
            ration between training and validate data is alwasy set to 0.9.
        testing_ratio (float): testing data ration in the whole datast. If
            None, the ration is default to 1 - training_ration.
    """
    import random

    random.seed(12391)

    if testing_ratio is not None:
        if testing_ratio + training_ratio > 1:
            raise ValueError(
                "The sum of training_ratio and testing_ratio" " should be less than 1."
            )
    if testing_ratio is None:
        testing_ratio = 1.0 - training_ratio
    if training_ratio >= 0.9 or training_ratio <= 0.1:
        raise ValueError("training_ratio should be a float in range (0.1, 0.9).")

    train, valid, test = dict(), dict(), dict()
    df = pd.read_csv(path)
    smiles = list(df[smiles_col].map(Smiles))
    random.shuffle(smiles)
    graphs = [
        s.to_graph(pad_atom=PAD_ATOM, pad_bond=PAD_BOND, sparse=True)
        for s in smiles
        if s.num_atoms < 71
    ]

    sep_tr = int(len(graphs) * training_ratio)  # training
    sep_te = int(len(graphs) * testing_ratio)  # testing
    sep_tv = int(sep_tr * 0.9)  # train/valid

    # def atom_feat_to_oh(features):
    #     arr = np.zeros((len(features), 79))
    #     for i, feat in enumerate(features):
    #         arr[i, features[i][0]] = 1
    #         arr[i, -6:] = features[i][-6:]
    #     return arr

    # feat = np.stack([g["atom_features"] for g in graphs])
    # feat = np.stack([atom_feat_to_oh(g["atom_features"]) for g in graphs])
    # feat = normalize(feat)
    pbar = tqdm(graphs, ascii=True)
    pbar.set_description("Getting atom features ")
    feat = list()
    for g in pbar:
        features = g["atom_features"]
        features = feat_to_oh(features, 5, 5)
        features = feat_to_oh(features, 2, 7)
        features = feat_to_oh(features, 1, 11)
        features = feat_to_oh(features, 0, 53)
        feat.append(features)
    feat = np.stack(feat)
    train["features"] = torch.FloatTensor(feat[:sep_tv])
    valid["features"] = torch.FloatTensor(feat[sep_tv:sep_tr])
    test["features"] = torch.FloatTensor(feat[-sep_te:])

    adjs = [g["adjacency"] for g in graphs]
    train["adj"] = [
        sparse_mx_to_torch_spare_tensor(adj + sp.identity(adj.shape[0]))
        for adj in adjs[:sep_tv]
    ]
    valid["adj"] = [
        sparse_mx_to_torch_spare_tensor(adj + sp.identity(adj.shape[0]))
        for adj in adjs[sep_tv:sep_tr]
    ]
    test["adj"] = [
        sparse_mx_to_torch_spare_tensor(adj + sp.identity(adj.shape[0]))
        for adj in adjs[-sep_te:]
    ]
    labels = df[label_cols].fillna(0).to_numpy()
    n_classes = labels.shape[1]
    train["labels"] = torch.FloatTensor(labels[:sep_tv])
    valid["labels"] = torch.FloatTensor(labels[sep_tv:sep_tr])
    test["labels"] = torch.FloatTensor(labels[-sep_te:])

    return train, valid, test, n_classes
