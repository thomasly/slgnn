import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

from slgnn.data_processing.zinc_to_hdf5 import Hdf5Loader
from PyFingerprint.All_Fingerprint import get_fingerprint
from chemreader.readers.readsmiles import Smiles
from slgnn.config import PAD_ATOM, PAD_BOND


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i:] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
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
    for i, mx in enumerate(matrices):
        rowsum = np.array(mx.sum(0))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = np.diag(r_inv)
        mx = mx.dot(r_mat_inv)
        matrices[i] = mx
    return matrices


def accuracy(output, labels):
    if any([s > 1 for s in labels.shape]):
        s = output.round() + labels
        numerator = (s == 2).sum().double() + (s == 0).sum().double()
        denominator = (s == 1).sum()
        # print("output + labels: {}".format(s),
        #       "numerator: {}".format(numerator.item()),
        #       "denominator: {}".format(denominator.item()))
        if denominator == 0:
            return 0.
        return (numerator / denominator).item()
    else:
        return (output.round() == labels).sum().item() / output.numel()


def sparse_mx_to_torch_spare_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def get_filtered_fingerprint(smiles):
    """ Get PubChem fingerprint wiht elements other than C, H, O, N, S, F, Cl,
    Br.

    Args:
        smiles (str): SMILES string.

    Return:
        fp (np.ndarray): The filtered PubChem fingerprint as a vector.
        length (int): length of the filtered vector.
    """
    fp = get_fingerprint(smiles, fp_type='pubchem', output="vector")
    del_pos = [26, 27, 28, 29, 30, 31, 32, 41, 42, 46, 47, 48, 295, 296, 298,
               303, 304, 348, 354, 369, 407, 411, 415, 456, 525, 627] + \
        list(range(49, 115)) + list(range(263, 283)) + \
        list(range(288, 293)) + list(range(310, 317)) + \
        list(range(318, 327)) + list(range(327, 332)) + \
        list(range(424, 427))
    fp = np.delete(fp, del_pos)
    return fp


def load_encoder_data(path):
    """ Load the data for training autoencoder
    """
    loader = Hdf5Loader(path)
    train, valid = dict(), dict()
    sep = int(loader.total * 0.9)
    features = loader.load_atom_features()

    def atom_feat_to_oh(features):
        arr = np.zeros((features.shape[0], features.shape[1], 56))
        for i, feat in enumerate(features):
            for j, f in enumerate(feat):
                arr[i, j, int(f[3])] = 1
                arr[i, j, -3:] = f[-3:]
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


def load_classifier_data(path,
                         smiles_col="smiles",
                         label_cols=[],
                         training_ratio=0.7,
                         testing_ratio=None):
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
            raise ValueError("The sum of training_ratio and testing_ratio"
                             " should be less than 1.")
    if testing_ratio is None:
        testing_ratio = 1. - training_ratio
    if training_ratio >= 0.9 or training_ratio <= 0.1:
        raise ValueError(
            "training_ratio should be a float in range (0.1, 0.9).")

    train, valid, test = dict(), dict(), dict()
    df = pd.read_csv(path)
    smiles = list(df[smiles_col].map(Smiles))
    random.shuffle(smiles)
    graphs = [s.to_graph(pad_atom=PAD_ATOM, pad_bond=PAD_BOND, sparse=True)
              for s in smiles if s.num_atoms < 71]

    sep_tr = int(len(graphs) * training_ratio)  # training
    sep_te = int(len(graphs) * testing_ratio)  # testing
    sep_tv = int(sep_tr * 0.9)  # train/valid

    def atom_feat_to_oh(features):
        arr = np.zeros((len(features), 56))
        for i, feat in enumerate(features):
            arr[i, features[i][0]] = 1
            arr[i, -3:] = features[i][-3:]
        return arr
    # feat = np.stack([g["atom_features"] for g in graphs])
    feat = np.stack([atom_feat_to_oh(g["atom_features"])
                     for g in graphs])
    # feat = normalize(feat)
    train["features"] = torch.FloatTensor(feat[:sep_tv])
    valid["features"] = torch.FloatTensor(feat[sep_tv:sep_tr])
    test["features"] = torch.FloatTensor(feat[-sep_te:])

    adjs = [g["adjacency"] for g in graphs]
    train["adj"] = [sparse_mx_to_torch_spare_tensor(
        adj+sp.identity(adj.shape[0])) for adj in adjs[: sep_tv]]
    valid["adj"] = [sparse_mx_to_torch_spare_tensor(
        adj+sp.identity(adj.shape[0])) for adj in adjs[sep_tv: sep_tr]]
    test["adj"] = [sparse_mx_to_torch_spare_tensor(
        adj+sp.identity(adj.shape[0])) for adj in adjs[-sep_te:]]
    labels = df[label_cols].fillna(0).to_numpy()
    n_classes = labels.shape[1]
    train["labels"] = torch.FloatTensor(labels[:sep_tv])
    valid["labels"] = torch.FloatTensor(labels[sep_tv:sep_tr])
    test["labels"] = torch.FloatTensor(labels[-sep_te:])

    return train, valid, test, n_classes
