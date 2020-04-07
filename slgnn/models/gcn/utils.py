import numpy as np
import scipy.sparse as sp
import torch

from slgnn.data_processing.zinc_to_hdf5 import Hdf5Loader
from PyFingerprint.All_Fingerprint import get_fingerprint


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i:] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def normalize(mx):
    """Row normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_spare_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def load_encoder_data(path):
    """ Load the data for training autoencoder
    """
    loader = Hdf5Loader(path)
    train, valid = dict(), dict()
    sep = int(loader.total * 0.9)
    features = loader.load_atom_features()
    train["features"] = features[:sep]
    valid["features"] = features[sep:]
    adjs = loader.load_adjacency_matrices()
    train["adj"] = adjs[:sep]
    valid["adj"] = adjs[sep:]
    smiles = loader.load_smiles()
    pubchem_fps = list()
    for sm in smiles:
        pubchem_fps.append(
            get_fingerprint(sm, fp_type='pubchem', output="vector"))
    pubchem_fps = np.stack(pubchem_fps)
    train["labels"] = pubchem_fps[:sep]
    valid["labels"] = pubchem_fps[sep:]
    return train, valid


def load_classifier_data(path):
    """
    """
