import torch
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score


def _to_numpy(*tensors):
    return [t.cpu() for t in tensors]


def accuracy(pred, target):
    pred, target = _to_numpy(pred, target)
    _, pred = pred.max(dim=1)
    return accuracy_score(target, pred)


def roc_auc(pred, target):
    pred = torch.sigmoid(pred)
    pred, target = _to_numpy(pred, target)
    return roc_auc_score(target, pred)


def f1(pred, target):
    pred = torch.sigmoid(pred)
    pred = torch.round(pred)
    pred, target = _to_numpy(pred, target)
    return f1_score(target, pred)
