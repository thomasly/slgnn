import torch
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score


def _to_numpy(*tensors):
    return [t.cpu() for t in tensors]


class Accuracy:
    def __call__(self, pred, target):
        pred, target = _to_numpy(pred, target)
        _, pred = pred.max(dim=1)
        return accuracy_score(target, pred)

    @property
    def name(self):
        return "accuracy"


class ROC_AUC:
    def __call__(self, pred, target):
        pred = torch.sigmoid(pred)
        pred, target = _to_numpy(pred, target)
        return roc_auc_score(target, pred)

    @property
    def name(self):
        return "roc_auc"


class F1:
    def __call__(self, pred, target):
        pred = torch.sigmoid(pred)
        pred = torch.round(pred)
        pred, target = _to_numpy(pred, target)
        return f1_score(target, pred)

    @property
    def name(self):
        return "f1"
