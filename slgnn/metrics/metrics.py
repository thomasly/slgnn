import torch
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score


def _to_numpy(*tensors):
    return [t.cpu().detach() for t in tensors]


class Accuracy:
    def __call__(self, pred, target):
        pred, target = _to_numpy(pred, target)
        if target.size()[1] == 1:
            _, pred = pred.max(dim=1)
        else:
            pred = torch.round(torch.sigmoid(pred))
        return accuracy_score(target, pred)

    @property
    def name(self):
        return "accuracy"


class ROC_AUC:
    def __init__(self):
        self._last = 0

    def __call__(self, pred, target):
        if target.size()[1] == 1:
            pred = torch.log_softmax(pred)
        else:
            pred = torch.sigmoid(pred)
        pred, target = _to_numpy(pred, target)
        try:
            score = roc_auc_score(target, pred)
            self._last = score
        except ValueError:
            score = self._last
        return score

    @property
    def name(self):
        return "roc_auc"


class F1:
    def __call__(self, pred, target):
        if target.size()[1] == 1:
            _, pred = pred.max(dim=1)
        else:
            pred = torch.round(torch.sigmoid(pred))
        pred, target = _to_numpy(pred, target)
        return f1_score(target, pred, average="micro")

    @property
    def name(self):
        return "f1"
