from abc import ABC, abstractmethod
import warnings

import torch
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    average_precision_score,
)
import numpy

numpy.set_printoptions(threshold=100000)


def _to_numpy(*tensors):
    return [t.cpu().detach() for t in tensors]


def _one_class(target):
    if len(target.size()) == 1:
        return True
    if target.size()[1] == 1:
        return True
    return False


class AUC(ABC):
    def __init__(self):
        self._last = 0

    @abstractmethod
    def judger(self):
        pass

    def __call__(self, pred, target):
        if _one_class(target):
            pred = torch.softmax(pred, 1)[:, 1]
        else:
            pred = torch.sigmoid(pred)
        pred, target = _to_numpy(pred, target)
        with warnings.catch_warnings():
            warnings.filterwarnings("error")
            try:
                score = self.judger(target, pred)
                self._last = score
            except ValueError:
                score = 0
                print("Value Error.")
                print(f"Target: {target}")
                print(f"Pred: {pred}")
                # score = self._last
            except RuntimeWarning:
                # score = self._last
                score = 0
                print("Runtime Warning.")
        return score


class Accuracy:
    def __call__(self, pred, target):
        pred, target = _to_numpy(pred, target)
        if _one_class(target):
            _, pred = pred.max(dim=1)
        else:
            pred = torch.round(torch.sigmoid(pred))
        return accuracy_score(target, pred)

    @property
    def name(self):
        return "Acc"


class ROC_AUC(AUC):
    @property
    def judger(self):
        return roc_auc_score

    @property
    def name(self):
        return "ROC_AUC"


class F1:
    def __call__(self, pred, target):
        if _one_class(target):
            _, pred = pred.max(dim=1)
        else:
            pred = torch.round(torch.sigmoid(pred))
        pred, target = _to_numpy(pred, target)
        return f1_score(target, pred, average="micro")

    @property
    def name(self):
        return "F1"


class AP(AUC):
    @property
    def judger(self):
        return average_precision_score

    @property
    def name(self):
        return "AP_AUC"
