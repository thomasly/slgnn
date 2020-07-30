""" **Custom metrics classes.**
"""

from abc import ABC, abstractmethod
import warnings

import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss
import torch.nn.functional as F
from torch.autograd import Variable
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
    """ Base class of AUC scores.
    """

    def __init__(self):
        self._last = 0

    @abstractmethod
    def judger(self):
        """ The function used to calculate AUC. Must be implemented by subclasses.
        """
        pass

    def __call__(self, pred, target):
        """ Calculate the AUC score.

        Args:
            pred: predicted probabilities.
            target: ground truthes.

        Returns:
            float: the calculated AUC score. If the score is not calculatable (all of
                targets belongs to the same class), return the last available score.
        """
        is_valid = ~(target == -1).cpu().detach()
        if _one_class(target):
            pred = torch.softmax(pred, 1)[:, 1]
        else:
            pred = torch.sigmoid(pred)
        pred, target = _to_numpy(pred, target)
        with warnings.catch_warnings():
            warnings.filterwarnings("error")
            try:
                score = self.judger(target[is_valid], pred[is_valid])
                self._last = score
            except ValueError:
                score = self._last
            except RuntimeWarning:
                score = self._last
        return score


class Accuracy:
    """ Calculate accuracy.

    Attributes:
        name: "Acc"
    """

    def __call__(self, pred, target):
        """
        Args:
            pred:
            target:
        """
        is_valid = ~(target == -1).cpu().detach()
        pred, target = _to_numpy(pred, target)
        if _one_class(target):
            _, pred = pred.max(dim=1)
        else:
            pred = torch.round(torch.sigmoid(pred))
        return accuracy_score(target[is_valid], pred[is_valid])

    @property
    def name(self):
        return "Acc"


class ROC_AUC(AUC):
    """ Calculate ROC_AUC score

    Attributes:
        name: "ROC".
    """

    @property
    def judger(self):
        """ roc_auc_score
        """
        return roc_auc_score

    @property
    def name(self):
        return "ROC"


class F1:
    """ Calculate F1 score.

    Attributes:
        name: "F1".
    """

    def __call__(self, pred, target):
        """ Calculate F1 with pred and target.

        Args:
            pred:
            target:
        """
        is_valid = ~(target == -1).cpu().detach()
        if _one_class(target):
            _, pred = pred.max(dim=1)
        else:
            pred = torch.round(torch.sigmoid(pred))
        pred, target = _to_numpy(pred, target)
        return f1_score(target[is_valid], pred[is_valid], average="micro")

    @property
    def name(self):
        return "F1"


class AP(AUC):
    """ Calculate average precision (AP) score

    Attributes:
        name: "AP".
    """

    @property
    def judger(self):
        """ average_precision_score
        """
        return average_precision_score

    @property
    def name(self):
        return "AP"


class FocalLoss(nn.Module):
    """
    | Calculate FocalLoss.
    | https://github.com/clcarwin/focal_loss_pytorch/blob/master/focalloss.py

    .. math::

        FL(p_t) = -(1-p_t)^\gamma log(p_t)

    Args:
        gamma (float): a float scalar larger than 0. The loss revert to cross entropy if
            gamma equals to 0.
        alpha (float): a float scalar between 0 and 1. Controls the contribution of
            possitive and negative samples. It should be set to a value less than 0.5
            when the dataset contains more negative samples than positive ones.
            Otherwise, it should be set to a value larger than 0.5.
        size_average (bool): average the losses if True. Sum otherwise.
    """

    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        """ Calculate focal loss. If your target have missing values, just fill the
        missing values with -1. This loss will apply a mask to the missing labels and
        calculate the loss with unmasked data.

        Args:
            input (torch.tensor): shape (N, C) where C = number of classes.
            target (torch.tensor): shape(N) where each value is
                :math:`0 \leq \text{targets}[i] \leq C-1`
        """
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)
        target = target.to(torch.long)
        is_valid = ~(target == -1).squeeze().cpu().detach()
        target[target == -1] = 0

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1 - pt[is_valid]) ** self.gamma * logpt[is_valid]
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


class MaskedBCEWithLogitsLoss(BCEWithLogitsLoss):
    __doc__ = f""" Maksed binary cross entropy loss. Filter out the effects of missing
    labels. In practise, missing labels should be filled with a number -1.

    {BCEWithLogitsLoss.__doc__}
    """

    def __init__(
        self, weight=None, reduction="mean", pos_weight=None,
    ):
        super().__init__(weight=weight, reduction=reduction, pos_weight=pos_weight)

    def forward(self, pred, target):
        mask = target == -1
        unmasked_loss = F.binary_cross_entropy_with_logits(
            pred, target, self.weight, pos_weight=self.pos_weight, reduction="none"
        )
        pad = (
            torch.zeros(unmasked_loss.shape)
            .to(unmasked_loss.device)
            .to(unmasked_loss.dtype)
        )
        masked_loss = torch.where(mask, pad, unmasked_loss)
        if self.reduction == "mean":
            loss = torch.sum(masked_loss) / torch.sum(~mask)
        else:  # "sum"
            loss = torch.sum(masked_loss)
        return loss
