from unittest import TestCase

import torch
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
from sklearn.metrics import roc_auc_score, average_precision_score

from slgnn.metrics.metrics import (
    MaskedBCEWithLogitsLoss,
    Accuracy,
    F1,
    ROC_AUC,
    AP,
    FocalLoss,
)


class TestMaskedLoss(TestCase):
    def test_masked_bce_loss(self):
        criterion = MaskedBCEWithLogitsLoss()
        pred = torch.tensor([-1.0, 0.9, 0.1])
        target = torch.tensor([0.0, 1.0, -1.0])
        masked_loss = criterion(pred, target).item()
        expected_loss = BCEWithLogitsLoss()(pred[:2], target[:2]).item()
        self.assertEqual(masked_loss, expected_loss)

    def test_masked_focal_loss(self):
        criterion = FocalLoss()
        pred = torch.tensor([[1.0, -2.0], [0.9, 2.0], [0.1, 0.1]])
        target = torch.tensor([0, 1, -1])
        masked_loss = criterion(pred, target).item()
        expected_loss = CrossEntropyLoss()(pred[:2], target[:2]).item()
        self.assertEqual(masked_loss, expected_loss)

        pred = torch.tensor([[1.0, -2.0], [0.9, 2.0], [0.1, 0.1]])
        target = torch.tensor([0, 1, 1])
        loss = criterion(pred, target).item()
        expected_loss = CrossEntropyLoss()(pred, target).item()
        self.assertEqual(loss, expected_loss)

        alpha = 0.3
        gamma = 1.5
        criterion = FocalLoss(alpha=alpha, gamma=gamma)
        pred = torch.tensor([[1.0, -2.0], [0.9, 2.0], [0.1, 0.1]])
        target = torch.tensor([0, 1, -1])
        masked_loss = criterion(pred, target).item()
        a = torch.log_softmax(pred, 1)[0, 0]
        a = alpha * (-((1 - a.exp()) ** gamma) * a)
        b = torch.log_softmax(pred, 1)[1, 1]
        b = (1 - alpha) * (-((1 - b.exp()) ** gamma) * b)
        expected_loss = ((a + b) / 2).item()
        self.assertAlmostEqual(masked_loss, expected_loss)

    def test_masked_acc(self):
        metric = Accuracy()
        pred = torch.tensor([[1.0, -1.0, 1.0, -0.1, 0.8]])
        target = torch.tensor([[-1, 0, 0, 0, 1]])
        masked_acc = metric(pred, target)
        self.assertEqual(masked_acc, 0.75)

        pred = torch.tensor([[1.0, -1.0, 1.0, -0.1, 0.8]])
        target = torch.tensor([[1, 0, 0, 0, 1]])
        masked_acc = metric(pred, target)
        self.assertEqual(masked_acc, 0.8)

        pred = torch.tensor([[1.0, -1.0, 1.0, -0.1, -0.8]])
        target = torch.tensor([[-1, 0, -1, 0, 1]])
        masked_acc = metric(pred, target)
        self.assertAlmostEqual(masked_acc, 0.6666, places=3)

    def test_masked_f1(self):
        metric = F1()
        pred = torch.tensor([[1.0, -1.0, 1.0, -0.1, 0.8]])
        target = torch.tensor([[-1, 0, 0, 0, 1]])
        masked_acc = metric(pred, target)
        self.assertAlmostEqual(masked_acc, 0.75)

        pred = torch.tensor([[1.0, -1.0, 1.0, -0.1, 0.8]])
        target = torch.tensor([[1, 0, 0, 0, 1]])
        masked_acc = metric(pred, target)
        self.assertAlmostEqual(masked_acc, 0.8)

        pred = torch.tensor([[1.0, -1.0, 1.0, -0.1, -0.8]])
        target = torch.tensor([[-1, 0, -1, 0, 1]])
        masked_acc = metric(pred, target)
        self.assertAlmostEqual(masked_acc, 0.6666, places=3)

    def test_masked_roc(self):
        metric = ROC_AUC()
        pred = torch.tensor([[1.0, -1.0, 1.0, -0.1, 0.8]])
        target = torch.tensor([[-1, 0, 0, 0, 1]])
        masked_acc = metric(pred, target)
        expected = roc_auc_score([0, 0, 0, 1], torch.sigmoid(pred[0, 1:]).detach())
        self.assertAlmostEqual(masked_acc, expected)

        pred = torch.tensor([[1.0, -1.0, 1.0, -0.1, 0.8]])
        target = torch.tensor([[1, 0, 0, 0, 1]])
        masked_acc = metric(pred, target)
        expected = roc_auc_score(target[0].detach(), torch.sigmoid(pred[0, :]).detach())
        self.assertAlmostEqual(masked_acc, expected)

        pred = torch.tensor([[1.0, -1.0, 1.0, -0.1, -0.8]])
        target = torch.tensor([[-1, 0, -1, 0, 1]])
        masked_acc = metric(pred, target)
        expected = roc_auc_score([0, 0, 1], torch.sigmoid(pred[0, [1, 3, 4]]).detach())
        self.assertAlmostEqual(masked_acc, expected, places=3)

    def test_masked_ap(self):
        metric = AP()
        pred = torch.tensor([[1.0, -1.0, 1.0, -0.1, 0.8]])
        target = torch.tensor([[-1, 0, 0, 0, 1]])
        masked_acc = metric(pred, target)
        expected = average_precision_score(
            [0, 0, 0, 1], torch.sigmoid(pred[0, 1:]).detach()
        )
        self.assertAlmostEqual(masked_acc, expected)

        pred = torch.tensor([[1.0, -1.0, 1.0, -0.1, 0.8]])
        target = torch.tensor([[1, 0, 0, 0, 1]])
        masked_acc = metric(pred, target)
        expected = average_precision_score(
            target[0].detach(), torch.sigmoid(pred[0, :]).detach()
        )
        self.assertAlmostEqual(masked_acc, expected)

        pred = torch.tensor([[1.0, -1.0, 1.0, -0.1, -0.8]])
        target = torch.tensor([[-1, 0, -1, 0, 1]])
        masked_acc = metric(pred, target)
        expected = average_precision_score(
            [0, 0, 1], torch.sigmoid(pred[0, [1, 3, 4]]).detach()
        )
        self.assertAlmostEqual(masked_acc, expected, places=3)
