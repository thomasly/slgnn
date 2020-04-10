import torch
from torch import nn
import torch.nn.functional as F

from slgnn.config import PAD_ATOM


class Decoder(nn.Module):

    def __init__(self, n_feat, n_hid, n_out, dropout):
        super().__init__()
        self.conv1 = nn.Conv2d(1, n_hid, (1, n_feat), stride=1, padding=0)
        self.conv2 = nn.Conv2d(1, n_out, (1, n_hid), stride=1, padding=0)
        self.dense = nn.Linear(n_out * PAD_ATOM, n_out)
        self.dropout = dropout

    def forward(self, x):
        x = x[None, None, :, :]
        x = F.relu(self.conv1(x))
        x = F.dropout(x, self.dropout, training=self.training)
        x.transpose_(1, 3)
        x = F.relu(self.conv2(x))
        x = F.dropout(x, self.dropout, training=self.training)
        x = x.flatten(start_dim=1)
        x = self.dense(x)
        return torch.sigmoid(x)
