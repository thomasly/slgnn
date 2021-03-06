from torch import nn
import torch.nn.functional as F

from slgnn.config import PAD_ATOM


class Decoder(nn.Module):
    """ A simple decoder with one fully connected layer.
    """

    def __init__(self, n_feat, n_hid, n_out, dropout):
        super().__init__()
        # self.conv1 = nn.Conv2d(1, n_hid, (1, n_feat), stride=1, padding=0)
        # self.conv2 = nn.Conv2d(1, n_out, (1, n_hid), stride=1, padding=0)
        self.dense = nn.Linear(n_feat * PAD_ATOM, n_out)
        # self.dense1 = nn.Linear(n_feat * PAD_ATOM, n_hid)
        # self.dense2 = nn.Linear(n_hid, n_out)
        self.dropout = dropout

    def forward(self, x):
        x = x[None, None, :, :]
        # x = F.relu(self.conv1(x))
        # x = F.dropout(x, self.dropout, training=self.training)
        # x.transpose_(1, 3)
        # x = F.relu(self.conv2(x))
        # x = F.dropout(x, self.dropout, training=self.training)
        x = x.flatten(start_dim=1)
        # x = F.relu(self.dense1(x))
        # x = F.dropout(x, self.dropout, training=self.training)
        x = self.dense(x)
        return x


class GINDecoder(nn.Module):
    """ The decoder for GIN model.

    Args:
        n_in (int): number of the input features.
        n_out (int): number of the ouput features.
        dropout (float): dropout probability (1 - keep_probability).
    """

    def __init__(self, n_in, n_out, dropout):
        super().__init__()
        if n_out > n_in:
            n_hidden = int(n_out / 2)
        else:
            n_hidden = int(n_in / 2)
        self.fc1 = nn.Linear(n_in, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_out)
        self.dropout = dropout

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.fc2(x)
        return x
