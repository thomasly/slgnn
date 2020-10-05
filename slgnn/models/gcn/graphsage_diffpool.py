from math import ceil
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch_geometric.nn import DenseSAGEConv, dense_diff_pool


class MyFilter(object):
    def __call__(self, data, max_nodes=150):
        return data.num_nodes <= max_nodes


class GNN(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        normalize=False,
        add_loop=False,
        lin=True,
    ):
        super(GNN, self).__init__()

        self.add_loop = add_loop

        self.conv1 = DenseSAGEConv(in_channels, hidden_channels, normalize)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv2 = DenseSAGEConv(hidden_channels, hidden_channels, normalize)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv3 = DenseSAGEConv(hidden_channels, out_channels, normalize)
        self.bn3 = torch.nn.BatchNorm1d(out_channels)

        if lin is True:
            self.lin = torch.nn.Linear(2 * hidden_channels + out_channels, out_channels)
        else:
            self.lin = None

    def bn(self, i, x):
        batch_size, num_nodes, num_channels = x.size()

        x = x.view(-1, num_channels)
        x = getattr(self, "bn{}".format(i))(x)
        x = x.view(batch_size, num_nodes, num_channels)
        return x

    def forward(self, x, adj, mask=None):
        # batch_size, num_nodes, in_channels = x.size()
        x0 = x
        a = self.conv1(x0, adj, mask)
        b = F.relu(a)
        c = self.bn(1, b)
        x1 = c
        x2 = self.bn(2, F.relu(self.conv2(x1, adj, mask)))
        x3 = self.bn(3, F.relu(self.conv3(x2, adj, mask)))

        x = torch.cat([x1, x2, x3], dim=-1)

        if self.lin is not None:
            x = F.relu(self.lin(x))

        return x


class GraphSAGEDiffPool(torch.nn.Module):
    def __init__(self, dim_features, dim_target, coarse_scale=0.25, max_nodes=150):
        super(GraphSAGEDiffPool, self).__init__()
        num_nodes = ceil(coarse_scale * max_nodes)
        self.gnn1_pool = GNN(dim_features, 64, num_nodes, add_loop=True)
        self.gnn1_embed = GNN(dim_features, 64, 64, add_loop=True, lin=False)

        num_nodes = ceil(coarse_scale * num_nodes)
        self.gnn2_pool = GNN(3 * 64, 64, num_nodes)
        self.gnn2_embed = GNN(3 * 64, 64, 64, lin=False)

        self.gnn3_embed = GNN(3 * 64, 64, 64, lin=False)

        self.lin1 = torch.nn.Linear(3 * 64, 64)
        self.lin2 = torch.nn.Linear(64, dim_target)

    def forward(self, data):
        x = data.x
        adj = data.adj
        mask = data.mask

        s = self.gnn1_pool(x, adj, mask)
        x = self.gnn1_embed(x, adj, mask)

        x, adj, l1, e1 = dense_diff_pool(x, adj, s, mask)
        s = self.gnn2_pool(x, adj)
        x = self.gnn2_embed(x, adj)

        x, adj, l2, e2 = dense_diff_pool(x, adj, s)
        x = self.gnn3_embed(x, adj)

        x = x.mean(dim=1)
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1), l1 + l2, e1 + e2

    def get_model_parameters_by_layer(self):
        """Get model parameters by layer.

        Returns:
            dict: model parameters with layer names as keys and list of parameters as
                values.
        """
        params = defaultdict(list)
        for name, param in self.named_parameters():
            layer = name.split(".")[1]
            params[layer].append(param)
        return params
