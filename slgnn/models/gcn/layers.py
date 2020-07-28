import math

import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, BatchNorm1d
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
from torch_geometric.nn.inits import zeros
from torch_geometric.nn.conv import MessagePassing
from torch_scatter import scatter


class GraphConvolution(Module):
    """
    GCN layer as in https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        """ Initialize/reset parameter values with uniform distribution
        """
        # Get boundary for the uniform distribution
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        adj = adj.to_dense()
        degree_mat = torch.diag(torch.sum(adj, dim=1))
        lap_mat = degree_mat - adj
        normed_lap = torch.diag(torch.diag(torch.pow(lap_mat, -0.5)))
        normed_lap[normed_lap == float("inf")] = 0.0
        normed_adj = torch.mm(normed_lap, adj)
        normed_adj = torch.mm(normed_adj, normed_lap)
        output = torch.mm(normed_adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return (
            self.__class__.__name__
            + " ("
            + str(self.in_features)
            + " -> "
            + str(self.out_features)
            + ")"
        )


class CPANConv(MessagePassing):
    """ CPAN layers.
    
    Args:
        config: an instance of slgnn.config.base.Config class. Must have heads, alpha,
            dropout, hidden_units attributes.
        mod: "origin", "additive", or "scaled".
    """

    def __init__(self, config, mod):

        super(CPANConv, self).__init__()

        self.heads = config["heads"]
        self.negative_slope = config["alpha"]
        self.dropout = config["dropout"]
        self.mod = mod
        self.nhid = config["hidden_units"][0]
        self.in_channels = self.nhid
        self.out_channels = self.nhid

        self.bias = Parameter(torch.Tensor(self.heads * self.out_channels))
        self.mlp = Sequential(
            Linear(self.nhid, self.nhid),
            ReLU(),
            BatchNorm1d(self.nhid),
            Linear(self.nhid, self.nhid),
            ReLU(),
            BatchNorm1d(self.nhid),
            Linear(self.nhid, self.nhid),
        )

        self.fc_i = Linear(int(self.in_channels / self.heads), 1, bias=True)
        self.fc_j = Linear(int(self.in_channels / self.heads), 1, bias=True)

        self.reset_parameters()

    def reset_parameters(self):
        # glorot(self.att)
        zeros(self.bias)

    def forward(self, x, edge_index):
        x = x.view(-1, self.heads, int(self.in_channels / self.heads))

        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(self.node_dim))

        out = self.propagate(edge_index, x=x, size=x.size(self.node_dim))
        out = out.view(-1, self.heads * self.out_channels)

        out = self.mlp(out)

        return out

    def message(self, x_i, x_j, edge_index_i, size_i):
        # Compute attention coefficients.
        h_i = self.fc_i(x_i)
        h_j = self.fc_j(x_j)
        alpha = (h_i + h_j).squeeze(-1)
        # print('alpha1 size', alpha.size())
        # alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)
        # print('alpha2 size', alpha.size())
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index_i, size_i)

        # CPA models are 'additive' and 'scaled', the 'origin' is the original GAT-based
        # self-attention
        if self.mod == "origin":
            alpha = alpha
        elif self.mod == "additive":
            alpha = torch.where(alpha > 0, alpha + 1, alpha)
        elif self.mod == "scaled":
            ones = alpha.new_ones(edge_index_i.size())
            add_degree = scatter(ones, edge_index_i, dim_size=size_i, reduce=self.aggr)
            degree = add_degree[edge_index_i].unsqueeze(-1)
            alpha = alpha * degree
        else:
            print("Wrong mod! Use Origin")
            alpha = alpha

        # Sample attention coefficients stochastically.
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        # print('x_j size', x_j.size())
        # print('alpha size', alpha.size())
        return x_j * alpha.view(-1, self.heads, 1)

    def update(self, aggr_out):
        aggr_out = aggr_out.view(-1, self.out_channels)
        return aggr_out

    def __repr__(self):
        return "{}({}, {}, heads={})".format(
            self.__class__.__name__, self.in_channels, self.out_channels, self.heads
        )
