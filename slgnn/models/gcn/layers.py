import math

import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


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
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        adj = adj.to_dense()
        degree_mat = torch.diag(torch.sum(adj, dim=1))
        lap_mat = degree_mat - adj
        normed_lap = torch.diag(torch.diag(torch.pow(lap_mat, -0.5)))
        normed_lap[normed_lap == float("inf")] = 0.
        normed_adj = torch.mm(normed_lap, adj)
        normed_adj = torch.mm(normed_adj, normed_lap)
        output = torch.mm(normed_adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + " ("\
            + str(self.in_features) + " -> "\
            + str(self.out_features) + ")"
