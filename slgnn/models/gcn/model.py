import copy

import torch
import torch.nn as nn
from torch.nn import BatchNorm1d
from torch.nn import Sequential, Linear, ReLU
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_add_pool, global_mean_pool

from slgnn.models.gcn.layers import GraphConvolution, CPANConv


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super().__init__()
        # self.gc1 = GraphConvolution(nfeat, nclass)
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x


class GIN(nn.Module):
    def __init__(self, dim_features, dim_target, config):
        super().__init__()

        self.config = config
        self.dropout = config["dropout"]
        self.embeddings_dim = [config["hidden_units"][0]] + config["hidden_units"]
        self.no_layers = len(self.embeddings_dim)
        self.first_h = []
        self.nns = []
        self.convs = []
        self.linears = []

        train_eps = config["train_eps"]
        if config["aggregation"] == "sum":
            self.pooling = global_add_pool
        elif config["aggregation"] == "mean":
            self.pooling = global_mean_pool

        for layer, out_emb_dim in enumerate(self.embeddings_dim):
            if layer == 0:
                self.first_h = Sequential(
                    Linear(dim_features, out_emb_dim),
                    BatchNorm1d(out_emb_dim),
                    ReLU(),
                    Linear(out_emb_dim, out_emb_dim),
                    BatchNorm1d(out_emb_dim),
                    ReLU(),
                )
                self.linears.append(Linear(out_emb_dim, dim_target))
            else:
                input_emb_dim = self.embeddings_dim[layer - 1]
                self.nns.append(
                    Sequential(
                        Linear(input_emb_dim, out_emb_dim),
                        BatchNorm1d(out_emb_dim),
                        ReLU(),
                        Linear(out_emb_dim, out_emb_dim),
                        BatchNorm1d(out_emb_dim),
                        ReLU(),
                    )
                )
                self.convs.append(GINConv(self.nns[-1], train_eps=train_eps))
                self.linears.append(Linear(out_emb_dim, dim_target))

        self.nns = torch.nn.ModuleList(self.nns)
        self.convs = torch.nn.ModuleList(self.convs)
        self.linears = torch.nn.ModuleList(self.linears)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        out = 0
        for layer in range(self.no_layers):
            if layer == 0:
                x = self.first_h(x)
                out += F.dropout(
                    self.pooling(self.linears[layer](x), batch), p=self.dropout
                )
            else:
                x = self.convs[layer - 1](x, edge_index)
                out += F.dropout(
                    self.linears[layer](self.pooling(x, batch)),
                    p=self.dropout,
                    training=self.training,
                )
        return out


class CPAN(nn.Module):
    def __init__(self, dim_features, dim_target, config, mod):
        super(CPAN, self).__init__()

        self.nhid = config["hidden_units"][0]
        self.alpha = config["alpha"]
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.intermediate_act_fn = nn.ReLU()
        self.readout = config["aggregation"]
        self.dropout = config["dropout"]
        self.concat_size = self.nhid * config["n_layer"] + dim_features

        conv_layer = CPANConv(config, mod)
        self.conv_layers = nn.ModuleList(
            [copy.deepcopy(conv_layer) for _ in range(config["n_layer"])]
        )

        self.fc = Linear(dim_features, self.nhid, bias=False)
        self.linears_prediction = torch.nn.ModuleList()
        for layer in range(config["n_layer"] + 1):
            if layer == 0:
                self.linears_prediction.append(nn.Linear(dim_features, self.nhid))
            else:
                self.linears_prediction.append(nn.Linear(self.nhid, self.nhid))

        self.bns_fc = torch.nn.ModuleList()
        for layer in range(config["n_layer"] + 1):
            if layer == 0:
                self.bns_fc.append(BatchNorm1d(dim_features))
            else:
                self.bns_fc.append(BatchNorm1d(self.nhid))

        self.lin_class = Linear(self.nhid, dim_target)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        if x.dim() == 1:
            x = x.unsqueeze(-1)

        if self.readout == "mean":
            output_list = [global_mean_pool(x, batch)]
        else:
            output_list = [global_add_pool(x, batch)]
        hid_x = self.fc(x)

        for conv in self.conv_layers:
            hid_x = conv(hid_x, edge_index)
            if self.readout == "mean":
                output_list.append(global_mean_pool(hid_x, batch))
            else:
                output_list.append(global_add_pool(hid_x, batch))

        score_over_layer = 0
        for layer, h in enumerate(output_list):
            h = self.bns_fc[layer](h)
            score_over_layer += F.relu(self.linears_prediction[layer](h))

        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin_class(score_over_layer)
        return x
