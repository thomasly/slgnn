import copy
from collections import defaultdict

import torch
import torch.nn as nn
from torch.nn import BatchNorm1d
from torch.nn import Sequential, Linear, ReLU, ELU
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_add_pool, global_mean_pool

from slgnn.models.gcn.layers import GraphConvolution, CPANConv


class GCN(nn.Module):
    """A simple GCN model as in https://arxiv.org/abs/1609.02907

    Args:
        nfeat (int): number of input features.
        nhid (int): number of nodes in hidden layers.
        nclass (int): number of target classes.
        dropout (float): dropout probability.
    """

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
    """The GIN model as in https://arxiv.org/abs/1810.00826

    Args:
        dim_features (int): dimension of features.
        dim_target (int): dimension of output.
        config: an instance of slgnn.configs.base.Config class. Must have dropout,
            hidden_units, train_eps, aggregation attributes.
    """

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
                # first hidden layer
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


class GIN_DISMAT(torch.nn.Module):
    def __init__(self, num_features, dim_node, dim_graph, config):
        """ GIN model from PyG examples. Output distance matrix.
        https://github.com/rusty1s/pytorch_geometric/blob/master/examples/mutag_gin.py
        """
        super().__init__()

        dim = config["hidden_units"]

        nn1 = Sequential(Linear(num_features, dim), ELU(), Linear(dim, dim))
        self.conv1 = GINConv(nn1)
        self.bn1 = torch.nn.BatchNorm1d(dim)

        nn2 = Sequential(Linear(dim, dim), ELU(), Linear(dim, dim))
        self.conv2 = GINConv(nn2)
        self.bn2 = torch.nn.BatchNorm1d(dim)

        nn3 = Sequential(Linear(dim, dim), ELU(), Linear(dim, dim))
        self.conv3 = GINConv(nn3)
        self.bn3 = torch.nn.BatchNorm1d(dim)

        nn4 = Sequential(Linear(dim, dim), ELU(), Linear(dim, dim))
        self.conv4 = GINConv(nn4)
        self.bn4 = torch.nn.BatchNorm1d(dim)

        nn5 = Sequential(Linear(dim, dim), ELU(), Linear(dim, dim_node))
        self.conv5 = GINConv(nn5)
        self.bn5 = torch.nn.BatchNorm1d(dim_node)

        self.fc1 = Linear(dim_node, dim_node)
        self.fc2 = Linear(dim_node, dim_graph)

    def forward(self, x, edge_index, batch):
        x = F.elu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = F.elu(self.conv2(x, edge_index))
        x = self.bn2(x)
        x = F.elu(self.conv3(x, edge_index))
        x = self.bn3(x)
        x = F.elu(self.conv4(x, edge_index))
        x = self.bn4(x)
        x = F.elu(self.conv5(x, edge_index))
        node_embeddings = self.bn5(x)
        pooled = global_add_pool(node_embeddings, batch)
        x = F.elu(self.fc1(pooled))
        x = F.dropout(x, p=0.25, training=self.training)
        x = self.fc2(x)
        #         print(f"Output from fc size: {x.size()}")
        x = torch.bmm(x[:, :, None], x[:, None, :])
        #         print(f"Size after batch mm: {x.size()}")
        #         x = torch.cat(
        #             [torch.mm(x[r][..., None], x[r][None, ...]) for r in range(x.size(0))],
        #             axis=0,
        #         )
        #         print(f"Pooled size: {pooled.size()}")
        #         print(f"Batch size: {x.size()}")
        #         print(f"expanded x size: {x.expand_as(pooled).size()}")
        #         output = torch.ones(x.size() + (pooled.size(-1), )).to(x.device)
        #         print(f"output size: {output.size()}")
        #         for i in range(x.size(0)):
        #             output[i] = pooled[i] * x[i, :, :, None]
        #         x = torch.mul(pooled, x)
        batch_size = x.size(0)
        feature_size = pooled.size(1)
        x_dim = x.size(1)
        pooled = pooled.repeat_interleave(x_dim * x_dim, dim=0).view(
            batch_size, x_dim, x_dim, -1
        )
        x = x.repeat_interleave(feature_size, dim=2).view(batch_size, x_dim, x_dim, -1)
        return pooled * x


class CPAN(nn.Module):
    """Shuo's CPAN model.

    Args:
        dim_features (int): dimension of features.
        dim_target (int): dimension of output.
        config: an instance of slgnn.configs.base.Config class. Must have alpha (leacky
        ReLU), dropout, hidden_units, train_eps, aggregation, n_layer attributes.
        mod (str): one of "origin", "additive", and "scaled".
    """

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
