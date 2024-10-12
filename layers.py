import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(1e-2 * torch.randn(
            in_features, out_features, dtype=torch.float64))
        if bias:
            self.bias = Parameter(1e-2 * torch.randn(
                1, out_features, dtype=torch.float64))
        else:
            self.register_parameter('bias', None)
        # self.reset_parameters()

    # def reset_parameters(self):
    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.mm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + f"({self.in_features} -> {self.out_features})"


class GraphPooling(nn.Module):
    def __init__(self, dim_graph_embedding, dim_hidden, dropout=0.2):
        super(GraphPooling, self).__init__()
        self.dropout = dropout
        self.dim_hidden = dim_hidden
        self.dim_graph_embedding = dim_graph_embedding
        self.weight1 = Parameter(1e-2 * torch.randn(
            self.dim_graph_embedding, self.dim_hidden, dtype=torch.float64))
        self.bias1 = Parameter(1e-2 * torch.randn(
            1, self.dim_hidden, dtype=torch.float64))
        self.weight2 = Parameter(1e-2 * torch.randn(
            self.dim_hidden, 2, dtype=torch.float64))
        self.bias2 = Parameter(1e-2 * torch.randn(1, 2, dtype=torch.float64))

    def forward(self, node_embedding, pooling="mean"):
        if pooling == "mean":
            self.graph_embedding = torch.mean(
                node_embedding, keepdim=True, dim=0)
        elif pooling == "max":
            self.graph_embedding = torch.max(
                node_embedding, keepdim=True, dim=0)[0]
        elif pooling == "sum":
            self.graph_embedding = torch.sum(
                node_embedding, keepdim=True, dim=0)
        else:
            raise ValueError(
                f"not support this pooling method {pooling}, check the pooling parameter")

        output = torch.mm(self.graph_embedding, self.weight1) + self.bias1
        output = F.relu(output)
        output = F.dropout(output, p=self.dropout, training=self.training)
        output = torch.mm(output, self.weight2) + self.bias2
        # return log_softmax and softmax
        return F.log_softmax(output, dim=1), F.softmax(output, dim=1)

    def __repr__(self):
        return self.__class__.__name__ + f"({self.dim_graph_embedding} -> {self.dim_hidden} -> 2)"
