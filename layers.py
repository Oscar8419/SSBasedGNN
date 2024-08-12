import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.randn(
            in_features, out_features, dtype=torch.float32))
        if bias:
            self.bias = Parameter(torch.randn(
                out_features, dtype=torch.float32))
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
        self.weight1 = Parameter(torch.randn(
            self.dim_graph_embedding, self.dim_hidden, dtype=torch.float32))
        self.bias1 = Parameter(torch.randn(
            1, self.dim_hidden, dtype=torch.float32))
        self.weight2 = Parameter(torch.randn(
            self.dim_hidden, 2, dtype=torch.float32))
        self.bias2 = Parameter(torch.randn(1, 2, dtype=torch.float32))

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
