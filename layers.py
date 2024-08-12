import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


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
