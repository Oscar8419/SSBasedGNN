import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution, GraphPooling


class GCN(nn.Module):
    def __init__(self, num_feature, num_hidden, num_class, dropout=0.2, pooling="mean"):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(num_feature, num_hidden)
        self.gc2 = GraphConvolution(num_hidden, num_class)
        self.dropout = dropout
        self.pooling = pooling
        self.gp = GraphPooling(num_class, 8, dropout=self.dropout)

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gc2(x, adj)
        output, output_softmax = self.gp(x, pooling=self.pooling)
        # return log_softmax and softmax
        return output, output_softmax
