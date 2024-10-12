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
        self.gp = GraphPooling(num_class, dim_hidden=32, dropout=self.dropout)

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gc2(x, adj)
        output, output_softmax = self.gp(x, pooling=self.pooling)
        # return log_softmax and softmax
        return output, output_softmax


class Confusion_Matrix():
    def __init__(self) -> None:
        super(Confusion_Matrix, self).__init__()
        self.TP = 0
        self.TN = 0
        self.FP = 0
        self.FN = 0

    def Pfa(self):
        if self.FP + self.TN == 0:
            return 0
        return round((self.FP)/(self.FP + self.TN), 2)

    def Pd(self):
        if self.TP + self.FN == 0:
            return 0
        return round((self.TP)/(self.TP + self.FN), 2)

    def clear(self):
        self.TP = 0
        self.TN = 0
        self.FP = 0
        self.FN = 0
