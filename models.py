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


class Confusion_Matrix:
    def __init__(self) -> None:
        super(Confusion_Matrix, self).__init__()
        self.TP = 0
        self.TN = 0
        self.FP = 0
        self.FN = 0

    def Pfa(self):
        if self.FP + self.TN == 0:
            return 0
        return round((self.FP) / (self.FP + self.TN), 2)

    def Pd(self):
        if self.TP + self.FN == 0:
            return 0
        return round((self.TP) / (self.TP + self.FN), 2)

    def clear(self):
        self.TP = 0
        self.TN = 0
        self.FP = 0
        self.FN = 0


class CNN_Net(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(CNN_Net, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=(3, 2), stride=(2, 1), padding=(1, 1)),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Conv2d(4, 8, kernel_size=(3, 2), stride=(2, 1), padding=(1, 1)),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=(3, 2),
                      stride=(2, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d((3, 2)),
            nn.Flatten(),
            nn.Linear(1344, 512),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(32, 2),
            # nn.Softmax(dim=1)
        )

    def forward(self, X):
        # print("in forward:", X.shape)
        output = self.layers(X)

        return output


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(start_dim=-2, end_dim=-1),
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            # nn.Dropout(p=0.3),
            nn.ReLU(),
            nn.Linear(16, 2),
            # nn.Softmax(dim=1)
        )

    def forward(self, X):
        output = self.layers(X)

        return output


class Squeeze(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, input):
        return input.squeeze(self.dim)


class LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            # nn.Linear(2, 1), nn.Flatten(start_dim=-2, end_dim=-1),nn.BatchNorm1d(1024),
            nn.LSTM(2, 8, batch_first=True, num_layers=2))
        self.mlp = nn.Linear(8, 2)

    def forward(self, input):
        lstm_out, _ = self.layers(input)
        lstm_out = lstm_out[:, -1, :]
        output = self.mlp(lstm_out)
        return output
