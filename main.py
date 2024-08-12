import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import time
from utils import *
from models import GCN

Data_Path_1 = "./signal1/SNR_-19_Signal1_x"  # .mat file path
Data_Path_0 = "./signal0/SNR_-19_Signal0_x"
num_SU = 10


def train(adj_matrix, features, model, labels, epoch=100):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    for i in range(epoch):
        t_start = time.time()
        optimizer.zero_grad()
        output = model(features, adj_matrix)
        loss_train = F.nll_loss(output, labels)
        loss_train.backward()
        optimizer.step()
        if (i+1) % 10 == 0:
            print(
                f"epoch {i}, loss:{loss_train}, time_spend:{time.time() - t_start }")


def main():
    torch.manual_seed(2024)
    np.random.seed(2024)

    mydevice = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_0 = node_feature(Data_Path_0, num_SU)
    data_1 = node_feature(Data_Path_1, num_SU)

    label_1 = torch.ones(10, device=mydevice, dtype=torch.int64)
    label_0 = torch.zeros(10, device=mydevice, dtype=torch.int64)

    adj_matrix_0 = adj_matrix(data_0, num_SU, rho=1)  # shape=(num_SU, num_SU)
    adj_matrix_1 = adj_matrix(data_1, num_SU, rho=1)
    adj_matrix_0 = torch.tensor(
        adj_matrix_0, dtype=torch.float32, device=mydevice)
    adj_matrix_1 = torch.tensor(
        adj_matrix_1, dtype=torch.float32, device=mydevice)

    data_0 = torch.tensor(data_0, dtype=torch.float32, device=mydevice)
    data_1 = torch.tensor(data_1, dtype=torch.float32, device=mydevice)

    model = GCN(64, 32, 2, 0.2).to(device=mydevice)
    # optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    train(adj_matrix_1, data_1, model, label_1)


if __name__ == "__main__":
    main()
