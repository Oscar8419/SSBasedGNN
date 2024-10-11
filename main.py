import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import time
from utils import *
from models import GCN, Confusion_Matrix
from torch.utils.data import DataLoader
from model_dataset import Signal_Dataset

confus_matrix = Confusion_Matrix()
THRESHOLD = 0.5

signal_data = Signal_Dataset(snr_target=0)
num_SU = 8
signal_dataloader = DataLoader(signal_data, batch_size=num_SU)
learning_rate = 1e-3
epochs = 10
mydevice = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(dataloader, model, optimizer):
    model.train()
    # optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    confus_matrix.clear()
    snr_target = signal_data.snr_target
    for Graph, (X, y) in enumerate(dataloader):
        A_hat = adj_matrix(X[:, :, 0], num_SU)
        A_hat = torch.from_numpy(A_hat).to(mydevice)
        # X = torch.tensor(X, device=mydevice)
        # print(X.dtype)
        # X.to(mydevice)
        output, output_softmax = model(X[:, :, 0].to(
            dtype=torch.double, device=mydevice), A_hat)
        loss_train = F.nll_loss(output, torch.tensor(
            [1], dtype=torch.int64, device=mydevice))
        loss_train.backward()
        optimizer.step()
        optimizer.zero_grad()
        if output_softmax[0, 1] > THRESHOLD:
            confus_matrix.TP += 1
        else:
            confus_matrix.FN += 1
        # H0 only sensing noise
        noise = noise_feature(snr_target, num_SU=num_SU)
        A_hat = adj_matrix(noise, num_SU)
        noise = torch.from_numpy(noise).to(mydevice)
        A_hat = torch.from_numpy(A_hat).to(mydevice)
        output, output_softmax = model(noise.to(
            dtype=torch.double, device=mydevice), A_hat)
        loss_train = F.nll_loss(output, torch.tensor(
            [0], dtype=torch.int64, device=mydevice))
        loss_train.backward()
        optimizer.step()
        optimizer.zero_grad()
        if output_softmax[0, 1] > THRESHOLD:
            confus_matrix.FP += 1
        else:
            confus_matrix.TN += 1

    print(
        f"SNR:{snr_target}, Pd:{confus_matrix.Pd():.2f}, Pfa:{confus_matrix.Pfa():.2f}")

    # # model.eval()


def main():
    torch.manual_seed(2024)
    np.random.seed(2024)

    # for X, y in signal_dataloader:
    #     print(X.shape, y.shape)
    #     A_hat = adj_matrix(X[:, :, 0], num_SU)
    #     print(A_hat.shape)
    #     break

    model = GCN(num_feature=1024, num_hidden=512, num_class=128,
                dropout=0.2, pooling="mean").to(device=mydevice)
    load_model(model)
    optimizer = optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=5e-4)
    for i in range(epochs):
        train(signal_dataloader, model, optimizer)
    save_model(model)
    # load_model(model)
    # optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    # train(adj_matrix_1, data_1, model, label_1)
    # train(Graph_dict, model)
    # save_model(model)


if __name__ == "__main__":
    main()
