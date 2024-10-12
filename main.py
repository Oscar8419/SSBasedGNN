import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import json
from utils import *
from models import GCN, Confusion_Matrix
from torch.utils.data import DataLoader
from model_dataset import Signal_Dataset

confus_matrix = Confusion_Matrix()
THRESHOLD = 0.5
snr_target = -20
signal_data = Signal_Dataset(snr_target=snr_target, train=True)
signal_test_data = Signal_Dataset(snr_target=snr_target, train=False)
num_SU = 8
signal_dataloader = DataLoader(signal_data, batch_size=num_SU)
signal_test_dataloader = DataLoader(signal_test_data, batch_size=num_SU)
learning_rate = 1e-3
epochs = 5
mydevice = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
test_result = {}


def train(dataloader, model, optimizer):
    model.train()
    # optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    confus_matrix.clear()
    train_loss = 0
    num_Graph = len(dataloader)  # 409 Graphs
    snr_target = dataloader.dataset.snr_target
    for Graph, (X, y) in enumerate(dataloader):
        A_hat = adj_matrix(X[:, :, 0], num_SU)
        A_hat = torch.from_numpy(A_hat).to(mydevice)
        output, output_softmax = model(X[:, :, 0].to(
            dtype=torch.double, device=mydevice), A_hat)
        loss_train = F.nll_loss(output, torch.tensor(
            [1], dtype=torch.int64, device=mydevice))
        loss_train.backward()
        optimizer.step()
        optimizer.zero_grad()
        train_loss += loss_train.item()
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
        train_loss += loss_train.item()
        if output_softmax[0, 1] > THRESHOLD:
            confus_matrix.FP += 1
        else:
            confus_matrix.TN += 1

    train_loss /= num_Graph
    # print(
    # f"SNR:{snr_target},\t train loss:{train_loss:.2f},\t Pd:{confus_matrix.Pd()},\t Pfa:{confus_matrix.Pfa()}")


def test(dataloader, model, save_result=False):
    model.eval()
    test_loss = 0
    num_Graph = len(dataloader)  # 103 Graphs
    confus_matrix.clear()
    snr_target = dataloader.dataset.snr_target
    with torch.no_grad():
        for X, y in dataloader:
            A_hat = adj_matrix(X[:, :, 0], num_SU)
            A_hat = torch.from_numpy(A_hat).to(mydevice)
            output, output_softmax = model(X[:, :, 0].to(
                dtype=torch.double, device=mydevice), A_hat)
            loss = F.nll_loss(output, torch.tensor(
                [1], dtype=torch.int64, device=mydevice))
            test_loss += loss.item()
            if output_softmax[0, 1] > THRESHOLD:
                confus_matrix.TP += 1
            else:
                confus_matrix.FN += 1
            # H0
            noise = noise_feature(snr_target, num_SU=num_SU)
            A_hat = adj_matrix(noise, num_SU)
            noise = torch.from_numpy(noise).to(mydevice)
            A_hat = torch.from_numpy(A_hat).to(mydevice)
            output, output_softmax = model(noise.to(
                dtype=torch.double, device=mydevice), A_hat)
            loss = F.nll_loss(output, torch.tensor(
                [0], dtype=torch.int64, device=mydevice))
            test_loss += loss.item()
            if output_softmax[0, 1] > THRESHOLD:
                confus_matrix.FP += 1
            else:
                confus_matrix.TN += 1
    test_loss /= num_Graph
    print(
        f"SNR:{snr_target},\t test loss:{test_loss:.2f},\t Pd:{confus_matrix.Pd()},\t Pfa:{confus_matrix.Pfa()}")
    print("=========")
    if save_result:
        result_dict = {"loss": round(
            test_loss, 2), "Pd": confus_matrix.Pd(), "Pfa": confus_matrix.Pfa()}
        test_result[f"SNR_{snr_target}"] = result_dict


def main():
    torch.manual_seed(2024)
    np.random.seed(2024)

    model = GCN(num_feature=1024, num_hidden=512, num_class=128,
                dropout=0.2, pooling="mean").to(device=mydevice)
    print(model)
    load_model(model)
    optimizer = optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=5e-4)
    for i in range(epochs):
        for snr in range(-20, 32, 2):
            signal_data.update(snr)
            signal_dataloader = DataLoader(signal_data, batch_size=num_SU)
            train(signal_dataloader, model, optimizer)
            # test(signal_test_dataloader, model)
    for snr in range(-20, 32, 2):
        signal_test_data.update(snr)
        signal_test_dataloader = DataLoader(
            signal_test_data, batch_size=num_SU)
        test(signal_test_dataloader, model, save_result=True)

    save_result(test_result)
    save_model(model)


if __name__ == "__main__":
    main()
