
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from utils import *
from models import CNN_Net, Confusion_Matrix, MLP
from torch.utils.data import DataLoader
from model_dataset import Signal_Dataset

confus_matrix = Confusion_Matrix()
THRESHOLD = 0.5
snr_target = 8
modulation_target = "16QAM"
signal_data = Signal_Dataset(
    snr_target=snr_target, train=True, modulation=modulation_target)
signal_test_data = Signal_Dataset(
    snr_target=snr_target, train=False, modulation=modulation_target)
num_SU = 4
signal_dataloader = DataLoader(signal_data, batch_size=num_SU)
signal_test_dataloader = DataLoader(signal_test_data, batch_size=num_SU)
learning_rate = 0.1
epochs = 3
mydevice = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
test_result = {}
test_raw_result = {}


def train(dataloader, model, optimizer):
    model.train()
    positive_result = []
    negative_result = []
    cnt = 0
    confus_matrix.clear()
    loss_total = 0
    snr_target = dataloader.dataset.snr_target
    batch_size = dataloader.batch_size
    target = torch.ones(
        size=(batch_size, ), dtype=torch.int64, device=mydevice)
    noise_target = torch.zeros(
        size=(batch_size, ), dtype=torch.int64, device=mydevice)
    for X, _ in dataloader:
        # positive
        X = X.to(mydevice)
        output_logits = model(X)
        output_softmax = F.softmax(output_logits, dim=1)
        TP = (output_softmax[:, 1] > THRESHOLD).sum().item()
        # TP = (output_softmax[..., 1] > output_softmax[..., 0]).sum().item()
        confus_matrix.TP += TP
        confus_matrix.FN += (batch_size - TP)
        positive_result.extend(output_softmax[:, 1].tolist())
        loss_train = F.cross_entropy(output_logits, target)
        loss_total += (loss_train*batch_size)
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()

        # negative
        noise = noise_feature(snr_target, num_SU=batch_size,
                              usingSNR=False, usingIQ=True)
        noise = torch.tensor(noise, dtype=torch.float32,
                             device=mydevice)
        noise_output_logits = model(noise)
        noise_output_softmax = F.softmax(noise_output_logits, dim=1)
        TN = (noise_output_softmax[:, 1] < THRESHOLD).sum().item()
        confus_matrix.TN += TN
        confus_matrix.FP += (batch_size - TN)
        negative_result.extend(noise_output_softmax[:, 1].tolist())
        noise_loss_train = F.cross_entropy(
            noise_output_logits, noise_target, reduction="mean")
        loss_total += (noise_loss_train*batch_size)
        optimizer.zero_grad()
        noise_loss_train.backward()
        optimizer.step()

    # result
    # train_raw_result = {"positive": [round(num, 3) for num in positive_result],
    #                     "negative": [round(num, 3) for num in negative_result]}
    # save_result(train_raw_result,  file='./data_result/train_raw_data.json')
    print(f"snr: {snr_target},  \tloss: {loss_total: .2f} Pd: {confus_matrix.Pd(): .2f}, Pfa: {confus_matrix.Pfa(): .2f},\
    TP: {confus_matrix.TP}, FN: {confus_matrix.FN}, TN: {confus_matrix.TN}, FP: {confus_matrix.FP}")


def test(dataloader, model):
    positive_result = []
    negative_result = []
    model.eval()
    confus_matrix.clear()
    snr_target = dataloader.dataset.snr_target
    batch_size = dataloader.batch_size
    target = torch.ones(
        size=(batch_size, ), dtype=torch.int64, device=mydevice)
    noise_target = torch.zeros(
        size=(batch_size, ), dtype=torch.int64, device=mydevice)
    loss_total = 0
    cnt = 0
    with torch.no_grad():
        for X, _ in dataloader:
            X = X.to(mydevice)
            output_logits = model(X)
            output_softmax = F.softmax(output_logits, dim=1)
            loss_batch = F.cross_entropy(output_logits, target)
            loss_total += (loss_batch * batch_size)
            TP = (output_softmax[:, 1] > THRESHOLD).sum().item()
            confus_matrix.TP += TP
            confus_matrix.FN += (batch_size - TP)
            positive_result.extend(output_softmax[:, 1].tolist())
            # negative
            noise = noise_feature(snr_target, num_SU=batch_size,
                                  usingSNR=False, usingIQ=True)
            noise = torch.tensor(noise, dtype=torch.float32,
                                 device=mydevice)
            noise_output_logits = model(noise)
            noise_output_softmax = F.softmax(noise_output_logits, dim=1)
            loss_batch = F.cross_entropy(noise_output_logits, noise_target)
            loss_total += (loss_batch * batch_size)
            TN = (noise_output_softmax[:, 1] < THRESHOLD).sum().item()
            confus_matrix.TN += TN
            confus_matrix.FP += (batch_size - TN)
            negative_result.extend(noise_output_softmax[:, 1].tolist())
    # result
    test_raw_result = {"positive": [round(num, 3) for num in positive_result],
                       "negative": [round(num, 3) for num in negative_result]}
    save_result(test_raw_result,  file='./data_result/test_raw_data.json')
    print("=======")
    print(f"snr: {snr_target},  \tloss: {loss_total: .2f}, Pd: {confus_matrix.Pd(): .2f}, Pfa: {confus_matrix.Pfa(): .2f},\
    TP: {confus_matrix.TP}, FN: {confus_matrix.FN}, TN: {confus_matrix.TN}, FP: {confus_matrix.FP}")
    print("=======")


def main():

    model = MLP().to(device=mydevice)
    # mlp_init(model)
    load_model(model, save_path="./model_param/model_MLP.pt")

    # optimizer = optim.Adam(
    #     model.parameters(), lr=learning_rate, weight_decay=5e-4)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, )
    for i in range(epochs):
        for snr in [-10]:  # range(-20, 12, 2):
            signal_data.update(snr)
            signal_dataloader = DataLoader(signal_data, batch_size=num_SU)
            train(signal_dataloader, model, optimizer)
            signal_test_data.update(snr)
            signal_test_dataloader = DataLoader(
                signal_test_data, batch_size=num_SU)
            test(signal_test_dataloader, model)

    # for snr in range(-20, 12, 2):
    # for snr in [-10]:
    #     signal_test_data.update(snr)
    #     signal_test_dataloader = DataLoader(
    #         signal_test_data, batch_size=num_SU)
    #     test(signal_test_dataloader, model)

    save_model(model, save_path="./model_param/model_MLP.pt")


if __name__ == "__main__":
    main()
