import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from utils import *
from models import CNN_Net, Confusion_Matrix
from torch.utils.data import DataLoader
from model_dataset import Signal_Dataset

confus_matrix = Confusion_Matrix()
THRESHOLD = 0.5
snr_target = -20
modulation_target = "16QAM"
signal_data = Signal_Dataset(
    snr_target=snr_target, train=True, modulation=modulation_target, cnn=True)
signal_test_data = Signal_Dataset(
    snr_target=snr_target, train=False, modulation=modulation_target, cnn=True)
num_SU = 8
signal_dataloader = DataLoader(signal_data, batch_size=num_SU)
signal_test_dataloader = DataLoader(signal_test_data, batch_size=num_SU)
learning_rate = 5e-2
epochs = 5
mydevice = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
test_result = {}


def train(dataloader, model, optimizer):
    loss_total = 0
    model.train()
    confus_matrix.clear()
    snr_target = dataloader.dataset.snr_target
    batch_size = dataloader.batch_size
    target = torch.ones(
        size=(batch_size, ), dtype=torch.int64, device=mydevice)
    noise_target = torch.zeros(
        size=(batch_size, ), dtype=torch.int64, device=mydevice)
    for X, _ in dataloader:
        X = X.to(mydevice)
        # print("X: ", X.shape)
        output_logits = model(X)
        output_softmax = F.softmax(output_logits, dim=1)
        TP = (output_softmax[..., 1] > THRESHOLD).sum().item()
        confus_matrix.TP += TP
        confus_matrix.FN += (batch_size - TP)
        loss_train = F.cross_entropy(output_logits, target)
        loss_total += (loss_train*batch_size)
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()

        # negative
        noise = noise_feature(snr_target, num_SU=batch_size,
                              usingSNR=False, usingIQ=True)
        noise = np.expand_dims(noise, axis=1)
        noise = torch.tensor(noise, dtype=torch.float32,
                             device=mydevice)
        noise_output_logits = model(noise)
        noise_output_softmax = F.softmax(noise_output_logits, dim=1)
        TN = (noise_output_softmax[:, 1] < THRESHOLD).sum().item()
        confus_matrix.TN += TN
        confus_matrix.FP += (batch_size - TN)
        noise_loss_train = F.cross_entropy(
            noise_output_logits, noise_target, reduction="mean")
        loss_total += (noise_loss_train*batch_size)
        optimizer.zero_grad()
        noise_loss_train.backward()
        # grad_center(model)
        optimizer.step()

    # result
    print(f"snr: {snr_target},  \tloss: {loss_total: .2f} Pd: {confus_matrix.Pd(): .2f}, Pfa: {confus_matrix.Pfa(): .2f},\
    TP: {confus_matrix.TP}, FN: {confus_matrix.FN}, TN: {confus_matrix.TN}, FP: {confus_matrix.FP}")


def test(dataloader, model):
    model.eval()
    confus_matrix.clear()
    snr_target = dataloader.dataset.snr_target
    batch_size = dataloader.batch_size
    cnt = 0
    for X, _ in dataloader:
        X = X.to(mydevice)
        output_logits = model(X)
        output_softmax = F.softmax(output_logits, dim=1)
        TP = (output_softmax[:, 1] > THRESHOLD).sum().item()
        confus_matrix.TP += TP
        confus_matrix.FN += (batch_size - TP)
        # negative
        noise = noise_feature(snr_target, num_SU=batch_size,
                              usingSNR=False, usingIQ=True)
        noise = np.expand_dims(noise, axis=1)
        noise = torch.tensor(noise, dtype=torch.float32,
                             device=mydevice)
        noise_output_logits = model(noise)
        noise_output_softmax = F.softmax(noise_output_logits, dim=1)
        TN = (noise_output_softmax[:, 1] < THRESHOLD).sum().item()
        confus_matrix.TN += TN
        confus_matrix.FP += (batch_size - TN)
        # print("signal softmax", output_softmax)
        # print("noise softmax", noise_output_softmax)
        # cnt += 1
        # if cnt >= 1:
        #     break
    # result
    print("=======")
    print(f"snr: {snr_target},  \t Pd: {confus_matrix.Pd(): .2f}, Pfa: {confus_matrix.Pfa(): .2f},\
    TP: {confus_matrix.TP}, FN: {confus_matrix.FN}, TN: {confus_matrix.TN}, FP: {confus_matrix.FP}")
    print("=======")


def main():

    model = CNN_Net().to(mydevice)
    # print(model)
    # load_model(model, save_path="./model_CNN.pt")

    optimizer = optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=5e-4)
    # for i in range(epochs):
    #     for snr in range(-20, 12, 2):
    #         signal_data.update(snr)
    #         signal_dataloader = DataLoader(signal_data, batch_size=num_SU)
    #         train(signal_dataloader, model, optimizer)
    # X, _ = next(iter(signal_dataloader))
    # print(X.shape)
    for i in range(epochs):
        train(signal_dataloader, model, optimizer)

    test(signal_test_dataloader, model)
    save_model(model, save_path="./model_param/model_CNN.pt")


if __name__ == "__main__":
    main()
