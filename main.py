import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import time
from utils import *
from models import GCN, Confusion_Matrix
import glob

# Data_Path_1 = "./signal1/SNR_-19_Signal1_x"  # .mat file path
# Data_Path_0 = "./signal0/SNR_-19_Signal0_x"
Data_Path_1 = "./signal1/*.mat"  # .mat file path
Data_Path_0 = "./signal0/*.mat"
num_SU = 10
Graph_dict = {}
Data_File_1 = glob.glob(Data_Path_1)
Data_File_0 = glob.glob(Data_Path_0)
SNR_low, SNR_high = -20, 5
Graph_num = len(Data_File_0) // (num_SU *
                                 (SNR_high-SNR_low+1))  # graph num per SNR
confus_matrix = Confusion_Matrix()
THRESHOLD = 0.5
# def train(adj_matrix, features, model, labels, epoch=10):


def train(Graph_dict, model, epoch=10):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    for ep in range(epoch):
        t_start = time.time()
        # optimizer.zero_grad()
        confus_matrix.clear()
        for SNR in [-10]:  # range(0, 6):
            for i in range(Graph_num):
                feature = Graph_dict[f"Signal0_SNR{SNR}_Graph{i}"]["feature"]
                adj_matrix = Graph_dict[f"Signal0_SNR{SNR}_Graph{i}"]["adj_matrix"]
                label = Graph_dict[f"Signal0_SNR{SNR}_Graph{i}"]["label"]
                output, output_softmax = model(feature, adj_matrix)
                loss_train = F.nll_loss(output, label)
                optimizer.zero_grad()
                loss_train.backward()
                optimizer.step()
                if output_softmax[0, 1] > THRESHOLD:
                    confus_matrix.FP += 1
                else:
                    confus_matrix.TN += 1
                # print(
                #     f"Signal0_SNR{SNR}_Graph{i}, loss:{loss_train},output:", output_softmax)

                feature = Graph_dict[f"Signal1_SNR{SNR}_Graph{i}"]["feature"]
                adj_matrix = Graph_dict[f"Signal1_SNR{SNR}_Graph{i}"]["adj_matrix"]
                label = Graph_dict[f"Signal1_SNR{SNR}_Graph{i}"]["label"]
                output, output_softmax = model(feature, adj_matrix)
                loss_train = F.nll_loss(output, label)
                optimizer.zero_grad()
                loss_train.backward()
                optimizer.step()
                if output_softmax[0, 1] > THRESHOLD:
                    confus_matrix.TP += 1
                else:
                    confus_matrix.FN += 1
                # print(
                #     f"Signal1_SNR{SNR}_Graph{i}, loss:{loss_train},output:", output_softmax)
                # print("=====")

        if (ep+1) % 1 == 0:
            print(
                f"epoch:{ep}, Pd:{confus_matrix.Pd():.2f}, Pfa:{confus_matrix.Pfa():.2f}")
        #     print(
        #         f"epoch {ep}, loss:{loss_train}, time_spend:{time.time() - t_start }")
        #     print("output: ", output_softmax)
        #     print("====")


def save_model(model, save_path='./model_param.pt'):
    # save whole model ,including model and parameter
    # torch.save(model, save_path)
    # only save model's parameter
    torch.save(model.state_dict(), save_path)


def load_model(model, save_path='./model_param.pt'):
    model.load_state_dict(torch.load(save_path))
    # model.eval()


def main():
    torch.manual_seed(2024)
    np.random.seed(2024)

    mydevice = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for SNR in range(SNR_low, SNR_high+1):
        for i in range(Graph_num):
            path_0 = f"./signal0/SNR_{SNR}_Signal0_x"
            path_1 = f"./signal1/SNR_{SNR}_Signal1_x"
            data_0 = node_feature(path_0, num_SU, 0, i*num_SU+1)
            data_1 = node_feature(path_1, num_SU, 1, i*num_SU+1)
            label_1 = torch.ones(1, device=mydevice, dtype=torch.int64)
            label_0 = torch.zeros(1, device=mydevice, dtype=torch.int64)

            # shape=(num_SU, num_SU)
            adj_matrix_0 = adj_matrix(data_0, num_SU, rho=1)
            adj_matrix_1 = adj_matrix(data_1, num_SU, rho=1)
            adj_matrix_0 = torch.tensor(
                adj_matrix_0, dtype=torch.float32, device=mydevice)
            adj_matrix_1 = torch.tensor(
                adj_matrix_1, dtype=torch.float32, device=mydevice)

            data_0 = torch.tensor(data_0, dtype=torch.float32, device=mydevice)
            data_1 = torch.tensor(data_1, dtype=torch.float32, device=mydevice)
            tmp = {"feature": data_0, "adj_matrix": adj_matrix_0,
                   "label": label_0, "SNR": SNR}
            Graph_dict[f"Signal0_SNR{SNR}_Graph{i}"] = tmp
            tmp = {"feature": data_1, "adj_matrix": adj_matrix_1,
                   "label": label_1, "SNR": SNR}
            Graph_dict[f"Signal1_SNR{SNR}_Graph{i}"] = tmp

    model = GCN(64, 32, 16, dropout=0.2, pooling="mean").to(device=mydevice)
    load_model(model)
    # optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    # train(adj_matrix_1, data_1, model, label_1)
    train(Graph_dict, model)
    save_model(model)


if __name__ == "__main__":
    main()
