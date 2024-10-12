import numpy as np
import scipy.io as sio
import h5py
import json
from datetime import datetime
import torch
from numpy import record, record, argwhere

frame_perSNR = 4096
frame_perModula = 4096 * 26


def save_model(model, save_path='./model_param.pt'):
    # save whole model ,including model and parameter
    # torch.save(model, save_path)
    # only save model's parameter
    torch.save(model.state_dict(), save_path)


def load_model(model, save_path='./model_param.pt'):
    model.load_state_dict(torch.load(save_path))


def init_dataset(Path_dataset, Path_classes):
    # Open the dataset
    hdf5_file = h5py.File(Path_dataset,  'r')
    # Load the modulation classes. You can also copy and paste the content of classes-fixed.txt.
    modulation_classes = json.load(open(Path_classes, 'r'))

    # Read the HDF5 groups
    data = hdf5_file['X']
    modulation_onehot = hdf5_file['Y']
    SNR = hdf5_file['Z']
    string_list = json.load(open(Path_classes, 'r'))
    modulation_dict = {item: idx for idx, item in enumerate(string_list)}

    return (data, modulation_onehot, SNR, modulation_dict)


def import_data(Path, label):
    # just select the first column
    return sio.loadmat(Path)[f'feature_{label}'][:, 0]


def node_feature(Path, num_SU, label, sequence):
    '''return shape is (num_SU, num of featurs per node)'''
    data = np.zeros((num_SU, 64))  # shape=(num_SU, num of featurs per node)
    i = 0
    for seq in range(sequence, sequence+num_SU):
        i += 1
        data[i-1] = import_data(Path.replace('x', str(seq)), label)
    return data


def adj_matrix(feature, num_SU, rho=1):
    '''return shape is (num_SU, num_SU)'''
    A = np.zeros((num_SU, num_SU))
    for i in range(num_SU):
        for j in range(num_SU):
            if i == j:
                A[i, j] = 1  # add the identity matrix
            else:
                tmp = feature[i] - feature[j]
                A[i, j] = np.exp(-1*(np.linalg.norm(tmp) ** 2) / (rho ** 2))
    # D is diagonal matrix of A, then power -1/2, get D_exp
    D_exp = np.diag(np.power(np.sum(A, axis=1, keepdims=False), [-1/2]))
    A_hat = D_exp @ A @ D_exp
    return A_hat


def noise_feature(snr, num_SU=8):
    '''return noise signal, shape = (num_SU, 1024)
    '''
    Num_Sample = 1024
    power_noise = 1/(10**(snr/10)+1)
    power_noise_iq = power_noise/2
    noise_I = np.sqrt(power_noise_iq) * np.random.randn(num_SU, Num_Sample)
    # noise_Q = np.sqrt(power_noise_iq) * np.random.randn(num_SU, Num_Sample)
    return noise_I


def save_result(result_dict, file="test_result.json"):
    result_dict["time"] = str(datetime.now())
    # 尝试读取现有的JSON文件内容
    try:
        with open(file, "r") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        # 如果文件不存在或为空，则创建一个新的数组
        data = []
    # 将新数据追加到数组中
    data.append(result_dict)
    # 将数组写回JSON文件
    with open(file, "w") as f:
        json.dump(data, f, indent=4)
