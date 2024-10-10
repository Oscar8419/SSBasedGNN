import numpy as np
import scipy.io as sio


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
