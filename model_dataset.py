from torch.utils.data import Dataset
import h5py
import torch

modulation_dict = {'OOK': 0, '4ASK': 1, '8ASK': 2, 'BPSK': 3, 'QPSK': 4, '8PSK': 5, '16PSK': 6, '32PSK': 7, '16APSK': 8, '32APSK': 9, '64APSK': 10, '128APSK': 11,
                   '16QAM': 12, '32QAM': 13, '64QAM': 14, '128QAM': 15, '256QAM': 16, 'AM-SSB-WC': 17, 'AM-SSB-SC': 18, 'AM-DSB-WC': 19, 'AM-DSB-SC': 20, 'FM': 21, 'GMSK': 22, 'OQPSK': 23}


# Dataset at given snr range and modulation
class Signal_Dataset(Dataset):
    def __init__(self, path_data="D:\Dataset\\2018.01\GOLD_XYZ_OSC.0001_1024.hdf5", modulation="16QAM", snr_target=-20, train=True) -> None:
        super(Signal_Dataset, self).__init__()
        self.path_data = path_data
        hdf5_file = h5py.File(path_data,  'r')
        self.data = hdf5_file['X']
        self.modulation = modulation
        self.frame_perSNR = 4096
        self.frame_perModula = 4096 * 26
        self.snr_target = snr_target
        self.train = train
        self.idx_modulation = modulation_dict[modulation]
        self.idxBase_modulation_data = self.idx_modulation * self.frame_perModula
        self.idxoffset_snr_data = (self.snr_target + 20)//2 * self.frame_perSNR
        # 4096 is divided into 3272/824   Train set/Test set
        self.train_num = 3272
        self.test_num = 824
        # self.modulation_onehot = hdf5_file['Y']
        # self.SNR = hdf5_file['Z']

    def __len__(self):
        # 4096 is divided into 3272/824   Train set/Test set
        if self.train:
            return self.train_num
        else:
            return self.test_num

    # def set_SNR(self, snr):
    #     if (snr % 2 != 0) or (snr < -20) or (snr > 30):
    #         raise ValueError("Wrong snr value")
    #     self.snr_target = snr

    def update(self, snr_target, modulation="16QAM"):
        if (snr_target % 2 != 0) or (snr_target < -20) or (snr_target > 30):
            raise ValueError("Wrong snr value")
        if modulation not in modulation_dict:
            raise ValueError("Wrong modulation")
        self.snr_target = snr_target
        self.idx_modulation = modulation_dict[modulation]
        self.idxBase_modulation_data = self.idx_modulation * self.frame_perModula
        self.idxoffset_snr_data = (self.snr_target + 20)//2 * self.frame_perSNR

    def __getitem__(self, idx):
        # idx is the index at particular modulation and snr
        idx_data = self.idxBase_modulation_data + self.idxoffset_snr_data + idx
        if self.train is False:
            idx_data += self.train_num  # when testing ,idx add self.train_num
        data_target = self.data[idx_data]
        label = torch.tensor([1])
        return (data_target, label)
