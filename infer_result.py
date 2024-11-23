# from main import test, test_result
import main
from models import *
from utils import *
from model_dataset import *
from torch.utils.data import DataLoader
mydevice = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# test_result = {}
confus_matrix = Confusion_Matrix()
THRESHOLD = 0.5
snr_target = -20
modulation_target = "16QAM"
signal_test_data = Signal_Dataset(
    snr_target=snr_target, train=False, modulation=modulation_target)
num_SU = 8
signal_test_dataloader = DataLoader(signal_test_data, batch_size=num_SU)


modulation_dict = {'OOK': 0, '4ASK': 1, '8ASK': 2, 'BPSK': 3, 'QPSK': 4, '8PSK': 5, '16PSK': 6, '32PSK': 7, '16APSK': 8, '32APSK': 9, '64APSK': 10, '128APSK': 11,
                   '16QAM': 12, '32QAM': 13, '64QAM': 14, '128QAM': 15, '256QAM': 16, 'AM-SSB-WC': 17, 'AM-SSB-SC': 18, 'AM-DSB-WC': 19, 'AM-DSB-SC': 20, 'FM': 21, 'GMSK': 22, 'OQPSK': 23}

modu_list = ["16QAM", "32QAM", "64QAM", "128QAM", "QPSK"]

model = GCN(num_feature=1024, num_hidden=512, num_class=128,
            dropout=0.2, pooling="mean").to(device=mydevice)
print(model)
load_model(model)

for modulation_target in modu_list:
    signal_test_data = Signal_Dataset(
        snr_target=snr_target, train=False, modulation=modulation_target)
    for snr in range(-20, 32, 2):
        signal_test_data.update(snr)
        signal_test_dataloader = DataLoader(
            signal_test_data, batch_size=num_SU)
        main.test(signal_test_dataloader, model, save_result=True)
    save_result(main.test_result, modulation=modulation_target,
                file="infer_result.json")
    main.test_result = {}
