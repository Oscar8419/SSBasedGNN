import json
from matplotlib import pyplot as plt
import numpy as np

json_filename = "./data_result/train_raw_data.json"

with open(json_filename, 'r') as json_file:
    data_dict = json.load(json_file)
positive_result = data_dict[-1]['positive']
negative_result = data_dict[-1]['negative']

sorted_negative_result = sorted(negative_result)
sorted_positive_result = sorted(positive_result)
positive_np = np.array(sorted_positive_result)
length_negative_result = len(sorted_negative_result)
length_positive_result = len(sorted_positive_result)
Pf_list = []
Pd_list = []
threshold_list = []
for i in range(1, 20):
    Pf = round(1-i*0.05, 2)
    idx = int(length_negative_result * (1-Pf))
    threshold = sorted_negative_result[idx]
    Pf_list.append(Pf)
    threshold_list.append(threshold)
    # print(f"Pf:{Pf}, threshold:{threshold}")
for threshold in threshold_list:
    Pd = ((positive_np > threshold).sum())/length_positive_result
    Pd_list.append(Pd)

plt.plot(Pf_list, Pd_list)
plt.xlabel('Pf')
plt.ylabel('Pd')
plt.show()
