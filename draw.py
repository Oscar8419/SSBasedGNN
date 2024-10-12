import json
import matplotlib.pyplot as plt

# 读取JSON文件
with open('test_result.json', 'r') as f:
    data = json.load(f)

# 提取SNR、Pd和Pfa的值
snr_values = []
pd_values = []
pfa_values = []

for key, value in data[0].items():
    if "time" in key:
        continue
    snr = key.split('_')[1]  # 提取SNR值
    pd = value['Pd']
    pfa = value['Pfa']
    snr_values.append(int(snr))
    pd_values.append(pd)
    pfa_values.append(pfa)

# 绘制点线图
plt.figure(figsize=(10, 5))

# Pd和Pfa的点线图
plt.plot(snr_values, pd_values, marker='o', label='Pd')
plt.plot(snr_values, pfa_values, marker='x', label='Pfa')

plt.title('Pd and Pfa vs SNR')
plt.xlabel('SNR')
plt.ylabel('Probability')
plt.legend()

# Pd的点线图
# plt.subplot(1, 2, 1)
# plt.plot(snr_values, pd_values, marker='o')
# plt.title('Pd vs SNR')
# plt.xlabel('SNR')
# plt.ylabel('Pd')

# Pfa的点线图
# plt.subplot(1, 2, 2)
# plt.plot(snr_values, pfa_values, marker='o')
# plt.title('Pfa vs SNR')
# plt.xlabel('SNR')
# plt.ylabel('Pfa')

# 显示图表
plt.tight_layout()
plt.show()
