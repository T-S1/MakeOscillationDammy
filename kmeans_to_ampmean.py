"""モジュールのインポート"""
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

"""データの読み込み"""
data = np.loadtxt(f"./data/normal/0000.csv", delimiter=",")
t = data[:, 0]
x = data[:, 1]

x_abs = np.abs(x)

"""代表値の抽出
1. 極値の検出
2. 振幅の平均値の算出
"""
# 1
peaks, _ = signal.find_peaks(x_abs, prominence=0.5, width=5)
# print(peaks)

fig = plt.figure()
plt.plot(t, x)
plt.plot(t[peaks], x[peaks], "x")
plt.show()

# 2
mean_amp = np.mean(x_abs[peaks])
print(f"振幅平均値: {mean_amp}")

"""k-meansの実行"""
