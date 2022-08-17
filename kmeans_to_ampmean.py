"""モジュールのインポート"""
import numpy as np
from scipy import signal
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

"""データの読み込み"""
data = np.loadtxt("./data/example1/0000.csv", delimiter=",")
t = data[:, 0]
x = data[:, 1]

"""代表値の抽出
1. 極値の検出
2. 振幅の平均値の算出
"""
# 1
x_abs = np.abs(x)
peaks, _ = signal.find_peaks(x_abs, prominence=0.3, width=5)

fig = plt.figure()
plt.plot(t, x)
plt.plot(t[peaks], x[peaks], "x")
plt.show()

# 2
m_amp = np.mean(x_abs[peaks])
print(f"振幅平均：{m_amp}")

"""全データの読み込み"""
n_data = 50
ms = np.zeros(n_data)

for i in range(n_data):
    data = np.loadtxt(f"./data/example1/{i:04}.csv", delimiter=",")
    t = data[:, 0]
    x = data[:, 1]

    x_abs = np.abs(x)
    peaks, _ = signal.find_peaks(x_abs, prominence=0.3, width=5)
    m_amp = np.mean(x_abs[peaks])

    ms[i] = m_amp

"""k-meansの実行"""
n_clusters = 3
colors = ["g", "r", "b"]

ms = ms.reshape(-1, 1)

kmeans = KMeans(
    n_clusters,
    random_state=100
).fit(ms)
centers = kmeans.cluster_centers_
labels = kmeans.predict(ms)

fig = plt.figure()
for i in range(n_clusters):
    idxs = np.arange(n_data)[labels == i]
    plt.scatter(idxs, ms[idxs], c=colors[i])
    plt.hlines(centers[i], xmin=0, xmax=n_data)
plt.show()

"""リアルタイムな異常検知"""
data = np.loadtxt("./data/rt_example1.csv", delimiter=",")
t = data[:, 0]
x = data[:, 1]
n_window = 256
width = 500
interval = 32

fig = plt.figure()

for i in range(len(x)):
    if i % n_window == n_window - 1:
        i_start = i - n_window + 1
        x_abs = np.abs(x[i_start: i])
        peaks, _ = signal.find_peaks(x_abs, prominence=0.3, width=5)
        m_amp = np.mean(x_abs[peaks])
        label = kmeans.predict([[m_amp]])[0]
        plt.axvspan(t[i_start], t[i], color=colors[label])
    if i % interval == interval - 1:
        i_left = max(0, i - width)
        i_right = max(width, i)
        plt.xlim(t[i_left], t[i_right])
        i_start = i - interval + 1
        plt.plot(t[i_start: i + 2], x[i_start: i + 2], color='k')
        plt.pause(0.1)

plt.xlim(t[0], t[-1])
plt.savefig("./figures/rt_ampmean.jpg")
plt.show()
