"""モジュールのインポート"""
# import pdb; pdb.set_trace()
import numpy as np
from scipy import signal
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

n_data = 50
ms = np.zeros(n_data)

for i in range(n_data):
    """データの読み込み"""
    data = np.loadtxt(f"./data/example1/{i:04}.csv", delimiter=",")
    t = data[:, 0]
    x = data[:, 1]

    x_abs = np.abs(x)

    """代表値の抽出
    1. 極値の検出
    2. 振幅の平均値の算出
    """
    # 1
    peaks, _ = signal.find_peaks(x_abs, prominence=0.3, width=5)
    # print(peaks)

    # fig = plt.figure()
    # plt.plot(t, x)
    # plt.plot(t[peaks], x[peaks], "x")
    # plt.show()

    # 2
    m_amp = np.mean(x_abs[peaks])
    # print(f"振幅平均値: {mean_amp}")

    ms[i] = m_amp

"""k-meansの実行"""
n_clusters = 3
colors = ["g", "r", "b"]

ms = ms.reshape(-1, 1)

kmeans = KMeans(
    n_clusters,
    n_init=10,
    max_iter=300,
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
n_samples = 4096
n_window = 256
width = 500
interval = 50
clus_names = ["normal", "abnormal_1", "abnormal_2"]

fig = plt.figure()

for i in range(1, n_samples):
    if i % n_window == n_window - 1:
        i_start = i - n_window + 1
        x_abs = np.abs(x[i_start: i])
        peaks, _ = signal.find_peaks(x_abs, prominence=0.3, width=5)
        m_amp = np.mean(x_abs[peaks])
        label = kmeans.predict([[m_amp]])[0]
        plt.axvspan(t[i_start], t[i], color=colors[label])
    if i % interval == 0:
        ileft = max(0, i - width)
        iright = max(width, i)
        plt.xlim(t[ileft], t[iright])
        plt.plot(t[i - interval: i + 1], x[i - interval: i + 1], color='k')
        plt.pause(0.2)

plt.show()
