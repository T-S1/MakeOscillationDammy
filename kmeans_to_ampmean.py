"""モジュールのインポート"""
import pdb; pdb.set_trace()
import numpy as np
from scipy import signal
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

n_data = 50
ms = np.zeros(n_data)

for i in range(n_data):
    """データの読み込み"""
    data = np.loadtxt(f"./data/example/{i:04}.csv", delimiter=",")
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
colors = ["r", "g", "b"]

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
np.random.seed(100)
n_iter = 20



for i in range(n_iter):
    x_new = np.loadtxt
