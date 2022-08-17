"""モジュールのインポート"""
# import pdb; pdb.set_trace()
import os
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

"""データの読み込み"""
data = np.loadtxt("./data/example2/0000.csv", delimiter=",")
t = data[:, 0]
x = data[:, 1]

"""FFT"""
# 1
fourier = np.fft.fft(x)
spectrum = fourier.real**2 + fourier.imag**2
f_0 = 1 / (t[-1] - t[0])
f = np.arange(len(t)) * f_0

fig = plt.figure()
plt.plot(f, spectrum)
plt.show()

"""全データの読み込み"""
n_data = 50
n_features = 256
feats = np.zeros((n_data, n_features))

if not os.path.isdir("./figures/fft_example2"):
    os.makedirs("./figures/fft_example2")

fig = plt.figure()

for i in range(n_data):
    data = np.loadtxt(f"./data/example2/{i:04}.csv", delimiter=",")
    t = data[:, 0]
    x = data[:, 1]

    fourier = np.fft.fft(x)
    spectrum = fourier.real**2 + fourier.imag**2
    f_0 = 1 / (t[-1] - t[0])
    f = np.arange(len(t)) * f_0

    # feats[i, :] = spectrum / np.amax(spectrum)
    feat = spectrum / np.sum(spectrum)

    filter_size = 3
    ave_win = np.ones(filter_size) / filter_size
    feat = np.convolve(ave_win, feat, "same")
    feats[i, :] = feat

    plt.plot(f, feat)
    plt.savefig(f"./figures/fft_example2/{i:04}.jpg")
    plt.cla()

plt.close()

"""k-meansの実行"""
n_clusters = 3
colors = ["r", "g", "b"]

kmeans = KMeans(
    n_clusters,
    random_state=100
).fit(feats)
centers = kmeans.cluster_centers_
labels = kmeans.predict(feats)

fig = plt.figure()
for i in range(n_clusters):
    idxs = np.arange(n_data)[labels == i]
    dists = np.sqrt(np.sum((feats[idxs] - np.array([centers[i]]))**2, axis=1))
    plt.bar(idxs, dists, color=colors[i])
plt.show()

fig, axs = plt.subplots(n_clusters, 1)
for i in range(n_clusters):
    axs[i].plot(f, centers[i], color=colors[i])
plt.show()

"""リアルタイムな異常検知"""
data = np.loadtxt("./data/rt_example2.csv", delimiter=",")
t = data[:, 0]
x = data[:, 1]
n_window = 256
width = 500
interval = 32

fig = plt.figure()

for i in range(len(x)):
    if i % n_window == n_window - 1:
        i_start = i - n_window + 1
        x_win = x[i_start: i + 1]

        fourier = np.fft.fft(x_win)
        spectrum = fourier.real**2 + fourier.imag**2
        f = np.arange(len(x_win)) * f_0
        feat = spectrum / np.sum(spectrum)

        filter_size = 3
        ave_win = np.ones(filter_size) / filter_size
        feat = np.convolve(ave_win, feat, "same")

        label = kmeans.predict([feat])[0]
        plt.axvspan(t[i_start], t[i], color=colors[label])

    if i % interval == interval - 1:
        i_left = max(0, i - width)
        i_right = max(width, i)
        plt.xlim(t[i_left], t[i_right])
        i_start = i - interval + 1
        plt.plot(t[i_start: i + 2], x[i_start: i + 2], color='k')
        plt.pause(0.1)

plt.xlim(t[0], t[-1])
plt.savefig("./figures/rt_fft.jpg")
plt.show()
