import os
import numpy as np
import matplotlib.pyplot as plt


plt.rcParams['font.family'] = "Meiryo"


def show_peaks(t, x, peaks):
    fig = plt.figure()
    plt.title("ピーク検出結果")
    plt.xlabel("時刻(t)")
    plt.ylabel("信号強度(x)")
    plt.plot(t, x)
    plt.plot(t[peaks], x[peaks], "x")
    plt.show()


def show_amp_clusters(ms, labels, centers, colors=["r", "g", "b"]):
    n_data = len(ms)
    n_clusters = len(colors)
    fig = plt.figure()
    plt.title("クラスタリング結果")
    plt.xlabel("データ番号")
    plt.ylabel("平均振幅")
    for cluster in range(n_clusters):
        idxs = np.arange(n_data)[labels == cluster]
        plt.scatter(idxs, ms[idxs], c=colors[cluster])
        plt.hlines(centers[cluster], xmin=0, xmax=n_data)
    plt.show()


class RT_Drawer():
    def __init__(self, t, x, width=500, interval=32):

        if not os.path.isdir("./figures"):
            os.makedirs("./figures")

        self.width = width
        self.interval = interval
        self.t = t
        self.x = x
        self.count = 0
        self.fig = plt.figure()
        plt.xlabel("時刻(t)")
        plt.ylabel("信号強度(x)")

    def update(self):
        if self.count % self.interval == self.interval - 1:
            i_left = max(0, self.count - self.width)
            i_right = max(self.width, self.count)
            plt.xlim(self.t[i_left], self.t[i_right])
            i_start = self.count - self.interval + 1
            plt.plot(
                self.t[i_start: self.count + 2],
                self.x[i_start: self.count + 2],
                color='k'
            )
            plt.pause(0.1)

        self.count += 1

    def paint_span(self, t_start, t_end, label, colors=["r", "g", "b"]):
        plt.axvspan(t_start, t_end, color=colors[label])

    def show(self):
        plt.xlim(self.t[0], self.t[-1])
        plt.savefig("./figures/rt_ampmean.jpg")
        plt.show()


def show_spectrum(f, spectrum):
    fig = plt.figure()
    plt.plot(f, spectrum)
    plt.show()


def show_fft_clusters(feats, labels, centers, colors=["r", "g", "b"]):
    n_clusters = len(centers)
    n_data = len(feats)
    fig = plt.figure()
    plt.title("各データの属するクラスタ・中心との距離")
    plt.ylabel("クラスタ中心までの距離")
    plt.xlabel("データ番号")
    for i in range(n_clusters):
        idxs = np.arange(n_data)[labels == i]
        dists = np.sqrt(np.sum((feats[idxs] - np.array([centers[i]]))**2, axis=1))
        plt.bar(idxs, dists, color=colors[i], label=f"Cluster {i}")
    plt.legend()
    plt.show()


def show_fft_centers(f, centers, colors=["r", "g", "b"]):
    n_clusters = len(centers)
    fig, axs = plt.subplots(n_clusters, 1)
    plt.title("クラスタ中心")
    for i in range(n_clusters):
        axs[i].set_ylabel("パワー")
        axs[i].plot(f, centers[i], color=colors[i])
    axs[-1].set_xlabel("周波数")
    plt.show()
