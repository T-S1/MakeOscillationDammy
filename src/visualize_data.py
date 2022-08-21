import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


plt.rcParams['font.family'] = "Meiryo"

if not os.path.isdir("./figures/fft_example2"):
    os.makedirs("./figures/fft_example2")


def show_signal(t, x, name):
    fig = plt.figure()
    plt.title(name)
    plt.xlabel("時刻")
    plt.ylabel("信号強度")
    plt.xlim(t[0], t[-1])
    plt.ylim(-2.5, 2.5)
    plt.plot(t, x)
    plt.show()


def show_peaks(t, x, peaks, name):
    fig = plt.figure()
    plt.title(name)
    plt.xlabel("時刻")
    plt.ylabel("信号強度")
    plt.xlim(t[0], t[-1])
    plt.ylim(-2.5, 2.5)
    plt.plot(t, x)
    plt.plot(t[peaks], x[peaks], "x")
    plt.show()


def show_amp_clusters(ms, labels, centers, colors=["r", "g", "y"]):
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


def show_spectrum(f, spectrum, y_max=160):
    fig = plt.figure()
    plt.ylim(0, y_max)
    plt.plot(f, spectrum)
    plt.show()


def savefig_spectrum(idx, f, spectrum, y_max=160):
    fig = plt.figure()
    plt.ylim(0, y_max)
    plt.plot(f, spectrum)
    plt.savefig(f"./figures/fft_example2/{idx:04}.jpg")
    plt.close()


def show_fft_clusters(f, feats, labels, colors=["r", "g", "y"]):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel("データ番号")
    ax.set_ylabel("周波数")
    ax.set_zlabel("パワー")
    for i in range(len(feats)):
        arr_i = np.ones_like(f) * i
        ax.plot(arr_i, f, feats[i, :], color=colors[labels[i]])
    plt.show()


def show_fft_centers(f, centers, colors=["r", "g", "y"]):
    n_clusters = len(centers)
    fig, axs = plt.subplots(n_clusters, 1)
    for i in range(n_clusters):
        axs[i].set_ylabel("パワー")
        axs[i].plot(f, centers[i], color=colors[i])
    axs[0].set_title("クラスタ中心")
    axs[-1].set_xlabel("周波数")
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

    def paint_span(self, t_start, t_end, label, colors=["r", "g", "y"]):
        plt.axvspan(t_start, t_end, color=colors[label])

    def show(self):
        plt.xlim(self.t[0], self.t[-1])
        plt.show()
