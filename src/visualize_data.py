# import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

plt.rcParams['font.family'] = "Meiryo"  # フォントの設定

# if not os.path.isdir("./figures/fft_example2"):
#     os.makedirs("./figures/fft_example2")


def show_signals(t, x, step=8, n_rows=3, n_cols=2, y_max=2.5):
    fig, axs = plt.subplots(n_rows, n_cols, sharex=True, sharey=True)   # 複数グラフを表示する準備
    for i in range(n_rows):
        for j in range(n_cols):
            idx = step * (n_cols * i + j)
            axs[i, j].set_title(f"{idx:04}")    # グラフタイトル
            axs[i, j].set_xlim(t[idx, 0], t[idx, -1])   # x軸の上下限
            axs[i, j].set_ylim(-y_max, y_max)           # y軸の上下限
            axs[i, j].plot(t[idx], x[idx])      # 波形のプロット
    plt.tight_layout()  # 体裁を整える
    plt.show()          # 表示


def show_peaks(t, x, peaks_list, step=8, n_rows=3, n_cols=2, y_max=2.5):
    fig, axs = plt.subplots(n_rows, n_cols, sharex=True, sharey=True)
    for i in range(n_rows):
        for j in range(n_cols):
            idx = step * (n_cols * i + j)
            peaks = peaks_list[idx]
            axs[i, j].set_title(f"{idx:04}")
            axs[i, j].set_xlim(t[idx, 0], t[idx, -1])
            axs[i, j].set_ylim(-y_max, y_max)
            axs[i, j].plot(t[idx], x[idx])
            axs[i, j].plot(t[idx, peaks], x[idx, peaks], "x")
    plt.tight_layout()
    plt.show()


def show_amp_means(amp_means):
    n_data = len(amp_means)
    fig = plt.figure()
    plt.title("代表値")
    plt.xlabel("データ番号")
    plt.ylabel("平均振幅")
    idxs = np.arange(n_data)
    plt.scatter(idxs, amp_means)
    plt.show()


def show_amp_clusters(amp_means, labels, centers, colors=["r", "g", "y"]):
    n_data = len(amp_means)
    n_clusters = len(colors)
    fig = plt.figure()
    plt.title("クラスタリング結果")
    plt.xlabel("データ番号")
    plt.ylabel("平均振幅")
    for cluster in range(n_clusters):
        idxs = np.arange(n_data)[labels == cluster]
        plt.scatter(idxs, amp_means[idxs], c=colors[cluster])
        plt.hlines(centers[cluster], xmin=0, xmax=n_data)
    plt.show()


def show_signals_with_cluster(
    t, x, labels, colors=["r", "g", "y"], step=8, n_rows=3, n_cols=2, y_max=2.5
):
    fig, axs = plt.subplots(n_rows, n_cols, sharex=True, sharey=True)
    for i in range(n_rows):
        for j in range(n_cols):
            idx = step * (n_cols * i + j)
            axs[i, j].set_title(f"{idx:04}")
            axs[i, j].set_xlim(t[idx, 0], t[idx, -1])
            axs[i, j].set_ylim(-y_max, y_max)
            axs[i, j].plot(t[idx], x[idx], color=colors[labels[idx]])
    plt.tight_layout()
    plt.show()


def show_spectrums(freq, spectrums, step=8, n_rows=3, n_cols=2, y_max=200):
    fig, axs = plt.subplots(n_rows, n_cols, sharex=True, sharey=True)
    for i in range(n_rows):
        for j in range(n_cols):
            idx = step * (n_cols * i + j)
            axs[i, j].set_title(f"{idx:04}")
            axs[i, j].set_xlim(freq[0], freq[-1])
            axs[i, j].set_ylim(0, y_max)
            axs[i, j].plot(freq, spectrums[idx])
    plt.tight_layout()
    plt.show()


# def savefig_spectrums(idx, f, spectrum, y_max=200):
#     fig = plt.figure()
#     plt.ylim(0, y_max)
#     plt.plot(f, spectrum)
#     plt.savefig(f"./figures/fft_example2/{idx:04}.jpg")
#     plt.close()


def show_all_spectrums(freq, spectrums):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel("データ番号")
    ax.set_ylabel("周波数")
    ax.set_zlabel("パワー")
    for i in range(len(spectrums)):
        arr_i = np.ones_like(freq) * i
        ax.plot(arr_i, freq, spectrums[i, :], color="tab:blue")
    plt.show()


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


def show_fft_centers(f, centers, colors=["r", "g", "y"], y_max=200):
    n_clusters = len(centers)
    fig, axs = plt.subplots(n_clusters, 1, sharex=True, sharey=True)
    plt.suptitle("クラスタ中心")
    for i in range(n_clusters):
        axs[i].set_ylabel("パワー")
        axs[i].plot(f, centers[i], color=colors[i])
        axs[i].set_ylim(0, y_max)
    axs[-1].set_xlabel("周波数")
    plt.tight_layout()
    plt.show()


class RT_Drawer():
    def __init__(self, t, x, width=500, interval=32, y_max=3):

        # if not os.path.isdir("./figures"):
        #     os.makedirs("./figures")

        self.width = width
        self.interval = interval
        self.t = t
        self.x = x
        self.count = 0
        self.fig = plt.figure()
        plt.xlabel("時刻(t)")
        plt.ylabel("信号強度(x)")
        plt.ylim(-y_max, y_max)

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
