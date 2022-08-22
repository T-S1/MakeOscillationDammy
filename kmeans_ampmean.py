"""モジュールのインポート"""
import numpy as np
from scipy import signal
from sklearn.cluster import KMeans
import pickle
from src.visualize_data import (
    show_signals, show_peaks, show_amp_means,
    show_amp_clusters, show_signals_with_cluster
)   # 自作モジュール

"""データの読み込み"""
n_data = 50         # データ数
n_samples = 256     # 1データに含まれる点の数
t = np.zeros((n_data, n_samples))   # 時刻
x = np.zeros((n_data, n_samples))   # 信号強度
for i in range(n_data):
    data = np.loadtxt(f"./data/example1/{i:04}.csv", delimiter=",")
    t[i, :] = data[:, 0]    # 1列目が時刻のデータ
    x[i, :] = data[:, 1]    # 2列目が信号強度のデータ
show_signals(t, x)  # 読み込んだデータのグラフ表示

"""代表値の抽出"""
amp_means = np.zeros(n_data)    # 代表値を格納する配列
peaks_list = []                 # ピークのインデックスを格納するリスト
for i in range(len(x)):
    xi_abs = np.abs(x[i])   # 絶対値を計算
    peaks, _ = signal.find_peaks(xi_abs, prominence=0.3, width=5)   # ピーク検出

    amp_mean = np.mean(xi_abs[peaks])   # 振幅平均を算出
    amp_means[i] = amp_mean
    peaks_list.append(peaks)

    if i % 8 == 0:     # 8回毎に表示
        print(f"データ{i:04}の平均値：{amp_mean}")  # 振幅平均の表示

show_peaks(t, x, peaks_list)
show_amp_means(amp_means)

"""k-meansの実行"""
n_clusters = 3  # クラスタ数

amp_means = amp_means.reshape(-1, 1)    # Kmeansの仕様に合わせる
kmeans = KMeans(n_clusters, random_state=100).fit(amp_means)    # 学習
centers = kmeans.cluster_centers_       # クラスタ中心
labels = kmeans.labels_                 # 各データが属するクラスタ

show_amp_clusters(amp_means, labels, centers)   # クラスタリング結果の表示
show_signals_with_cluster(t, x, labels)         # 波形をクラスタで色分け表示

with open("kmeans_model_1.pkl", "wb") as fp:
    pickle.dump(kmeans, fp)     # モデル保存
