"""モジュールのインポート"""
import numpy as np
from scipy import signal
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# 自作モジュール
from src.visualize_data \
    import show_spectrum, show_fft_clusters, show_fft_centers, RT_Drawer

"""データの読み込み"""
n_data = 50         # データ数
n_samples = 256     # 1データに含まれる点の数
t = np.zeros((n_data, n_samples))   # 時刻
x = np.zeros((n_data, n_samples))   # 信号強度
for i in range(n_data):
    data = np.loadtxt(f"./data/example2/{i:04}.csv", delimiter=",")
    t[i, :] = data[:, 0]    # 1列目が時刻のデータ
    x[i, :] = data[:, 1]    # 2列目が信号強度のデータ

"""FFT"""
f_0 = 1 / (t[0, -1] - t[0, 0])
f = np.arange(len(t[0])) * f_0     # 周波数
feats = np.zeros((n_data, n_samples))   # 特徴量を格納する配列
for i in range(len(x)):
    fourier = np.fft.fft(x[i, :])
    spectrum = np.sqrt(fourier.real**2 + fourier.imag**2)   # パワースペクトルの算出

    # 各種のFFT結果を1度だけ表示
    if i == 0 or i == 30 or i == 40:
        show_spectrum(f, spectrum)

    # k-meansのための前処理
    feat = spectrum / np.sum(spectrum)  # スペクトルの総和による正規化
    filter_size = 3
    ave_filter = np.ones(filter_size) / filter_size
    feat = np.convolve(ave_filter, feat, "same")    # 近傍の値で平均化

    feats[i, :] = feat

"""k-meansの実行"""
n_clusters = 3

kmeans = KMeans(n_clusters, random_state=100).fit(feats)
centers = kmeans.cluster_centers_
labels = kmeans.predict(feats)

show_fft_clusters(feats, labels, centers)   # クラスタリング結果のグラフ
show_fft_centers(f, centers)                # クラスタ中心のグラフ

"""リアルタイムな異常検知"""
data = np.loadtxt("./data/rt_example2.csv", delimiter=",")
t = data[:, 0]
x = data[:, 1]
n_window = 256
drawer = RT_Drawer(t, x)

for i in range(len(x)):
    # 一定サンプル毎にクラスタ予測
    if i % n_window == n_window - 1:
        i_start = i - n_window + 1
        x_win = x[i_start: i + 1]       # 時間窓分のサンプル取得
        fourier = np.fft.fft(x_win)
        spectrum = np.sqrt(fourier.real**2 + fourier.imag**2)
        feat = spectrum / np.sum(spectrum)
        filter_size = 3
        ave_filter = np.ones(filter_size) / filter_size
        feat = np.convolve(ave_filter, feat, "same")    # 前処理
        label = kmeans.predict([feat])[0]   # クラスタ予測
        drawer.paint_span(t[i_start], t[i], label)  # グラフに反映
    drawer.update()     # グラフのリアルタイムプロット
drawer.show()   # 最終の結果表示
