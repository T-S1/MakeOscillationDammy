"""モジュールのインポート"""
import numpy as np
from scipy import signal
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# 自作モジュール
from src.visualize_data \
    import show_peaks, show_amp_clusters, RT_Drawer

"""データの読み込み"""
n_data = 50         # データ数
n_samples = 256     # 1データに含まれる点の数
t = np.zeros((n_data, n_samples))   # 時刻
x = np.zeros((n_data, n_samples))   # 信号強度
for i in range(n_data):
    data = np.loadtxt(f"./data/example1/{i:04}.csv", delimiter=",")
    t[i, :] = data[:, 0]    # 1列目が時刻のデータ
    x[i, :] = data[:, 1]    # 2列目が信号強度のデータ

"""代表値の抽出"""
amp_means = np.zeros(n_data)    # 代表値を格納する配列
for i in range(len(x)):
    # 極値の検出
    xi_abs = np.abs(x[i])   # 絶対値を計算
    peaks, _ = signal.find_peaks(xi_abs, prominence=0.3, width=5)
    # ピーク検出

    # 初回だけ極値検出のグラフを表示
    if i == 0:
        show_peaks(t[i], x[i], peaks)

    # 振幅の平均値を算出
    amp_mean = np.mean(xi_abs[peaks])

    amp_means[i] = amp_mean

"""k-meansの実行"""
n_clusters = 3  # クラスタ数

amp_means = amp_means.reshape(-1, 1)
kmeans = KMeans(
    n_clusters, random_state=100
).fit(amp_means)    # 学習
centers = kmeans.cluster_centers_   # クラスタ中心
labels = kmeans.predict(amp_means)  # 各データが属するクラスタ

show_amp_clusters(amp_means, labels, centers)   # クラスタリング結果の表示

"""リアルタイムな異常検知"""
data = np.loadtxt("./data/rt_example1.csv", delimiter=",")
t = data[:, 0]
x = data[:, 1]
n_window = 256  # クラスタ判定を行う時間窓の幅
drawer = RT_Drawer(t, x)    # グラフ表示クラス

for i in range(len(x)):
    # 一定サンプル毎にクラスタ予測
    if i % n_window == n_window - 1:
        i_start = i - n_window + 1
        x_abs = np.abs(x[i_start: i])   # 時間窓分のサンプル取得
        peaks, _ = signal.find_peaks(x_abs, prominence=0.3, width=5)
        amp_mean = np.mean(x_abs[peaks])    # 代表値の抽出
        label = kmeans.predict([[amp_mean]])[0]     # クラスタ予測
        drawer.paint_span(t[i_start], t[i], label)  # グラフに反映
    drawer.update()     # グラフのリアルタイムプロット
drawer.show()   # 最終の結果表示
