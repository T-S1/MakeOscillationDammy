import numpy as np
import pickle
from src.visualize_data import RT_Drawer

"""リアルタイムな異常検知"""
data = np.loadtxt("./data/rt_example2.csv", delimiter=",")
t = data[:, 0]
x = data[:, 1]
n_window = 256  # クラスタ判定を行う時間窓の幅

with open("kmeans_model_2.pkl", "rb") as fp:
    kmeans = pickle.load(fp)    # k-meansモデルの読み込み
drawer = RT_Drawer(t, x)        # 動的グラフ表示クラス

for i in range(len(x)):
    if i % n_window == n_window - 1:    # 一定サンプル毎にクラスタ予測
        i_start = i - n_window + 1
        fourier = np.fft.fft(x[i_start: i + 1])
        spectrum = np.sqrt(fourier.real**2 + fourier.imag**2)   # パワースペクトルの算出
        spectrum = spectrum[: len(spectrum) // 2]
        label = kmeans.predict([spectrum])[0]       # クラスタ予測
        drawer.paint_span(t[i_start], t[i], label)  # グラフに反映
    drawer.update()     # グラフのリアルタイムプロット
drawer.show()   # 最終の結果表示
