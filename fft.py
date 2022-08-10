import numpy as np

# データの読み込み
x = np.loadtxt(f"./data/temp/00000.txt")



np.fft.fft(x)


