import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

# データの読み込み
x = np.loadtxt(f"./data/temp/00000.txt")

x_abs = np.abs(x)

peaks, _ = signal.find_peaks(x_abs, prominence=0.8, width=5)
# print(peaks)



fig = plt.figure()
plt.plot(x)

plt.plot(peaks, x[peaks], "x")
plt.show()
