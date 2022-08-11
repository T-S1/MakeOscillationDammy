import pdb; pdb.set_trace()
import numpy as np
from scipy import signal
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

N_SAMPLES = 4096

data = np.loadtxt(f"./data/rt_example1.csv", delimiter=",")
t = data[:, 0]
x = data[:, 1]

spectrum = np.fft.fft(x)
freq_0 = 1 / N_SAMPLES
f = np.arange(N_SAMPLES, step=freq_0)

fig = plt.figure()
plt.plot(f, spectrum)
plt.show()
