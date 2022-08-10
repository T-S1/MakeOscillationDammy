# import pdb; pdb.set_trace()
import os
import numpy as np
import matplotlib.pyplot as plt

txt_dir = "./data/temp"
fig_dir = "./figures/temp"

AMP = 1
FREQ = 1
WAV_SIG = 0.1
PHASE_SIG = 1

NUM_DATA = 30
TIME_LEN = 10
NUM_SAMP = 256


def main():

    if not os.path.isdir(txt_dir):
        os.makedirs(txt_dir)
    if not os.path.isdir(fig_dir):
        os.makedirs(fig_dir)

    fig = plt.figure()

    x = np.linspace(0, TIME_LEN, NUM_SAMP)

    for i in range(NUM_DATA):

        sig = np.random.normal(0, WAV_SIG, NUM_SAMP)
        y = AMP * np.sin(2 * np.pi * FREQ * x + PHASE_SIG) + sig

        np.savetxt(f"{txt_dir}/{i:05}.txt", y)

        plt.plot(x, y)
        plt.savefig(f"{fig_dir}/{i:05}.jpg")
        plt.cla()

    print("Done")


if __name__ == "__main__":
    main()
