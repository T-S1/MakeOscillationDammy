import pdb; pdb.set_trace()
import os
import numpy as np
import matplotlib.pyplot as plt

txt_dir = "./data/normal"
fig_dir = "./figures/normal"

BASE_AMP = 1
BASE_FREQ = 1

AMP_SIGMA = 0.1
FREQ_SIGMA = 0
PHASE_SHIFT_SIGMA = 1

NUM_DATA = 30
TIME_LEN = 10
SAMP_RATE = 30

SEED = 200


def main():

    np.random.seed(SEED)

    if not os.path.isdir(txt_dir):
        os.makedirs(txt_dir)
    if not os.path.isdir(fig_dir):
        os.makedirs(fig_dir)

    fig = plt.figure()

    NUM_SAMP = SAMP_RATE * TIME_LEN

    for i in range(NUM_DATA):
        xs = np.zeros(NUM_SAMP)
        ys = np.zeros(NUM_SAMP)
        a = np.random.poisson(BASE_AMP)
        f = np.random.poisson(BASE_FREQ)
        b = 0
        phi = np.random.normal(0, PHASE_SHIFT_SIGMA)
        a_old = a
        x_old = 0
        for j in range(NUM_SAMP):
            x = j / SAMP_RATE
            y = a * np.sin(2 * np.pi * f * x + phi) + b
            xs[j] = x
            ys[j] = y

            if x > x_old + 1 / (4 * f):
                a = np.random.poisson(BASE_AMP)
                f = np.random.poisson(BASE_FREQ)
                dif = (np.sin(2 * np.pi * f * x + phi + np.pi / 2)
                       - np.sin(2 * np.pi * f + phi))
                if dif < 0:
                    b -= a_old - a
                else:
                    b += a_old - a
                x_old = x

        np.savetxt(f"{txt_dir}/{i:05}.txt", ys)

        plt.plot(xs, ys)
        plt.savefig(f"{fig_dir}/{i:05}.jpg")
        plt.cla()

    print("Done")


if __name__ == "__main__":
    main()
