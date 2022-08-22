# import pdb; pdb.set_trace()
import argparse
import numpy as np
import matplotlib.pyplot as plt

DURATION = 100
N_SAMPLES = 2560
NOISE_DEV = 0.1
amp_lists = [[1], [0.5, 0.3, 0.2], [0.5, 0.3, 0.2]]
freq_lists = [[1], [1, 2, 4], [1, 3, 5]]
pattern = [0, 1, 1, 2, 2]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed",
        help="seed value",
        type=int, default=100
    )
    args = parser.parse_args()

    np.random.seed(args.seed)

    t = np.linspace(0, DURATION, N_SAMPLES)
    x = np.zeros(N_SAMPLES)
    n_shift = N_SAMPLES // len(pattern)
    amp_list_old = amp_lists[pattern[0]]
    freq_list_old = freq_lists[pattern[0]]

    for i, label in enumerate(pattern):
        i_start = n_shift * i
        i_end = n_shift * (i + 1)
        amp_list = amp_lists[label]
        freq_list = freq_lists[label]

        for j in range(len(amp_list_old)):
            amp = amp_list_old[j]
            freq = freq_list_old[j]
            a = np.linspace(amp, 0, n_shift)
            x[i_start: i_end] += a * np.sin(2 * np.pi * freq * t[i_start: i_end])

        for j in range(len(amp_list)):
            amp = amp_list[j]
            freq = freq_list[j]
            a = np.linspace(0, amp, n_shift)
            x[i_start: i_end] += a * np.sin(2 * np.pi * freq * t[i_start: i_end])

        amp_list_old = amp_list
        freq_list_old = freq_list

    eps = np.random.normal(0, NOISE_DEV, N_SAMPLES)
    x += eps

    data = np.stack([t, x], axis=1)
    np.savetxt("./data/rt_example2.csv", data, delimiter=",")

    fig = plt.figure()
    plt.plot(t, x)
    plt.xlim(0, DURATION)
    plt.savefig("./figures/rt_example2.jpg")

    print("Done")


if __name__ == "__main__":
    main()
