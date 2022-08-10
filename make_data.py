# import pdb; pdb.set_trace()
import argparse
import os
import json
import numpy as np
import matplotlib.pyplot as plt

FIG_DIR = "./figures"
DATA_DIR = "./data"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "conf_json",
        help="configuration json path",
        type=str
    )
    parser.add_argument(
        "--seed",
        help="seed value",
        type=int, default=100
    )
    args = parser.parse_args()

    np.random.seed(args.seed)

    with open(args.conf_json, "r") as f:
        confs = json.load(f)

    for conf in confs:
        name = conf["name"]
        n_data = conf["n_data"]
        n_samp = conf["n_samp"]
        dur = conf["dur"]
        noise_dev = conf["noise_dev"]
        amp_mus = conf["amp_mus"]
        amp_devs = conf["amp_devs"]
        freq_mus = conf["freq_mus"]
        freq_devs = conf["freq_devs"]

        data_dir = f"{DATA_DIR}/{name}"
        fig_dir = f"{FIG_DIR}/{name}"

        if not os.path.isdir(data_dir):
            os.makedirs(data_dir)
        if not os.path.isdir(fig_dir):
            os.makedirs(fig_dir)

        fig = plt.figure()

        n_swav = len(amp_mus)

        t = np.linspace(0, dur, n_samp)
        for i in range(n_data):
            amps = np.random.normal(amp_mus, amp_devs)
            freqs = np.random.normal(freq_mus, freq_devs)
            phases = np.random.uniform(0, 2 * np.pi, n_swav)
            eps = np.random.normal(0, noise_dev, n_samp)

            x = np.zeros(n_samp)
            for j in range(n_swav):
                a = amps[j]
                f = freqs[j]
                phi = phases[j]

                x += a * np.sin(2 * np.pi * f * t + phi)

            x += eps

            data = np.stack([t, x], axis=1)
            np.savetxt(f"{data_dir}/{i:04}.csv", data, delimiter=",")

            plt.plot(t, x)
            plt.savefig(f"{fig_dir}/{i:04}.jpg")
            plt.cla()

    print("Done")


if __name__ == "__main__":
    main()
