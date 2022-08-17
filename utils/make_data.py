# import pdb; pdb.set_trace()
import argparse
import os
import json
import numpy as np
import matplotlib.pyplot as plt


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

    dir_name = os.path.splitext(os.path.basename(args.conf_json))[0]
    data_dir = f"./data/{dir_name}"
    fig_dir = f"./figures/{dir_name}"

    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)
    if not os.path.isdir(fig_dir):
        os.makedirs(fig_dir)

    with open(args.conf_json, "r") as f:
        confs = json.load(f)

    count = 0

    y_max = 0
    for conf in confs:
        amp_mus = conf["amp_mus"]
        amp_devs = conf["amp_devs"]
        y_max = max(y_max, np.sum(amp_mus + amp_devs * 3))

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

        fig = plt.figure()

        n_swav = len(amp_mus)

        t = np.linspace(0, dur, n_samp)
        for _ in range(n_data):
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
            np.savetxt(f"{data_dir}/{count:04}.csv", data, delimiter=",")

            plt.plot(t, x)
            plt.ylim(-y_max, y_max)
            plt.savefig(f"{fig_dir}/{count:04}.jpg")
            plt.cla()

            count += 1

    print("Done")


if __name__ == "__main__":
    main()
