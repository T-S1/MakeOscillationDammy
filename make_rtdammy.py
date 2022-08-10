# import pdb; pdb.set_trace()
import argparse
import os
import json
import numpy as np
import matplotlib.pyplot as plt


def calc_shift = 


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "rtconf_json",
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

    dir_name = os.path.splitext(os.path.basename(args.rtconf_json))[0]
    data_dir = f"./data/{dir_name}_rt"
    fig_dir = f"./figures/{dir_name}_rt"

    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)
    if not os.path.isdir(fig_dir):
        os.makedirs(fig_dir)

    with open(args.rtconf_json, "r") as f:
        rtconf = json.load(f)

    conf_json = rtconf["ref_json"]
    dur = rtconf["duration"]
    n_samp = rtconf["n_samples"]
    n_clus = rtconf["n_clusters"]
    c_dur_mu = rtconf["change_dur"]
    c_dur_dev = rtconf["c_dur_dev"]
    c_probs = rtconf["c_probs"]

    
    with open(args.conf_json, "r") as f:
        confs = json.load(f)

    cum_dist = np.zeros(n_clus)
    cum_dist[0] = c_probs[0]
    for i in range(1, n_clus):
        cum_dist[i] = cum_dist[i-1] + c_probs[i]

    t_change = np.random.normal(c_dur_mu, c_dur_dev)
    rand_var = np.random.uniform(0, 1)
    for j, th_rand in enumerate(cum_dist):
        if rand_var < th_rand:
            conf = confs[j]
            name = conf["name"]
            n_data = conf["n_data"]
            n_samp = conf["n_samp"]
            dur = conf["dur"]
            noise_dev = conf["noise_dev"]
            amp_mus = conf["amp_mus"]
            amp_devs = conf["amp_devs"]
            freq_mus = conf["freq_mus"]
            freq_devs = conf["freq_devs"]
            break

    x = np.zeros(n_samp)
    for i in range(n_samp):
        t = dur * i / n_samp

        if t > t_change:
            rand_var = np.random.uniform(0, 1)

            for j, th_rand in enumerate(cum_dist):
                if rand_var < th_rand:
                    conf = confs[j]
                    name = conf["name"]
                    n_data = conf["n_data"]
                    n_samp = conf["n_samp"]
                    noise_dev = conf["noise_dev"]
                    amp_mus = conf["amp_mus"]
                    amp_devs = conf["amp_devs"]
                    freq_mus = conf["freq_mus"]
                    freq_devs = conf["freq_devs"]

                    a = np.random.normal(amp_mus, amp_devs)
                    f = np.random.normal(freq_mus, freq_devs)
                    break

            t_change += np.random.normal(c_dur_mu, c_dur_dev)

        x[i] = a * np.sin(2 * np.pi * f * t + )

    count = 0

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
            plt.savefig(f"{fig_dir}/{count:04}.jpg")
            plt.cla()

            count += 1

    print("Done")


if __name__ == "__main__":
    main()
