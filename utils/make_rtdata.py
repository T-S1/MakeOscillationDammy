# import pdb; pdb.set_trace()
import argparse
import os
import json
import numpy as np
import matplotlib.pyplot as plt


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
    colors = ["g", "r", "b"]

    with open(args.rtconf_json, "r") as f:
        rtconf = json.load(f)

    conf_json = rtconf["ref_json"]
    dur = rtconf["duration"]
    n_samp = rtconf["n_samples"]
    n_clus = rtconf["n_clusters"]
    c_dur_mu = rtconf["change_dur"]
    c_dur_dev = rtconf["c_dur_dev"]
    c_probs = rtconf["c_probs"]

    with open(conf_json, "r") as f:
        confs = json.load(f)

    cum_dist = np.zeros(n_clus)
    cum_dist[0] = c_probs[0]
    for i in range(1, n_clus):
        cum_dist[i] = cum_dist[i-1] + c_probs[i]

    fig = plt.figure()

    t_change = np.random.normal(c_dur_mu, c_dur_dev)
    phi = np.random.uniform(0, 2 * np.pi)
    rand_var = np.random.uniform(0, 1)
    for j, th_rand in enumerate(cum_dist):
        if rand_var < th_rand:
            conf = confs[j]
            name = conf["name"]
            noise_dev = conf["noise_dev"]
            amp_mus = conf["amp_mus"]
            amp_devs = conf["amp_devs"]
            freq_mus = conf["freq_mus"]
            freq_devs = conf["freq_devs"]

            amps = np.random.normal(amp_mus, amp_devs)
            freqs = np.random.normal(freq_mus, freq_devs)

            plt.axvspan(0, t_change, color=colors[j])
            break

    ts = np.zeros(n_samp)
    x = np.zeros(n_samp)
    for i in range(n_samp):
        t = dur * i / n_samp

        if t > t_change:
            rand_var = np.random.uniform(0, 1)
            t_change += np.random.normal(c_dur_mu, c_dur_dev)

            for j, th_rand in enumerate(cum_dist):
                if rand_var < th_rand:
                    conf = confs[j]
                    name = conf["name"]
                    noise_dev = conf["noise_dev"]
                    amp_mus = conf["amp_mus"]
                    amp_devs = conf["amp_devs"]
                    freq_mus = conf["freq_mus"]
                    freq_devs = conf["freq_devs"]

                    amps = np.random.normal(amp_mus, amp_devs)
                    freqs = np.random.normal(freq_mus, freq_devs)

                    plt.axvspan(t, t_change, color=colors[j])
                    break

        eps = np.random.normal(0, noise_dev)
        for a, f in zip(amps, freqs):
            x[i] += a * np.sin(2 * np.pi * f * t + phi)
        x[i] += eps
        ts[i] = t

    bname = os.path.splitext(os.path.basename(args.rtconf_json))[0]

    data = np.stack([ts, x], axis=1)
    np.savetxt(f"./data/{bname}.csv", data, delimiter=",")

    plt.plot(ts, x, color="k")
    plt.xlim(ts[0], ts[-1])
    plt.savefig(f"./figures/{bname}.jpg")

    print("Done")


if __name__ == "__main__":
    main()
