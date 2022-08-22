# import pdb; pdb.set_trace()
import argparse
import numpy as np
import matplotlib.pyplot as plt

DURATION = 100
N_SAMPLES = 2560
FREQ = 1
AMP_INIT = 1
AMP_END = 2.5
NOISE_DEV = 0.1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed",
        help="seed value",
        type=int, default=100
    )
    args = parser.parse_args()

    np.random.seed(args.seed)

    slope = (AMP_END - AMP_INIT) / DURATION
    t = np.linspace(0, DURATION, N_SAMPLES)
    eps = np.random.normal(0, NOISE_DEV, N_SAMPLES)
    x = (slope * t + AMP_INIT) * np.sin(2 * np.pi * FREQ * t) + eps

    data = np.stack([t, x], axis=1)
    np.savetxt("./data/rt_example1.csv", data, delimiter=",")

    fig = plt.figure()
    plt.plot(t, x)
    plt.xlim(0, DURATION)
    plt.savefig("./figures/rt_example1.jpg")

    print("Done")


if __name__ == "__main__":
    main()
