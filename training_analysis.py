import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import argparse
import pickle


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-r', '--results', type=str, nargs='+')
    parser.add_argument('--img', type=str)

    args = parser.parse_args()

    fig, ax = plt.subplots(figsize=(6, 4))

    for ii, res_file in enumerate(args.results):
        with open(res_file, 'rb') as file:
            results = pickle.load(file)

        ix, t = results['spikes']['exc']

        bins = np.arange(0, results['simulation_params']['simulation_time']+1e-5, 1)

        counts, _ = np.histogram(t, bins=bins)

        ax.plot(bins[:-1], counts/8000)

    plt.savefig(args.img)