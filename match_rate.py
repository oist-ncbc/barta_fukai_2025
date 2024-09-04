from network import run_n_save

import numpy as np
import argparse
import pickle
from copy import deepcopy


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--train', type=str)
    parser.add_argument('-o', '--output', type=str)
    parser.add_argument('-m', '--matrix', type=str)
    parser.add_argument('-t', '--time', type=float)

    args = parser.parse_args()

    with open(args.train, 'rb') as file:
        results = pickle.load(file)

    exc_ix, exc_t = results['spikes']['exc']
    rate_calc_time = 10
    burn = results['simulation_params']['simulation_time'] - rate_calc_time
    tot_spikes = (exc_t > burn).sum()
    rate = tot_spikes / rate_calc_time / 8000
    print(rate)

    results['simulation_params']['heterosynaptic'] = True
    results['simulation_params']['target_rate'] = rate

    if args.time is not None:
        results['simulation_params']['simulation_time'] = args.time

    run_n_save(simulation_params=results['simulation_params'], args=args, matrix_file=results['params']['input'])