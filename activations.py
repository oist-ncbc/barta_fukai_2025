import numpy as np
import h5py
import yaml
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import argparse
import logging
import os
import multiprocessing

from analysis import get_spike_counts
from utils import *

# Set up logging
process_id = os.getpid()
log_filename = f"logs/{process_id}_linapprox.log"
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

with open('config/server_config.yaml') as f:
    server_config = yaml.safe_load(f)

def get_activations(system, patterns, remaining_tasks):
    logging.info(f'Starting {system} with {patterns} patterns')
    folder = f"{server_config['data_path']}/lognormal"
    filename = f"{folder}/{system}_perturbation{patterns}.h5"

    with h5py.File(filename, "r") as h5f:
        spikes_exc = h5f['spikes_exc'][:]
        spikes_inh = h5f['spikes_inh'][:]
        max_t = h5f.attrs['simulation_time']
        N_exc = h5f['connectivity'].attrs['N_exc']
        N_inh = h5f['connectivity'].attrs['N_inh']

    factor = 1

    _, sc_exc = get_spike_counts(*spikes_exc.T, max_t, dt=0.1/factor, N=N_exc)
    _, sc_inh = get_spike_counts(*spikes_inh.T, max_t, dt=0.1/factor, N=N_inh)

    results = {
        'postsynaptic': [],
        'n_index': [],
        'rate': [],
        'activation_exc': [],
        'activation_inh': []
    }

    for sc, N, ei in zip([sc_exc, sc_inh], [N_exc, N_inh], ['exc','inh']):
        averaged_counts_exc = sc.reshape(N, -1, 40*factor)[:,1::2,:].mean(axis=1)[:,::5*factor]
        averaged_counts_inh = sc.reshape(N, -1, 40*factor)[:,2::2,:].mean(axis=1)[:,::5*factor]

        n_trials = sc.size / (40*factor*N)

        for neuron_ix in range(N):
            xx = np.linspace(-0.35, 0.35, 8)
            yy_exc = averaged_counts_exc[neuron_ix]
            yy_inh = averaged_counts_inh[neuron_ix]

            coefs_exc = np.polyfit(xx, yy_exc, deg=3)
            rate_exc = coefs_exc[-1]
            results['activation_exc'].append(coefs_exc[-2])

            coefs_inh = np.polyfit(xx, yy_inh, deg=3)
            rate_inh = coefs_inh[-1]
            results['activation_inh'].append(coefs_inh[-2])

            results['rate'].append((rate_exc+rate_inh)/2)

        results['postsynaptic'].extend([ei]*N)
        results['n_index'].extend(np.arange(N))

    logging.info(f'Completed {system} with {patterns} patterns and {n_trials} trials. {memory_usage()}')
    logging.info(f'Remaining tasks: {remaining_tasks - 1}')

    results['rate'] = np.array(results['rate'])
    results['activation_exc'] = np.array(results['activation_exc'])
    results['activation_inh'] = np.array(results['activation_inh'])

    pd.DataFrame(results).set_index(['postsynaptic','n_index']).to_csv(f"{folder}/linear_approx/{system}{patterns}.csv")

def process_task(args):
    system, npat, remaining_tasks = args
    get_activations(system, npat, remaining_tasks)

if __name__ == '__main__':
    systems=[
    "hebb_recharge_minus_strong_tr2.0",
    "hebb_recharge_minus_strong_tr2.1",
    "hebb_recharge_minus_strong_tr2.2",
    "hebb_recharge_minus_strong_tr2.3",
    "hebb_recharge_minus_strong_tr2.4",
    "hebb_recharge_minus_strong_tr2.5",
    "hebb_recharge_minus_strong_tr2.6",
    "hebb_recharge_minus_strong_tr2.7",
    "hebb_recharge_plus_strong_tr1.5",
    "hebb_recharge_plus_strong_tr1.6",
    "hebb_recharge_plus_strong_tr1.7",
    "hebb_recharge_plus_strong_tr1.8",
    "hebb_recharge_plus_strong_tr1.9",
    "hebb_recharge_plus_strong_tr2.0",
    "hebb_recharge_plus_strong_tr2.1",
    "hebb_recharge_plus_strong_tr2.2",
    "hebb_recharge_plus_strong_tr2.3",
    "hebb_recharge_plus_strong_tr2.4",
    "hebb_recharge_plus_strong_tr2.5"]

    # systems = ['hebb','rate']

    pat_counts = [
        1000, 1200, 1400, 1600, 1800, 2000
    ]

    tasks = [(system, npat, len(systems) * len(pat_counts) - i) for i, (system, npat) in enumerate([(s, p) for p in pat_counts for s in systems])]

    logging.info('Starting multiprocessing execution')
    with multiprocessing.Pool(processes=8) as pool:
        pool.map(process_task, tasks)
    logging.info('Multiprocessing execution completed')
