import numpy as np
import h5py
import yaml
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import argparse

from analysis import get_spike_counts


with open('config/server_config.yaml') as f:
    server_config = yaml.safe_load(f)

def get_activations(system, patterns):
    folder = f"{server_config['data_path']}/lognormal"
    filename = f"{folder}/{system}_perturbation{patterns}.h5"

    with h5py.File(filename, "r") as h5f:
        spikes_exc = h5f['spikes_exc'][:]
        spikes_inh = h5f['spikes_inh'][:]
        max_t = h5f.attrs['simulation_time']
        N_exc = h5f['connectivity'].attrs['N_exc']
        N_inh = h5f['connectivity'].attrs['N_inh']
        # print(list(h5f['connectivity'].attrs.keys()))

    _, sc_exc = get_spike_counts(*spikes_exc.T, max_t, dt=0.1, N=N_exc)
    _, sc_inh = get_spike_counts(*spikes_inh.T, max_t, dt=0.1, N=N_inh)

    results = {
        'postsynaptic': [],
        'n_index': [],
        'rate': [],
        'activation_exc': [],
        'activation_inh': []
    }

    for sc, N, ei in zip([sc_exc, sc_inh], [N_exc, N_inh], ['exc','inh']):
        averaged_counts_exc = sc.reshape(N, -1, 40)[:,1::2,:].mean(axis=1)[:,::5]
        averaged_counts_inh = sc.reshape(N, -1, 40)[:,2::2,:].mean(axis=1)[:,::5]

        for neuron_ix in tqdm(range(N)):
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

    results['rate'] = np.array(results['rate'])
    results['activation_exc'] = np.array(results['activation_exc'])
    results['activation_inh'] = np.array(results['activation_inh'])

    pd.DataFrame(results).set_index(['postsynaptic','n_index']).to_csv(f"{folder}/linear_approx/{system}{patterns}.csv")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-s', '--system', type=str)
    parser.add_argument('-p', '--patterns', type=int)

    args = parser.parse_args()

    get_activations(args.system, args.patterns)