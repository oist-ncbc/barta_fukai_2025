import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
import pickle

from utils import *


def get_spike_counts(spike_indices, spike_times, t_max, N=8000, dt=0.1, offset=0):
    bins_indices = np.arange(-0.5, N, 1) + offset
    bins_time = np.arange(0, t_max+dt/10, dt) + offset*dt

    mask = spike_times < t_max

    histdata, *_ = np.histogram2d(spike_indices[mask], spike_times[mask], [bins_indices, bins_time])

    return bins_time, histdata

def get_firing_rates(system, npat, folder='lognormal', interval=None, which='ei'):
    path_to_folder = f"{data_path()}/{folder}"
    filename = f"{path_to_folder}/{system}_spontaneous{npat}.h5"

    rates_exc = []
    rates_inh = []

    with h5py.File(filename, 'r', swmr=True) as h5f:
        N_exc = h5f['connectivity'].attrs['N_exc']
        N_inh = h5f['connectivity'].attrs['N_inh']

        spikes_exc = h5f['spikes_exc'][:].T
        spikes_inh = h5f['spikes_exc'][:].T
        max_t = h5f.attrs['simulation_time']

    if interval is None:
        interval = (0, max_t)

    rates = {}

    if 'e' in which:
        mask = (spikes_exc[1] >= interval[0]) & (spikes_exc[1] < interval[1])
        rates['exc'] = pd.Series(spikes_exc[0][mask]).value_counts().sort_index().values / max_t
    if 'i' in which:
        mask = (spikes_inh[1] >= interval[0]) & (spikes_inh[1] < interval[1])
        rates['inh'] = pd.Series(spikes_inh[0][mask]).value_counts().sort_index().values / max_t

    return rates

def get_mean_exc(system, npat, folder='lognormal', interval=None, which='ei'):
    path_to_folder = f"{data_path()}/{folder}"
    filename = f"{path_to_folder}/{system}_spontaneous{npat}.h5"

    with h5py.File(filename, 'r', swmr=True) as h5f:
        N_exc = h5f['connectivity'].attrs['N_exc']
        # N_inh = h5f['connectivity'].attrs['N_inh']

        num_exc_spikes = h5f['spikes_exc'].shape[0]
        # spikes_inh = h5f['spikes_exc'][:].T
        max_t = h5f.attrs['simulation_time']

    return (num_exc_spikes / max_t / N_exc).item()
    # if interval is None:
    #     interval = (0, max_t)

    # rates = {}

    # if 'e' in which:
    #     mask = (spikes_exc[1] >= interval[0]) & (spikes_exc[1] < interval[1])
    #     rates['exc'] = pd.Series(spikes_exc[0][mask]).value_counts().sort_index().values / max_t
    # if 'i' in which:
    #     mask = (spikes_inh[1] >= interval[0]) & (spikes_inh[1] < interval[1])
    #     rates['inh'] = pd.Series(spikes_inh[0][mask]).value_counts().sort_index().values / max_t

    # return rates

def get_pattern_activations(spike_counts, patterns):
    activations = []

    for pattern in patterns:
        activations.append((spike_counts[pattern.astype(bool)] > 0).mean(axis=0))

    return np.array(activations)

def get_correlations(patterns, spike_counts, N_exc=8000, dead=100):
    pattern_correlations = {}

    pattern_correlations['pat'] = []
    pattern_correlations['rnd'] = []

    for pattern in tqdm(patterns):
        corrs = np.corrcoef(spike_counts[pattern.astype(bool)][:,dead:])
        indices = np.triu_indices_from(corrs, k=1)
        pattern_correlations['pat'].append(np.nanmean(corrs[indices]))

        rand_pattern = np.random.permutation(N_exc)[:int(pattern.sum())]
        corrs = np.corrcoef(spike_counts[rand_pattern][:,dead:])
        indices = np.triu_indices_from(corrs, k=1)
        pattern_correlations['rnd'].append(np.nanmean(corrs[indices]))

    pattern_correlations['pat'] = np.array(pattern_correlations['pat'])
    pattern_correlations['rnd'] = np.array(pattern_correlations['rnd'])

    return pattern_correlations


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-r', '--results', type=str, nargs='+')
    parser.add_argument('-f', '--force', action='store_true')
    parser.add_argument('-t', '--max_time', type=float)
    parser.add_argument('--dt', type=float, default=0.1)
    parser.add_argument('--img', type=str)

    args = parser.parse_args()

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 6))

    for ii, res_file in enumerate(args.results):
        with open(res_file, 'rb') as file:
            results = pickle.load(file)

        mat_file = results['params']['input']

        with open(mat_file, 'rb') as file:
            Z, N_exc, patterns, exc_alpha, delays, params = pickle.load(file)

        ix, t = results['spikes']['exc']
        ix_inh, t_inh = results['spikes']['inh']

        if args.max_time is None:
            if 'simulation_time' in results.keys():
                max_time = results['simulation_time']
            else:
                max_time = results['params']['time']
        else:
            max_time = args.max_time

        if args.force or 'analysis' not in results.keys():
            _count_times, spike_counts = get_spike_counts(t, ix, max_time, dt=args.dt)
            count_times = _count_times[:-1]
            results['analysis'] = {}
            results['analysis']['t'] = count_times
            results['analysis']['spike_counts'] = spike_counts

            _count_times_inh, spike_counts_inh = get_spike_counts(t_inh, ix_inh, max_time, dt=args.dt, N=2000)
            count_times_inh = _count_times_inh[:-1]
            results['analysis']['t_inh'] = count_times_inh
            results['analysis']['spike_counts_inh'] = spike_counts_inh

            activations = get_pattern_activations(spike_counts, patterns)
            correlations = get_correlations(patterns, spike_counts)

            results['analysis']['activations'] = activations
            results['analysis']['correlations'] = correlations

            with open(res_file, 'wb') as file:
                pickle.dump(results, file)
        else:
            spike_counts = results['analysis']['spike_counts']
            count_times = results['analysis']['t']

            activations = results['analysis']['activations']
            correlations = results['analysis']['correlations']

        bins = np.linspace(0, 10, 50)
        bins = np.logspace(-2, 2, 50)
        axes[0,0].hist(spike_counts.mean(axis=1)*10, bins=bins, histtype='step')
        axes[0,0].axvline(spike_counts.mean()*10, color=f'C{ii}')
        axes[0,0].set_xlabel('neuron firing rate (Hz)')
        axes[0,0].set_xscale('log')

        bins = np.linspace(-0.02, 0.2, 50)
        axes[0,1].hist(correlations['pat'], bins=bins, histtype='step')
        axes[0,1].axvline(correlations['rnd'].mean(), color=f'C{ii}')
        axes[0,1].set_xlabel('within-pattern correlation')

        bins = np.linspace(-0.01, 0.01, 30)
        axes[1,1].hist(correlations['rnd'], bins=bins, histtype='step')
        axes[1,1].axvline(correlations['rnd'].mean(), color=f'C{ii}')
        axes[1,1].set_xlabel('within-rand.pattern correlation')

        axes[1,0].plot(count_times[:], 100*(np.cumsum(activations[:,:] > 0.9, axis=1) > 0).mean(axis=0))
        axes[1,0].set_xlabel('time (s)')
        axes[1,0].set_ylabel('% of patterns activated')

        # axes[1,0].set_xlim(0, 200)

    fig.tight_layout()
    plt.savefig(args.img)