"""Estimate per‑neuron linear sensitivity from isolated perturbation runs.

This script consumes results produced by `network_ch` in **isolation mode** with
systematic external perturbations applied to E and I populations. For each
neuron, it computes a simple **linear sensitivity** to the external input by
fitting a cubic polynomial to binned spike counts vs. perturbation amplitude
and taking the **linear coefficient** of that fit.

Overview
--------
1) Load spikes and metadata from `<data_path>/lognormal/{system}_perturbation{patterns}.h5`.
2) Bin spikes (100 ms by default) → shape `(N, T)` for each population.
3) Reshape/average bins to aggregate repeated trials and segregate E‑ and I‑
   driven perturbation blocks.
4) For each neuron, fit `polyfit(amplitude, response, deg=3)` separately for
   E‑driven and I‑driven responses; store:
   - `activation_exc` → linear term of the E‑fit,
   - `activation_inh` → linear term of the I‑fit,
   - `rate`           → mean of intercepts from both fits.
5) Write a CSV to `<data_path>/lognormal/linear_approx/{system}{patterns}.csv`
   with a MultiIndex `(postsynaptic, n_index)`.

Notes
-----
- Logging writes per‑process logs to `logs/<pid>_linapprox.log`.
"""

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
from utils import *  # data_path, memory_usage, etc.

# Set up logging to a per‑process file (useful when running via multiprocessing)
process_id = os.getpid()
log_filename = f"logs/{process_id}_linapprox.log"
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def get_activations(system, patterns, namespace, remaining_tasks):
    """Compute per‑neuron sensitivity metrics for one `(system, patterns)` job.

    Parameters
    ----------
    system : str
        Network model (hebb/rate/...).
    patterns : int
        Number of patterns.
    remaining_tasks : int
        For logging only; indicates how many tasks are left after this one.

    Side effects
    ------------
    Writes a CSV to `<data_path>/lognormal/linear_approx/{system}{patterns}.csv`
    with columns: `postsynaptic`, `n_index`, `rate`, `activation_exc`, `activation_inh`.
    The index is set to `(postsynaptic, n_index)`.
    """
    logging.info(f'Starting {system} with {patterns} patterns')
    folder_path = data_path(namespace)
    filename = f"{folder_path}/{system}_perturbation{patterns}.h5"

    # Load excitatory/inhibitory spikes and metadata
    with h5py.File(filename, "r") as h5f:
        spikes_exc = h5f['spikes_exc'][:]
        spikes_inh = h5f['spikes_inh'][:]
        max_t = h5f.attrs['simulation_time']
        N_exc = h5f['connectivity'].attrs['N_exc']
        N_inh = h5f['connectivity'].attrs['N_inh']

    # Temporal binning factor (kept as in original; 1 → use 100 ms bins)
    factor = 1

    # Bin spikes into counts per neuron per bin
    # dt = 0.1 / factor ⇒ 100 ms when factor=1; N ensures proper neuron binning
    _, sc_exc = get_spike_counts(*spikes_exc.T, max_t, dt=0.1/factor, N=N_exc)
    _, sc_inh = get_spike_counts(*spikes_inh.T, max_t, dt=0.1/factor, N=N_inh)

    # Accumulators for results
    results = {
        'postsynaptic': [],  # 'exc' or 'inh' target population label
        'n_index': [],       # neuron index within its population
        'rate': [],          # baseline rate (intercept averaged across E/I fits)
        'activation_exc': [],  # linear sensitivity to E‑driven input
        'activation_inh': []   # linear sensitivity to I‑driven input
    }

    # For each postsyn population, reshape trial structure and fit response curves
    for sc, N, ei in zip([sc_exc, sc_inh], [N_exc, N_inh], ['exc','inh']):
        # The reshape to `(..., 40*factor)` groups bins into blocks corresponding
        # to one perturbation sweep; slicing `[ :, 1::2, : ]` vs `[ :, 2::2, : ]`
        # selects E‑driven vs I‑driven blocks, respectively. Then mean over trials
        # (axis=1) and decimate every 5*factor to match the 8 amplitudes in `xx`.
        averaged_counts_exc = sc.reshape(N, -1, 40*factor)[:,1::2,:].mean(axis=1)[:,::5*factor]
        averaged_counts_inh = sc.reshape(N, -1, 40*factor)[:,2::2,:].mean(axis=1)[:,::5*factor]

        # Total number of repeated trials per neuron (derived from dimensions)
        n_trials = sc.size / (40*factor*N)

        # Amplitude grid used during isolated perturbations
        xx = np.linspace(-0.35, 0.35, 8)

        for neuron_ix in range(N):
            # Responses vs amplitude for this neuron
            yy_exc = averaged_counts_exc[neuron_ix]
            yy_inh = averaged_counts_inh[neuron_ix]

            # Fit cubic: highest‑order term first; linear term is coefs[-2], intercept coefs[-1]
            coefs_exc = np.polyfit(xx, yy_exc, deg=3)
            rate_exc = coefs_exc[-1]
            results['activation_exc'].append(coefs_exc[-2])

            coefs_inh = np.polyfit(xx, yy_inh, deg=3)
            rate_inh = coefs_inh[-1]
            results['activation_inh'].append(coefs_inh[-2])

            # Average intercept from both fits as the baseline rate estimate
            results['rate'].append((rate_exc+rate_inh)/2)

        # Bookkeeping for index columns
        results['postsynaptic'].extend([ei]*N)
        results['n_index'].extend(np.arange(N))

    logging.info(f'Completed {system} with {patterns} patterns and {n_trials} trials. {memory_usage()}')
    logging.info(f'Remaining tasks: {remaining_tasks - 1}')

    # Convert lists to arrays for consistent dtypes in the DataFrame
    results['rate'] = np.array(results['rate'])
    results['activation_exc'] = np.array(results['activation_exc'])
    results['activation_inh'] = np.array(results['activation_inh'])

    # Persist to CSV with a MultiIndex
    pd.DataFrame(results).set_index(['postsynaptic','n_index']).to_csv(
        f"{folder_path}/linear_approx/{system}{patterns}.csv"
    )


def process_task(args):
    """Thin wrapper for pool.map (unpacks a tuple into `get_activations`)."""
    system, npat, namespace, remaining_tasks = args
    get_activations(system, npat, namespace, remaining_tasks)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--system', type=str, required=True)
    parser.add_argument('--npat', type=int, required=True)
    parser.add_argument('--namespace', type=str, default='lognormal')

    args = parser.parse_args()

    systems = [args.system]

    pat_counts = [
        args.npat
    ]

    # Create a job list with a countdown in `remaining_tasks` for logging
    tasks = [
        (system, npat, args.namespace, len(systems) * len(pat_counts) - i)
        for i, (system, npat) in enumerate([(s, p) for p in pat_counts for s in systems])
    ]

    logging.info('Starting multiprocessing execution')
    with multiprocessing.Pool(processes=8) as pool:
        pool.map(process_task, tasks)
    logging.info('Multiprocessing execution completed')
