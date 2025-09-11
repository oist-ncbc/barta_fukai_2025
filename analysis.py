"""Analysis utilities for spike trains, activations, and pattern statistics.

This module contains helper functions used throughout the project to:

- Load activation events saved by `get_activations.py`.
- Compute per‑pattern activation counts.
- Bin spikes into time windows (100 ms by default) for downstream
  analyses;
- Estimate excitatory/inhibitory firing rates from simulation HDF5 files.
- Compute Fano factors and mean rates.
- Derive activation traces for provided patterns, and pairwise correlation
  summaries inside a pattern (vs. random baselines).
"""

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import *  # brings in data_path(), possibly h5py via utils, etc.

import logging
logger = logging.getLogger(__name__)


def load_activation(system, npat, run='spontaneous', namespace='lognormal'):
    """Load activation events produced by `get_activations.py`.

    Parameters
    ----------
    system : str
        System identifier used in the filename (e.g., "hebb").
    npat : int
        Number of patterns.
    run : str, default 'spontaneous'
        Run tag used in the filename.
    namespace : str, default 'lognormal'
        Data folder (namespace) under the configured `data_path()`.

    Returns
    -------
    tuple of numpy.ndarray
        `(act_times, durations, pattern_ixs)` taken from the HDF5 dataset
        `'activations'`. All arrays are 1‑D and aligned element‑wise.
    """
    folder = f"{data_path()}/{namespace}"
    filename = f"{folder}/{system}_{run}{npat}_activations.h5"

    # NOTE: h5py is referenced here; make sure it's available in the runtime.
    with h5py.File(filename, "r", swmr=True) as h5f:  # type: ignore[name-defined]
        act_times, durations, pattern_ixs = h5f['activations'][:]

    return act_times, durations, pattern_ixs


def get_act_counts(system, npat):
    """Return the number of activation events per pattern.

    This is computed by loading `(act_times, durations, pattern_ixs)` and
    counting occurrences of each pattern index in `pattern_ixs`.
    """
    act_times, _, pattern_ixs = load_activation(system, npat)
    act_counts = pd.Series(pattern_ixs).value_counts().reindex(np.arange(npat), fill_value=0).values
    return act_counts


def get_spike_counts(spike_indices, spike_times, t_max, N=8000, dt=0.1, offset=0):
    """Bin spikes into time windows (2D histogram: neuron × time bin).

    Parameters
    ----------
    spike_indices : (M,) array_like
        Neuron ids for each spike (0..N-1).
    spike_times : (M,) array_like
        Spike times in seconds.
    t_max : float
        Upper bound (exclusive) for the analysis window `[0, t_max)`.
    N : int, default 8000
        Total number of excitatory neurons (defines the index bin edges).
    dt : float, default 0.1
        Bin length (seconds). `0.1` ⇒ 100 ms windows.
    offset : float, default 0
        Phase offset applied to the time bin edges **in units of `dt`** because
        the implementation subtracts `offset * dt`. For example, with `dt=0.1`:
        `offset=0.1` shifts the window grid by `0.01 s` (10 ms).

    Returns
    -------
    tuple
        `(bins_time, histdata)` where:
        - `bins_time` is the array of time bin edges (shifted by `offset * dt`),
        - `histdata` is a 2D array of spike counts with shape `(N, T)`.
    """
    # Neuron index bins: centers on integers by using edges at i±0.5
    bins_indices = np.arange(-0.5, N, 1)

    # Time bin edges from 0..t_max; add a small `dt/10` to include the rightmost edge
    # and shift by `offset * dt` to realize the desired phase.
    bins_time = np.arange(0, t_max+dt/10, dt) - offset*dt

    # Consider only spikes that lie strictly before `t_max` to avoid a trailing edge effect
    mask = spike_times < t_max

    # 2D histogram: rows=indices, cols=time bins
    histdata, *_ = np.histogram2d(spike_indices[mask], spike_times[mask], [bins_indices, bins_time])

    return bins_time, histdata


def get_firing_rates(system, npat, folder='lognormal', interval=None, which='ei'):
    """Compute per‑population firing rates from a spontaneous simulation file.

    Parameters
    ----------
    system : str
        System identifier (filename prefix).
    npat : int
        Number of patterns used in the simulation filename.
    folder : str, default 'lognormal'
        Data namespace folder under `data_path()`.
    interval : tuple[float, float] | None
        If `None`, use the full `[0, max_t)`; otherwise, restrict to `(t0, t1)`.
    which : str, default 'ei'
        Select populations: any subset of `'e'` and `'i'`.

    Returns
    -------
    dict
        Keys `'exc'` and/or `'inh'` (depending on `which`), each a 1‑D array of
        per‑neuron firing rates (Hz) estimated as **spikes / max_t**.
    """
    path_to_folder = f"{data_path()}/{folder}"
    filename = f"{path_to_folder}/{system}_spontaneous{npat}.h5"

    logger.info(f"Loading data from {filename}")

    # (Unused placeholders preserved from original code)
    rates_exc = []
    rates_inh = []

    with h5py.File(filename, 'r', swmr=True) as h5f:  # type: ignore[name-defined]
        N_exc = h5f['connectivity'].attrs['N_exc']
        N_inh = h5f['connectivity'].attrs['N_inh']
        spikes_exc = h5f['spikes_exc'][:].T  # shape (2, M)
        spikes_inh = h5f['spikes_inh'][:].T  # shape (2, M)
        max_t = h5f.attrs['simulation_time']

    logger.info("Data loaded successfully.")

    if interval is None:
        interval = (0, max_t)

    rates = {}

    if 'e' in which:
        logger.info("Calculating excitatory firing rates...")
        mask = (spikes_exc[1] >= interval[0]) & (spikes_exc[1] < interval[1])
        rates['exc'] = pd.Series(spikes_exc[0][mask]).value_counts().reindex(np.arange(N_exc), fill_value=0).values / max_t
        logger.info("Excitatory rates calculated.")

    if 'i' in which:
        logger.info("Calculating inhibitory firing rates...")
        mask = (spikes_inh[1] >= interval[0]) & (spikes_inh[1] < interval[1])
        rates['inh'] = pd.Series(spikes_inh[0][mask]).value_counts().reindex(np.arange(N_inh), fill_value=0).values / max_t
        logger.info("Inhibitory rates calculated.")

    return rates


def get_fano(system, npat, folder='lognormal', interval=None):
    """Compute Fano factors (variance/mean of counts) for excitatory neurons.

    Parameters
    ----------
    system, npat, folder : see `get_firing_rates` for filename construction.
    interval : float | tuple | None
        If not `None`, **overrides** `max_t` (uses that horizon for binning).

    Returns
    -------
    numpy.ndarray
        1‑D array of Fano factors per excitatory neuron based on 100 ms bins.
    """
    path_to_folder = f"{data_path()}/{folder}"
    filename = f"{path_to_folder}/{system}_spontaneous{npat}.h5"

    with h5py.File(filename, 'r', swmr=True) as h5f:  # type: ignore[name-defined]
        N_exc = h5f['connectivity'].attrs['N_exc']
        spikes_exc = h5f['spikes_exc'][:].T
        max_t = h5f.attrs['simulation_time']

    if interval is not None:
        max_t = interval

    # Bin excitatory spikes with 100 ms windows; offset left at default (phase 0)
    _, sc = get_spike_counts(*spikes_exc, max_t, N_exc, dt=0.1)

    # Fano factor per neuron: Var / Mean across time bins
    fanos = sc.var(axis=1) / sc.mean(axis=1)

    return fanos


def get_mean_exc(system, npat, folder='lognormal'):
    """Return the **mean excitatory firing rate** over the entire simulation.

    Computed as: `(total number of excitatory spikes) / (max_t * N_exc)`.
    The optional parameters `interval` and `which` are unused (kept for API symmetry).
    """
    path_to_folder = f"{data_path()}/{folder}"
    filename = f"{path_to_folder}/{system}_spontaneous{npat}.h5"

    with h5py.File(filename, 'r', swmr=True) as h5f:
        N_exc = h5f['connectivity'].attrs['N_exc']
        num_exc_spikes = h5f['spikes_exc'].shape[0]
        max_t = h5f.attrs['simulation_time']

    return (num_exc_spikes / max_t / N_exc).item()


def get_correlations(patterns, spike_counts, N_exc=8000, dead=100):
    """Estimate mean pairwise correlations within patterns vs. random baselines.

    Parameters
    ----------
    patterns : iterable
        Each element is a boolean mask or integer index list for one pattern.
    spike_counts : (N, T) array_like
        Binned spike counts per neuron (e.g., from `get_spike_counts`).
    N_exc : int, default 8000
        Population size used when drawing random patterns for the baseline.
    dead : int, default 100
        Number of initial time bins to exclude (e.g., discard transients).

    Returns
    -------
    dict
        Dictionary with two entries:
        - `'pat'`: mean pairwise correlation across member neurons within each pattern.
        - `'rnd'`: same metric for size‑matched random neuron subsets.
        Values are stored as 1‑D arrays aligned with `patterns`.
    """
    pattern_correlations = {}

    pattern_correlations['pat'] = []
    pattern_correlations['rnd'] = []

    for pattern in tqdm(patterns):
        # Correlations among *true* pattern members (skip initial `dead` bins)
        corrs = np.corrcoef(spike_counts[pattern.astype(bool)][:,dead:])
        indices = np.triu_indices_from(corrs, k=1)
        pattern_correlations['pat'].append(np.nanmean(corrs[indices]))

        # Correlations among a random subset of equal size (baseline)
        rand_pattern = np.random.permutation(N_exc)[:int(pattern.sum())]
        corrs = np.corrcoef(spike_counts[rand_pattern][:,dead:])
        indices = np.triu_indices_from(corrs, k=1)
        pattern_correlations['rnd'].append(np.nanmean(corrs[indices]))

    pattern_correlations['pat'] = np.array(pattern_correlations['pat'])
    pattern_correlations['rnd'] = np.array(pattern_correlations['rnd'])

    return pattern_correlations
