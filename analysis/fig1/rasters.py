import numpy as np
from scipy import sparse
import h5py
from tqdm import tqdm
import pickle

from utils import load_patterns, data_path
from analysis import get_spike_counts
from eigenvalues import get_W


def load_spikes(system, npat, start, end, namespace):
    folder = data_path(namespace)
    filename = f"{folder}/{system}_spontaneous{npat}.h5"

    with h5py.File(filename, "r") as h5f:
        exc_indices, exc_times = h5f['spikes_exc'][:8000*4*end].T
        inh_indices, inh_times = h5f['spikes_inh'][:2000*15*end].T

    mask_exc = (exc_times >= start) & (exc_times < end)
    spikes_exc = np.array(([exc_indices[mask_exc], exc_times[mask_exc]]))

    mask_inh = (inh_times >= start) & (inh_times < end)
    spikes_inh = np.array([inh_indices[mask_inh], inh_times[mask_inh]])

    return spikes_exc, spikes_inh


if __name__ == '__main__':
    namespace = 'lognormal'
    for system in tqdm(['rate','hebb_smooth_rate','hebb']):
        spikes_exc, spikes_inh = load_spikes(system, 1000, 0, 20, namespace)

        np.savetxt(f'plotting/data/rasters/{system}_excitatory.csv', spikes_exc)
        np.savetxt(f'plotting/data/rasters/{system}_inhibitory.csv', spikes_inh)