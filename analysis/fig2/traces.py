import h5py
import numpy as np

from utils import data_path, load_patterns
from analysis import get_spike_counts


if __name__ == '__main__':
    folder = 'lognormal'
    npat = 1800
    system = 'hebb'

    path_to_folder = f"{data_path()}/{folder}"
    filename = f"{path_to_folder}/{system}_spontaneous{npat}.h5"

    rates_exc = []
    rates_inh = []

    with h5py.File(filename, 'r', swmr=True) as h5f:
        N_exc = h5f['connectivity'].attrs['N_exc']
        N_inh = h5f['connectivity'].attrs['N_inh']

        spikes_exc = h5f['spikes_exc'][:].T
        spikes_inh = h5f['spikes_inh'][:].T

    _, sc = get_spike_counts(*spikes_exc, 21, dt=0.01, offset=0)
    sc = sc[:,1500:]

    run = 'spontaneous'
    system = 'hebb'

    patterns = load_patterns(npat)
    folder = f"{data_path()}/lognormal"
    filename = f"{folder}/{system}_{run}{npat}_activations.h5"

    with h5py.File(filename, "r", swmr=True) as h5f:
        act_times, durations, pattern_ixs = h5f['activations'][:]

    act_times = act_times / 100
    mask = (act_times > 5) & (act_times < 10)

    pattern_ix_list = np.unique(pattern_ixs[mask])

    pattern_rates = []
    pattern_activations = []

    for i, patix in enumerate(pattern_ixs[mask]):
        pattern = patterns[int(patix)]

        pattern_rates.append((sc[pattern]).mean(axis=0)[:500]*100)

        act_trace = np.zeros(500)

        for i in range(10):
            act_trace[i::10] = (sc[patterns[int(patix)]][:,i:500+i].reshape(len(pattern), 50, 10).sum(axis=2) > 0).mean(axis=0)

        pattern_activations.append(act_trace)

    np.savetxt('plotting/data/assembly_traces/rates.csv', pattern_rates)
    np.savetxt('plotting/data/assembly_traces/activations.csv', pattern_activations)