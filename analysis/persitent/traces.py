import h5py
import numpy as np
from tqdm import tqdm
import pandas as pd

from utils import data_path, load_patterns
from analysis import get_spike_counts


def load_pr(system, run, namespace):
    npat = 1000

    path_to_folder = data_path(namespace)
    filename = f"{path_to_folder}/{system}_{run}{npat}.h5"


    with h5py.File(filename, 'r', swmr=True) as h5f:
        N_exc = h5f['connectivity'].attrs['N_exc']
        N_inh = h5f['connectivity'].attrs['N_inh']

        spikes_exc = h5f['spikes_exc'][:].T
        spikes_inh = h5f['spikes_exc'][:].T
        max_t = h5f.attrs['simulation_time']

    patterns = load_patterns(npat, namespace=namespace)

    nstim = min(int(max_t-1), npat)

    _, sc = get_spike_counts(*spikes_exc, max_t, N_exc, dt=0.01)

    pattern_rates = []
    # pattern_acts = []

    for ix in tqdm(range(npat)):
        pattern_rates.append(sc[patterns[ix]].mean(axis=0))
        # pattern_acts.append((sc[patterns[ix]] > 0).mean(axis=0))

    pattern_rates = np.array(pattern_rates)
    # pattern_acts = np.array(pattern_acts)

    return pattern_rates, nstim


if __name__ == '__main__':
    namespace = 'lognormal'

    pattern_rates, _ = load_pr('hebb_nonadapt', 'stimulus100ms_persist', namespace=namespace)

    xx = np.arange(len(pattern_rates[0])) * 0.01
    mask = (xx >= 10.5) & (xx < 22)

    export_persistent = {
        'xx': xx[mask]
    }

    for ix in [1, 2, 470, 979, 618, 235, 877, 785, 268]:
        export_persistent[ix] = pattern_rates[ix][mask]

    export_persistent['best'] = np.delete(pattern_rates, [1,2], axis=0).max(axis=0)[mask]

    pd.DataFrame(export_persistent).to_csv('plotting/data/persistent/persistent_traces.csv', index=False)

    patr_list = []
    best_list = []

    for ix in range(23):
        rate_slice = pattern_rates[:,ix*1000:ix*1000+500]
        patr_list.append(rate_slice[ix])
        best_list.append(np.delete(rate_slice, ix, axis=0).max(axis=0))

    compare_others = {
        'stimulated': np.mean(patr_list, axis=0),
        'other': np.mean(best_list, axis=0)
    }
    
    pd.DataFrame(compare_others).to_csv('plotting/data/persistent/averaged.csv', index=False)


    pattern_rates, nstim = load_pr('hebb_smooth_rate', 'stimulus100ms_full', namespace=namespace)

    patr_list = []
    best_list = []

    for ix in tqdm(range(nstim-5)):
        rate_slice = pattern_rates[:,ix*100+50:ix*100+500]
        patr_list.append(rate_slice[ix])
        best_list.append(np.delete(rate_slice, ix, axis=0).max(axis=0))

    compare_others_sfa = {
        'stimulated': np.mean(patr_list, axis=0),
        'other': np.mean(best_list, axis=0)
    }
    
    pd.DataFrame(compare_others_sfa).to_csv('plotting/data/persistent/averaged_sfa.csv', index=False)