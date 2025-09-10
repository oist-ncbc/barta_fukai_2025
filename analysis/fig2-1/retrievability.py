import numpy as np
import pickle
from tqdm import tqdm
import h5py

from analysis import get_spike_counts, get_act_counts
from utils import data_path, load_patterns, create_stim_tuples


if __name__ == '__main__':
    # with open('data/stimuli/frac10pc1000stim.pkl', 'rb') as f:
    #     tuples = pickle.load(f)

    res = {}

    syslist = ['hebb','hebb_smooth_rate','rate']

    for npat in [1000, 1400, 2000, 3000]:
        res[npat] = {}
        for system in tqdm(syslist):
            folder = 'lognormal'

            path_to_folder = f"{data_path()}/{folder}"
            filename = f"{path_to_folder}/{system}_stimulus100ms{npat}.h5"

            rates_exc = []
            rates_inh = []

            with h5py.File(filename, 'r', swmr=True) as h5f:
                N_exc = h5f['connectivity'].attrs['N_exc']
                N_inh = h5f['connectivity'].attrs['N_inh']

                spikes_exc = h5f['spikes_exc'][:].T
                spikes_inh = h5f['spikes_exc'][:].T
                max_t = h5f.attrs['simulation_time']

            nstim = min(npat, int(max_t - 1))

            patterns = load_patterns(npat)

            tuples = create_stim_tuples(patterns, 10, npat)

            _, sc = get_spike_counts(*spikes_exc, max_t, N_exc, dt=0.01)

            stim_responses = np.zeros((nstim, 150), dtype=float)
            nonstim_responses = np.zeros((nstim, 150), dtype=float)

            res[npat][system] = {
                'stim': stim_responses,
                'nonstim': nonstim_responses,
                'sact': [],
                'nsact': [],
                'act_counts': get_act_counts(system, npat)
            }

            for i in range(nstim):
                stimulated = tuples[i][2]
                mask = np.isin(patterns[i], tuples[i][2])

                sr = sc[:,(i+1)*100-50:(i+2)*100][patterns[i][mask]]
                nsr = sc[:,(i+1)*100-50:(i+2)*100][patterns[i][~mask]]

                stim_responses[i] = sr.mean(axis=0)
                nonstim_responses[i] = nsr.mean(axis=0)

                s_activation = (sr.reshape(((mask).sum(), 15, 10)).sum(axis=2) > 0).mean(axis=0)
                res[npat][system]['sact'].append(np.repeat(s_activation, 10))

                ns_activation = (nsr.reshape(((~mask).sum(), 15, 10)).sum(axis=2) > 0).mean(axis=0)
                res[npat][system]['nsact'].append(np.repeat(ns_activation, 10))

    with open('plotting/data/retrieve2.pkl', 'wb') as f:
        pickle.dump(res, f)