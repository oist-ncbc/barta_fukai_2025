import numpy as np
import h5py
from tqdm import tqdm
import pickle
import argparse

from utils import *
from analysis import get_spike_counts


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--npat', type=int)
    args = parser.parse_args()
    npat = args.npat

    spike_counts = {}

    system = 'hebb_smooth_rate'

    for system in tqdm(['hebb','hebb_smooth_rate','rate']):
        input_file  = f"{data_path()}/lognormal/{system}_conductances_long{npat}.h5"

        with h5py.File(input_file, "r") as h5f:
            # ge = h5f[f'state/exc/ge'][1000:,:].T
            # gi = h5f[f'state/exc/gi'][1000:,:].T

            spikes_exc = h5f['spikes_exc'][:].T
            spikes_inh = h5f['spikes_inh'][:].T

            max_t = h5f.attrs['simulation_time']

            _, sc_exc = get_spike_counts(*spikes_exc, t_max=max_t, N=8000, dt=0.01)
            _, sc_inh = get_spike_counts(*spikes_inh, t_max=max_t, N=2000, dt=0.01)

        # conductances[system] = (ge, gi)
        spike_counts[system] = (sc_exc, sc_inh)

    patterns = load_patterns(npat)
    norm_patterns = (patterns.dense().T / patterns.sizes()).T

    traces_exc = {}
    traces_inh = {}
    traces_inh_rest = {}
    traces_exc_rest = {}

    all_neur_exc = {}
    all_neur_inh = {}

    traces_exc_sc = {}
    traces_inh_sc = {}
    traces_inh_rest_sc = {}
    activ_prob = {}

    pix_lists = {}

    for i, system in enumerate(['hebb','hebb_smooth_rate','rate']):
        input_file  = f"{data_path()}/lognormal/{system}_conductances_long{npat}.h5"

        with h5py.File(input_file, "r") as h5f:
            run = 'conductances_long'
            folder = f"{data_path()}/lognormal"
            filename = f"{folder}/{system}_{run}{npat}_activations.h5"

            with h5py.File(filename, "r", swmr=True) as h5f_act:
                act_times, durations, pattern_ixs = h5f_act['activations'][:]

            # exc = norm_patterns @ conductances[system][0]
            # inh = norm_patterns @ conductances[system][1]

            sexc = norm_patterns @ spike_counts[system][0]

            traces_exc[system] = []
            traces_inh[system] = []
            traces_inh_rest[system] = []
            traces_exc_rest[system] = []

            all_neur_exc[system] = []
            all_neur_inh[system] = []

            traces_exc_sc[system] = []
            traces_inh_sc[system] = []
            activ_prob[system] = []

            pix_lists[system] = []

            before = 1000
            after = 1000

            for pix, pat in tqdm(zip(pattern_ixs.astype(int)[:], act_times.astype(int)), total=len(act_times[:])):
                if (pat*10+after < 200000) and (pat*10-before > 0):
                    activ_prob[system].append(act_times[(act_times > pat - before//10) & (act_times < pat + after//10)] - pat)

                    ge = h5f[f'state/exc/ge'][pat*10-before:pat*10+after,:].T
                    gi = h5f[f'state/exc/gi'][pat*10-before:pat*10+after,:].T

                    all_neur_exc[system].append(ge.mean(axis=0))
                    all_neur_inh[system].append(gi.mean(axis=0))

                    exc = norm_patterns @ ge
                    inh = norm_patterns @ gi

                    exc_pattern = exc[pix,:]
                    inh_pattern = inh[pix,:]
                    inh_all = inh[:,:].sum(axis=0)
                    exc_all = exc[:,:].sum(axis=0)

                    traces_exc[system].append(exc_pattern)
                    traces_inh[system].append(inh_pattern)
                    traces_inh_rest[system].append((inh_all-inh_pattern) / 999)
                    traces_exc_rest[system].append((exc_all-exc_pattern) / 999)

                    exc_pattern = sexc[pix,pat-before//10:pat+after//10]
                    inh_all = spike_counts[system][1][:,pat-before//10:pat+after//10].mean(axis=0)

                    if len(exc_pattern) != (before+after)//10:
                        print(system, pix, pat, len(exc_pattern))

                    traces_exc_sc[system].append(exc_pattern)
                    traces_inh_sc[system].append(inh_all)

                    pix_lists[system].append(pix)
            
            
            traces_exc[system] = np.array(traces_exc[system])
            traces_inh[system] = np.array(traces_inh[system])
            traces_inh_rest[system] = np.array(traces_inh_rest[system])
            traces_exc_rest[system] = np.array(traces_exc_rest[system])

            all_neur_exc[system] = np.array(all_neur_exc[system])
            all_neur_inh[system] = np.array(all_neur_inh[system])

            traces_exc_sc[system] = np.array(traces_exc_sc[system])
            traces_inh_sc[system] = np.array(traces_inh_sc[system])
            pix_lists[system] = np.array(pix_lists[system])


    results = {
        'traces_exc': traces_exc,
        'traces_inh': traces_inh,
        'all_neur_exc': all_neur_exc,
        'all_neur_inh': all_neur_inh,
        'traces_inh_rest': traces_inh_rest,
        'traces_exc_rest': traces_exc_rest,
        'traces_exc_sc': traces_exc_sc,
        'traces_inh_sc': traces_inh_sc,
        'traces_inh_rest_sc': traces_inh_rest_sc,
        'pix_lists': pix_lists,
        'activ_prob': activ_prob
    }

    with open(f'data/conductance_traces{npat}.pkl', 'wb') as f:
        pickle.dump(results, f)