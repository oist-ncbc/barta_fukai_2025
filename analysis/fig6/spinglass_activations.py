import numpy as np
from scipy import sparse
import h5py
from tqdm import tqdm
import pickle
from itertools import product

from utils import load_patterns, data_path
from analysis import get_spike_counts
from eigenvalues import get_W


def spike_counts(system='rate', npat=2000, max_time=1000):
    folder = f"{data_path()}/lognormal"
    filename = f"{folder}/{system}_spontaneous{npat}.h5"

    with h5py.File(filename, "r") as h5f:
        max_t = h5f.attrs['simulation_time']
        N_exc = h5f['connectivity'].attrs['N_exc']
        spikes_exc = h5f['spikes_exc'][:,:max_time*4*N_exc]

    _, sc_exc = get_spike_counts(*spikes_exc.T, min(max_t, max_time), dt=0.1, N=N_exc)

    return sc_exc

def pattern_activity(patterns, sc_exc):
    activations = np.zeros_like(sc_exc)
    patrates = np.zeros_like(sc_exc)

    for ix in tqdm(range(npat)):
        activations[ix] = ((sc_exc[patterns[ix]] > 0).mean(axis=0))
        patrates[ix] = ((sc_exc[patterns[ix]]).mean(axis=0))

    return activations, patrates


if __name__ == '__main__':
    npat = 1000
    system = 'rate'

    all_res = {}

    for npat, system in product([1000, 1400, 2000], ['rate','hebb']):
        patterns = load_patterns(npat)

        sc_exc = spike_counts(system=system, npat=npat, max_time=1000)
        activations, patrates = pattern_activity(patterns, sc_exc[:,:10000])

        W = get_W(system, npat)
        vals, vecs = sparse.linalg.eigs(W, k=10)

        # vec_ixs = [(1, -0.02), (4, 0.025), (5, -0.025), (5, 0.025)]
        ix = np.argmax(vals.real)
        cut_val1 = vecs[:,ix][np.argsort(vecs[:,ix].real[:8000])[-50]].real
        cut_val2 = vecs[:,ix][np.argsort(vecs[:,ix].real[:8000])[50]].real
        vec_ixs = [(ix, cut_val1),(ix, cut_val2)]

        neuron_lists = [
            # np.argwhere(vecs[:,3].real[:8000] < -0.05).flatten(),
            np.argwhere(vecs[:,vi[0]].real[:8000]*np.sign(vi[1]) >= np.abs(vi[1])).flatten()
            for vi in vec_ixs
        ]

        newpat_rates = np.zeros(shape=(len(neuron_lists), sc_exc.shape[1]))
        newpat_activ = np.zeros(shape=(len(neuron_lists), sc_exc.shape[1]))

        for ix, neurons in enumerate(neuron_lists):
            newpat_rates[ix] = (sc_exc[neurons]).mean(axis=0)
            newpat_activ[ix] = (sc_exc[neurons] > 0).mean(axis=0)

        # overlapping_patterns_list = []
        all_res[(npat, system)] = []

        for nix, (nl, vi) in enumerate(zip(neuron_lists, vec_ixs)):
            res = {}
            res['neurons'] = nl
            res['rates'] = newpat_rates[nix]
            res['activation'] = newpat_activ[nix]
            res['overlapping_patterns'] = []
            res['overlaps_activity'] = []
            res['eigenvector'] = vecs[:,vi[0]].real[:8000]
            res['ev_cutoff'] = vi[1]

            newpat = np.zeros(8000)
            newpat[nl] = 1

            pixs = np.argsort(patterns.dense() @ newpat)[::-1][:5]
            overlapping_patterns = []
            for pix in pixs:
                res['overlaps_activity'].append(activations[pix])
                res['overlapping_patterns'].append(patterns[pix])

            all_res[(npat, system)].append(res)

    with open('plotting/data/overlap_activity.pkl', 'wb') as f:
        pickle.dump(all_res, f)