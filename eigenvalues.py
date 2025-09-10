import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm

from utils import *
from analysis import get_spike_counts, get_firing_rates, get_mean_exc

def get_W(system, npat, effective=False, exc_vals=True):
    if exc_vals:
        data = pd.read_csv(f'{data_path()}/lognormal/linear_approx/{system}{npat}.csv', index_col=[0,1])

    connectivity = load_connectivity(system,'train', npat)
    patterns = load_patterns(npat)

    dense_patterns = patterns.dense()
    norm = np.linalg.norm(dense_patterns, axis=1)
    norm_patterns = (dense_patterns.T / norm).T

    vals = connectivity['weights']['EE']['weights']
    sources = connectivity['weights']['EE']['sources']
    targets = connectivity['weights']['EE']['targets']

    coo = sparse.coo_array((vals, (targets, sources)))
    if exc_vals:
        excitability = data.loc['exc']['activation_exc'].values
    else:
        excitability = 1
    WEE = (coo.toarray().T * excitability).T

    vals = connectivity['weights']['EI']['weights']
    sources = connectivity['weights']['EI']['sources']
    targets = connectivity['weights']['EI']['targets']

    coo = sparse.coo_array((vals, (targets, sources)))
    if exc_vals:
        excitability = data.loc['exc']['activation_inh'].values
    else:
        excitability = 1
    WEI = (coo.toarray().T * excitability).T

    vals = connectivity['weights']['IE']['weights']
    sources = connectivity['weights']['IE']['sources']
    targets = connectivity['weights']['IE']['targets']

    coo = sparse.coo_array((vals, (targets, sources)))
    if exc_vals:
        excitability = data.loc['inh']['activation_exc'].values
    else:
        excitability = 1
    WIE = (coo.toarray().T * excitability).T

    vals = connectivity['weights']['II']['weights']
    sources = connectivity['weights']['II']['sources']
    targets = connectivity['weights']['II']['targets']

    coo = sparse.coo_array((vals, (targets, sources)))
    if exc_vals:
        excitability = data.loc['inh']['activation_inh'].values
    else:
        excitability = 1
    WII = (coo.toarray().T * excitability).T

    if effective:
        # W = WEE + WEI @ np.linalg.inv(np.eye(2000)-WII) @ WIE
        W = WEE + WEI @ WIE

    else:
        WEX = np.concatenate([WEE,WEI], axis=1)
        WIX = np.concatenate([WIE,WII], axis=1)
        W = np.concatenate([WEX,WIX], axis=0)

    return W


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--system', type=str, required=True)
    parser.add_argument('--patterns', type=int, required=True)
    parser.add_argument('--effective', action='store_true')
    parser.add_argument('--exc', action='store_true')
    parser.add_argument('--vals_only', action='store_true')

    args = parser.parse_args()

    system = args.system
    npat = args.patterns

    W = get_W(system, npat, args.effective)

    if args.exc:
        vals, vecs = np.linalg.eig(W[:8000,:8000])
    else:
        vals, vecs = np.linalg.eig(W)
    
    if args.vals_only:
        np.savetxt(f'{data_path()}/lognormal/eigensystem/{system}{npat}.csv', vals)
    else:
        np.savetxt(f'{data_path()}/lognormal/eigensystem/{system}{npat}.csv', np.concatenate([[vals], vecs]))