import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm

from utils import *

import numpy as np

from utils import *
from analysis import *
from eigenvalues import get_W


def load_activation(system, npat, run='spontaneous', namespace='lognormal'):
    folder = f"{data_path()}/{namespace}"
    filename = f"{folder}/{system}_{run}{npat}_activations.h5"

    with h5py.File(filename, "r", swmr=True) as h5f:
        act_times, durations, pattern_ixs = h5f['activations'][:]

    return act_times, durations, pattern_ixs


def get_act_counts(system, npat):
    act_times, _, pattern_ixs = load_activation(system, npat)
    act_counts = pd.Series(pattern_ixs).value_counts().reindex(np.arange(npat), fill_value=0).values
    return act_counts


if __name__ == '__main__':
    for npat in tqdm([800, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400, 2600, 2800, 3000]):
        res = {}
        patterns = load_patterns(npat).dense()
        norm_patterns = (patterns.T / patterns.sum(axis=1)).T

        for system in ['hebb','hebb_smooth_rate','rate']:
            W = get_W(system=system, npat=npat, exc_vals=False)

            Wei = W[:8000,8000:]
            Wee = W[:8000,:8000]
            
            var_stats = pd.read_csv(f'{data_path()}/lognormal/var_stats/{system}_conductances{npat}_stats.csv', index_col=[0,1])

            totexcit = var_stats.loc['exc']['mean_e']
            totinhib = var_stats.loc['exc']['mean_i']

            act_counts = get_act_counts(system, npat)

            res[(system, npat, 'tot_excit')] = norm_patterns @ totexcit
            res[(system, npat, 'tot_inhib')] = norm_patterns @ totinhib
            res[(system, npat, 'act_counts')] = act_counts

        pd.DataFrame(res).to_csv(f'plotting/data/totweights/totweights_{npat}.csv')