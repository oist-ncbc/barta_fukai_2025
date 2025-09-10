import argparse
import numpy as np
import time
from scipy import stats
from tqdm import tqdm

from utils import *
from analysis import *
from eigenvalues import get_W


if __name__ == '__main__':
    for npat in tqdm([800, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400, 2600, 2800, 3000]):
        patterns = load_patterns(npat)
        dense_patterns = patterns.dense()
        rand_patterns = patterns.randomize().dense()

        entropies = {}
        entropies_ass = {}
        entropies_rand_ass = {}

        for system in ['hebb','hebb_smooth_rate', 'rate','uniform']:
            if system != 'uniform':
                W = get_W(system=system, npat=npat, exc_vals=False)
            else:
                W = get_W(system='rate', npat=npat, exc_vals=False)
                W[:8000,8000:] = (W[:8000,8000:] > 0).astype(float) * 0.5

            entropies[system] = []
            entropies_ass[system] = []
            entropies_rand_ass[system] = []

            for weights in W[:8000,8000:]:
                distr = weights / weights.sum()
                entropies[system].append(stats.entropy(distr))

            for weights in (dense_patterns @ W[:8000,8000:]).T:
                distr = weights / weights.sum()
                entropies_ass[system].append(stats.entropy(distr))

            for weights in (rand_patterns @ W[:8000,8000:]).T:
                distr = weights / weights.sum()
                entropies_rand_ass[system].append(stats.entropy(distr))

            entropies[system] = np.array(entropies[system])
            entropies_ass[system] = np.array(entropies_ass[system])
            entropies_rand_ass[system] = np.array(entropies_rand_ass[system])

        pd.DataFrame(entropies).to_csv(f'plotting/data/inhibitory_specialization/entropies_{npat}.csv')
        pd.DataFrame(entropies_ass).to_csv(f'plotting/data/inhibitory_specialization/entropies_assemblies_{npat}.csv')
        pd.DataFrame(entropies_rand_ass).to_csv(f'plotting/data/inhibitory_specialization/entropies_rand_assemblies_{npat}.csv')

    ass_diff = {}
    npat = 1000

    patterns = load_patterns(npat)
    dense_patterns = patterns.dense()

    for system in ['hebb','hebb_smooth_rate','rate']:
        W = get_W(system=system, npat=npat, exc_vals=False)

        inhibs = dense_patterns @ (W[:8000,8000:] @ (W[8000:,:8000] @ dense_patterns.T)) / dense_patterns.sum(axis=1)
        # i-th column: how i-th assembly affects other assemblies
        # we want to first normalize by how is an assembly affected in general by any other assembly
        # so we divide by the sum over the row
        inhib_by_others = (inhibs.sum(axis=1) - np.diag(inhibs)) / (npat-1)
        inhibs = (inhibs.T / inhib_by_others).T
        other_inh = (inhibs.sum(axis=0) - np.diag(inhibs)) / (npat-1)
        ass_diff[system] = (np.diag(inhibs) - other_inh) / other_inh

    pd.DataFrame(ass_diff).to_csv(f'plotting/data/self_feedback_{npat}.csv')