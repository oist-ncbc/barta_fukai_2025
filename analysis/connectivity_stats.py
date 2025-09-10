import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle

from utils import *


def get_W(system, npat):
    connectivity = load_connectivity(system,'train', npat)

    vals = connectivity['weights']['EE']['weights']
    sources = connectivity['weights']['EE']['sources']
    targets = connectivity['weights']['EE']['targets']

    coo = sparse.coo_array((vals, (targets, sources)))
    WEE = coo.toarray().T.T

    # vals = connectivity['weights']['EI']['weights']
    # sources = connectivity['weights']['EI']['sources']
    # targets = connectivity['weights']['EI']['targets']

    # coo = sparse.coo_array((vals, (targets, sources)))
    # WEI = coo.toarray().T.T

    # vals = connectivity['weights']['IE']['weights']
    # sources = connectivity['weights']['IE']['sources']
    # targets = connectivity['weights']['IE']['targets']

    # coo = sparse.coo_array((vals, (targets, sources)))
    # WIE = coo.toarray().T.T

    # vals = connectivity['weights']['II']['weights']
    # sources = connectivity['weights']['II']['sources']
    # targets = connectivity['weights']['II']['targets']

    # coo = sparse.coo_array((vals, (targets, sources)))
    # WII = coo.toarray().T.T

    # WEX = np.concatenate([WEE,WEI], axis=1)
    # WIX = np.concatenate([WIE,WII], axis=1)
    # W = np.concatenate([WEX,WIX], axis=0)

    return WEE

def connectivity_stats(npat):
    system = 'hebb'
    WEE = get_W(system, npat)

    patterns = load_patterns(npat, system)
    rand_patterns = patterns.randomize()

    conn_probs = []
    conn_weights = []

    conn_probs_rand = []
    conn_weights_rand = []

    for pat, rpat in zip(patterns, rand_patterns):
        submatrix = WEE[pat].T[pat]
        conn_probs.append((submatrix > 0).mean())
        conn_weights.append(submatrix[submatrix>0].mean())

        submatrix = WEE[rpat].T[rpat]
        conn_probs_rand.append((submatrix > 0).mean())
        conn_weights_rand.append(submatrix[submatrix>0].mean())

    res = {
        'pattern_conn_prob': np.mean(conn_probs),
        'random_conn_prob': np.mean(conn_probs_rand),
        'pattern_conn_weight': np.mean(conn_weights),
        'random_conn_weight': np.mean(conn_weights_rand)
    }

    return res


if __name__ == '__main__':
    npat_list = [800, 1000, 1200, 1400, 1600, 1800,
                 2000, 2200, 2400, 2600, 2800, 3000]

    res = {}

    for npat in tqdm(npat_list):
        res[npat] = connectivity_stats(npat)

    df = pd.DataFrame(res).T
    print(df)
    print(df.to_latex(float_format="%.2f"))