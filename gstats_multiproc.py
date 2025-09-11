"""Robust conductance statistics from conductance-recording simulations (multiprocessing).

This script consumes HDF5 files produced by running `network_ch` with
`config/runtypes/conductances.yml`, where per-neuron **excitatory** and
**inhibitory** conductances (`ge`, `gi`) were recorded under `state/{exc,inh}`.

For each neuron it estimates:
- robust mean of `(ge, gi)`
- robust covariance matrix of `(ge, gi)`

using **Minimum Covariance Determinant** (MCD, `sklearn.covariance.MinCovDet`).
The results are combined into a single CSV containing, for every neuron across
E and I populations:

- `mean_e`, `mean_i`   → robust means of `ge`, `gi`
- `std_e`, `std_i`     → square roots of the robust covariance diagonal
- `pearsonr`           → robust correlation = cov / (std_e * std_i)

Parallelization
---------------
Neurons are processed in **batches of size 5** via `multiprocessing.Pool`.
Each batch is passed to `process_batch`, which calls `robust_estimate` for
all neurons in the batch. The number of processes defaults to `cpu_count()`.

Notes
-----
- The HDF5 is expected at: `<data_path>/<folder>/<name><patterns>.h5` and must
  contain datasets `state/exc/ge`, `state/exc/gi`, `state/inh/ge`, `state/inh/gi`.
- The code discards the first 1000 time steps (via `[1000:, :]`) and then
  reshapes the remaining arrays to `(n_batches, batch_size, T)` after a transpose.
  If `N` is not a multiple of `batch_size`, the **last remainder neurons are
  dropped** (matching the original behavior).
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
import yaml

from sklearn.covariance import MinCovDet
from multiprocessing import Pool, cpu_count

from utils import *


def robust_estimate(x, y):
    """Compute robust location and covariance for paired (ge, gi) samples.

    Parameters
    ----------
    x, y : array_like
        Time series of `ge` and `gi` for a **single neuron**.

    Returns
    -------
    (location, covariance)
        `location` is a length-2 array (robust mean of ge and gi),
        `covariance` is a 2×2 array from MinCovDet.
    """
    X = np.array([x, y]).T
    robust_cov = MinCovDet().fit(X)

    return robust_cov.location_, robust_cov.covariance_


def process_batch(data):
    """Process one batch of neurons: robust stats for each neuron's (ge, gi).

    Parameters
    ----------
    data : tuple
        `(ge_arrs, gi_arrs)` where each has shape `(batch_size, T)` after
        pre-reshaping, representing `batch_size` neurons' time series.

    Returns
    -------
    (means, covs)
        `means` is a list of 2-element arrays; `covs` is a list of 2×2 arrays.
    """
    ge_arrs, gi_arrs = data
    means = []
    covs = []

    for ge, gi in zip(ge_arrs, gi_arrs):
        mean, cov = robust_estimate(ge, gi)
        means.append(mean)
        covs.append(cov)

    return means, covs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--folder', type=str, required=True, help="Data namespace folder under data_path")
    parser.add_argument('--name', type=str, required=True, help="Base filename prefix (system/run identifier)")
    parser.add_argument('--patterns', type=int, required=True, help="Number of patterns used in filename")

    args = parser.parse_args()

    # Load base data directory from server config
    with open('config/server_config.yaml') as f:
        server_config = yaml.safe_load(f)

    batch_size = 5  # number of neurons processed together per worker call

    # Input/Output paths
    folder_path = f"{server_config['data_path']}/{args.folder}"
    input_file  = f"{folder_path}/{args.name}{args.patterns}.h5"
    output_file = f"{folder_path}/var_stats/{args.name}{args.patterns}_stats.csv"
    

    # Accumulators for both populations
    stats = {
        'mean_e': np.array([]),
        'mean_i': np.array([]),
        'std_e': np.array([]),
        'std_i': np.array([]),
        'cov': np.array([])
    }

    index_ei = []  # 'exc'/'inh' labels aligned with rows
    index_n  = []  # neuron indices aligned with rows

    for ei in ['exc','inh']:
        print(f'calculating {ei}')

        # Open HDF5 and slice by population
        with h5py.File(input_file, "r") as h5f:  # type: ignore[name-defined]
            N = h5f['connectivity'].attrs[f'N_{ei}']
            print(f'{N} neurons')
            n_batches = N // batch_size

            # Load and transpose to shape (T, N), drop first 1000 time steps
            ge = h5f[f'state/{ei}/ge'][1000:,:].T.reshape(n_batches, batch_size, -1)
            gi = h5f[f'state/{ei}/gi'][1000:,:].T.reshape(n_batches, batch_size, -1)

        # Parallel robust estimation across batches
        n_cpu = cpu_count()
        with Pool(processes=n_cpu) as pool:
            # tqdm shows a progress bar; smoothing/mininterval tuned for multi-core
            results = list(tqdm(
                pool.imap(process_batch, zip(ge, gi)),
                total=n_batches,
                smoothing=0,
                mininterval=n_cpu
            ))

        # Unpack lists of arrays to flat arrays and concatenate
        means = np.concatenate(np.array([res[0] for res in results]))
        covs  = np.concatenate(np.array([res[1] for res in results]))

        # Split robust stats into separate columns
        stats['mean_e'] = np.concatenate([stats['mean_e'], means[:,0]])
        stats['mean_i'] = np.concatenate([stats['mean_i'], means[:,1]])
        stats['std_e'] = np.concatenate([stats['std_e'], np.sqrt(covs[:,0,0])])
        stats['std_i'] = np.concatenate([stats['std_i'], np.sqrt(covs[:,1,1])])
        stats['cov'] = np.concatenate([stats['cov'], covs[:,0,1]])

        # Build row index entries
        index_ei.extend(N * [ei])
        index_n.extend(list(range(N)))

    # Assemble DataFrame with a 2-level index and compute robust Pearson r
    df = pd.DataFrame(stats, index=[index_ei, index_n])
    df['pearsonr'] = df['cov'] / (df['std_e'] * df['std_i'])

    # Persist a subset of columns (as in original code)
    df[['mean_e','mean_i','std_e','std_i','pearsonr']].to_csv(output_file)
