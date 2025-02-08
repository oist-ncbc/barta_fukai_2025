import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
import argparse
import yaml

from sklearn.covariance import MinCovDet
from multiprocessing import Pool

from utils import *


def robust_estimate(x, y):
    X = np.array([x, y]).T
    robust_cov = MinCovDet().fit(X)

    return robust_cov.location_, robust_cov.covariance_

def process_batch(data):
    ge_arrs, gi_arrs = data
    means = []
    covs = []

    for ge, gi in zip(ge_arrs, gi_arrs):
        mean, cov = robust_estimate(ge, gi)
        means.append(mean)
        covs.append(cov)

    return means, covs


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()

    # parser.add_argument('results', type=str, help='results file')
    # parser.add_argument('--plast', type=str)
    # parser.add_argument('--patterns', type=int)
    # parser.add_argument('--prefix', type=str)
    # parser.add_argument('--suffix', type=str, default='')


    # args = parser.parse_args()

    # with open('config/server_config.yaml') as f:
    #     config = yaml.safe_load(f)
    batch_size = 5

    with h5py.File('data/hebb1000_cond_stats.h5', "w") as h5f:
        h5f.create_group('exc')
        h5f.create_group('inh')

    for ei in ['exc','inh']:
        print(f'calculating {ei}')

        with h5py.File("data/hebb_conductances1000.h5", "r") as h5f:
            N = h5f['connectivity'].attrs[f'N_{ei}']
            print(f'{N} neurons')
            n_batches = N // batch_size
            ge = h5f[f'state/{ei}/ge'][1000::1,:].T.reshape(n_batches, batch_size, -1)
            gi = h5f[f'state/{ei}/gi'][1000::1,:].T.reshape(n_batches, batch_size, -1)

        with Pool(10) as pool:
            results = list(tqdm(pool.imap(process_batch, zip(ge, gi)), total=n_batches))

        means = np.concatenate(np.array([res[0] for res in results]))
        covs  = np.concatenate(np.array([res[1] for res in results]))

        with h5py.File('data/hebb1000_cond_stats.h5', "a") as h5f:
            h5f[ei].create_dataset('mean', data=means, dtype='float32')
            h5f[ei].create_dataset('cov', data=covs, dtype='float32')