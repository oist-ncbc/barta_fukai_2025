import numpy as np
import pandas as pd
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
    parser = argparse.ArgumentParser()

    parser.add_argument('--folder', type=str, required=True)
    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--patterns', type=int, required=True)

    args = parser.parse_args()

    # args = parser.parse_args()

    with open('config/server_config.yaml') as f:
        server_config = yaml.safe_load(f)


    batch_size = 5

    # with h5py.File('data/hebb1000_cond_stats.h5', "w") as h5f:
    #     h5f.create_group('exc')
    #     h5f.create_group('inh')

    folder_path = f"{server_config['data_path']}/{args.folder}"
    input_file  = f"{folder_path}/{args.name}{args.patterns}.h5"
    output_file = f"{folder_path}/var_stats/{args.name}{args.patterns}_stats.csv"
    

    stats = {
        'mean_e': np.array([]),
        'mean_i': np.array([]),
        'std_e': np.array([]),
        'std_i': np.array([]),
        'cov': np.array([])
    }

    index_ei = []
    index_n  = []

    for ei in ['exc','inh']:
        print(f'calculating {ei}')

        with h5py.File(input_file, "r") as h5f:
            N = h5f['connectivity'].attrs[f'N_{ei}']
            print(f'{N} neurons')
            n_batches = N // batch_size
            ge = h5f[f'state/{ei}/ge'][1000:,:].T.reshape(n_batches, batch_size, -1)
            gi = h5f[f'state/{ei}/gi'][1000:,:].T.reshape(n_batches, batch_size, -1)

        with Pool(processes=None) as pool:
            results = list(tqdm(pool.imap(process_batch, zip(ge, gi)), total=n_batches))

        means = np.concatenate(np.array([res[0] for res in results]))
        covs  = np.concatenate(np.array([res[1] for res in results]))

        stats['mean_e'] = np.concatenate([stats['mean_e'], means[:,0]])
        stats['mean_i'] = np.concatenate([stats['mean_i'], means[:,1]])
        stats['std_e'] = np.concatenate([stats['std_e'], np.sqrt(covs[:,0,0])])
        stats['std_i'] = np.concatenate([stats['std_i'], np.sqrt(covs[:,1,1])])
        stats['cov'] = np.concatenate([stats['cov'], covs[:,0,1]])

        index_ei.extend(N * [ei])
        index_n.extend(list(range(N)))

    df = pd.DataFrame(stats, index=[index_ei, index_n])
    df['pearsonr'] = df['cov'] / (df['std_e'] * df['std_i'])
    df[['mean_e','mean_i','std_e','std_i','pearsonr']].to_csv(output_file)
        # with h5py.File('data/hebb1000_cond_stats.h5', "a") as h5f:
        #     h5f[ei].create_dataset('mean', data=means, dtype='float32')
        #     h5f[ei].create_dataset('cov', data=covs, dtype='float32')