import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
import argparse
import yaml

from sklearn.covariance import MinCovDet

from utils import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--plast', type=str)
    parser.add_argument('--patterns', type=int)
    parser.add_argument('--prefix', type=str)
    parser.add_argument('--suffix', type=str, default='')

    args = parser.parse_args()

    with open('config/server_config.yaml') as f:
        config = yaml.safe_load(f)

    folder = 'sp1'
    npat = args.patterns
    path = f"{config['data_path']}/{folder}"

    plast = args.plast

    if args.suffix != '':
        suffix = '_' + args.suffix
    else:
        suffix = ''

    # ----------------- STIMULUS -----------------

    # res_file = f'data/trained_{plast}_nonburst{npat}_results_stim.pkl'
    res_file = f'{path}/data/trained_{plast}_{args.prefix}{npat}_results_state{underscore(args.suffix)}.pkl'

    with open(res_file, 'rb') as file:
        results_state = pickle.load(file)


    cond_data = []

    for ix in tqdm(range(8000)):
        xx, yy = results_state['state_exc']['ge'][1000:, ix], results_state['state_exc']['gi'][1000:, ix]
        X = np.array([xx[::100], yy[::100]]).T
        robust_cov = MinCovDet().fit(X)
        res = [robust_cov.location_[0], robust_cov.location_[1], robust_cov.covariance_[0, 0],
               robust_cov.covariance_[1, 1], robust_cov.covariance_[1, 0]]
        cond_data.append(res)

    pd.DataFrame(cond_data, columns=['mean_e', 'mean_i', 'var_e', 'var_i', 'cov']).to_csv(f'config/var_data_{plast}_{args.prefix}{npat}{suffix}_e.csv', index=False)

    cond_data = []

    for ix in tqdm(range(2000)):
        xx, yy = results_state['state_inh']['ge'][1000:, ix], results_state['state_inh']['gi'][1000:, ix]
        X = np.array([xx[::100], yy[::100]]).T
        robust_cov = MinCovDet().fit(X)
        res = [robust_cov.location_[0], robust_cov.location_[1], robust_cov.covariance_[0, 0],
               robust_cov.covariance_[1, 1], robust_cov.covariance_[1, 0]]
        cond_data.append(res)

    pd.DataFrame(cond_data, columns=['mean_e', 'mean_i', 'var_e', 'var_i', 'cov']).to_csv(f'config/var_data_{plast}_{args.prefix}{npat}{suffix}_i.csv', index=False)