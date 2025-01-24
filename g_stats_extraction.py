import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
import argparse

from sklearn.covariance import MinCovDet

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--plast', type=str)

    args = parser.parse_args()

    folder = 'sp1'
    npat = 1000
    path = f'/media/tomasbarta/DATA/StructuredInihibition/{folder}'

    plast = args.plast

    # ----------------- STIMULUS -----------------

    # res_file = f'data/trained_{plast}_nonburst{npat}_results_stim.pkl'
    res_file = f'{path}/data/trained_{plast}_nonburst{npat}_results_state.pkl'

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

    pd.DataFrame(cond_data, columns=['mean_e', 'mean_i', 'var_e', 'var_i', 'cov']).to_csv(f'config/var_data_{plast}_e.csv', index=False)

    cond_data = []

    for ix in tqdm(range(2000)):
        xx, yy = results_state['state_inh']['ge'][1000:, ix], results_state['state_inh']['gi'][1000:, ix]
        X = np.array([xx[::100], yy[::100]]).T
        robust_cov = MinCovDet().fit(X)
        res = [robust_cov.location_[0], robust_cov.location_[1], robust_cov.covariance_[0, 0],
               robust_cov.covariance_[1, 1], robust_cov.covariance_[1, 0]]
        cond_data.append(res)

    pd.DataFrame(cond_data, columns=['mean_e', 'mean_i', 'var_e', 'var_i', 'cov']).to_csv(f'config/var_data_{plast}_i.csv', index=False)