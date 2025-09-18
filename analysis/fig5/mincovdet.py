import h5py
import matplotlib.pyplot as plt
from utils import plot_covariance_ellipse, despine_ax, data_path
from gstats_multiproc import robust_estimate
import numpy as np
import pickle


def cov_estimate(x, y):
    cov = np.cov(x, y)
    means = np.array([np.mean(x), np.mean(y)])

    return means, cov


if __name__ == '__main__':
    npat = 1000

    folder = f"{data_path()}/lognormal"
    filename = f"{folder}/rate_conductances{npat}.h5"

    neuron_ix = 30

    with h5py.File(filename, "r") as h5f:
        ge = h5f['state/exc']['ge'][1000:,neuron_ix]
        gi = h5f['state/exc']['gi'][1000:,neuron_ix]

    cov_rob = robust_estimate(ge, gi)
    cov_nrb = cov_estimate(ge, gi)

    res = {
        'ge': ge,
        'gi': gi,
        'cov_rob': cov_rob,
        'cov_nrb': cov_nrb
    }

    with open('plotting/data/conductances.pkl', 'wb') as f:
        pickle.dump(res, f)

    # fig, ax = plt.subplots(figsize=(6,4))
    # plot_cov_est(ax, which='r')
    # plot_cov_est(ax, which='n')
    # # ax.legend(frameon=False)

    # fig.tight_layout()
    # plt.savefig('plotting/img/fig5.png', dpi=300)