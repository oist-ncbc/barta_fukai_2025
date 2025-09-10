import numpy as np
import pandas as pd

from utils import data_path
from analysis import get_spike_counts


if __name__ == '__main__':
    folder = f"{data_path()}/lognormal"
    filename = f"{folder}/linear_approx/hebb1000.csv"

    pd.read_csv(filename)

    with h5py.File(filename, "r") as h5f:
        spikes_exc = h5f['spikes_exc'][:].T
        N_exc = h5f['connectivity'].attrs['N_exc']
        t_max = h5f.attrs['simulation_time']
        # print(len(h5f['connectivity/patterns/splits']))

    _, sc = get_spike_counts(*spikes_exc, t_max, dt=0.01)

    exc_response = sc[:,400:].reshape(8000,-1,400)[:,::2,:].mean(axis=0).mean(axis=0)
    inh_response = sc[:,400:].reshape(8000,-1,400)[:,1::2,:].mean(axis=0).mean(axis=0)

    np.savetxt('plotting/data/perturb_responses.csv', np.array([exc_response, inh_response]))