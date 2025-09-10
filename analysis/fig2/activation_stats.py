import numpy as np
import pandas as pd
import h5py
from scipy import stats
from scipy.optimize import nnls
import pickle

from utils import *


def load_activation(system, npat, run='spontaneous', namespace='lognormal'):
    folder = f"{data_path()}/{namespace}"
    filename = f"{folder}/{system}_{run}{npat}_activations.h5"

    with h5py.File(filename, "r", swmr=True) as h5f:
        act_times, durations, pattern_ixs = h5f['activations'][:]

    return act_times, durations, pattern_ixs


def get_act_counts(act_times, pattern_ixs, npat, measure_time=10000):
    act_counts = pd.Series(pattern_ixs[act_times < measure_time]).value_counts().reindex(np.arange(npat), fill_value=0).values
    return act_counts

def estimate_entropy(*x_list):
    k_list = []
    e_list = []

    for x in x_list:
        k_list.append(x.sum())
        probs = x / x.sum()
        e_list.append(stats.entropy(probs))

    A = np.zeros((len(k_list), 3))
    A[:,0] = 1
    A[:,1] = -1 / np.array(k_list)
    A[:,2] = -1 / np.array(k_list)**2

    b = np.array(e_list)

    e = nnls(A, b)[0][0]
    # print (e_list, e)

    return e, e_list[-1]


if __name__ == '__main__':
    measure_time = 10000

    res = {}
    interpolations = {}
    inter_event_intervals = {}

    entropies = {}

    new_xx = np.linspace(0, measure_time, 201)

    for npat in [800, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400, 2600, 2800, 3000]:
    # for npat in [1000, 1200, 1400, 1600, 1800, 2000]:

        for system in ['hebb','hebb_smooth_rate','rate','shuffle']:
            if system == 'shuffle':
                run = 'spontaneous_shuffle'
                act_times, durations, pattern_ixs = load_activation('hebb', npat, run)
            else:
                run = 'spontaneous'
                act_times, durations, pattern_ixs = load_activation(system, npat, run)

            splits = np.argwhere(np.diff(pattern_ixs) != 0).flatten()+1
            splitted = np.split(act_times, splits)


            if len(act_times) > 0:
                act_counts = get_act_counts(act_times, pattern_ixs, npat, measure_time=1000000)
                act_counts7500 = get_act_counts(act_times, pattern_ixs, npat, measure_time=750000)
                act_counts5000 = get_act_counts(act_times, pattern_ixs, npat, measure_time=500000)
                act_counts2500 = get_act_counts(act_times, pattern_ixs, npat, measure_time=250000)

                if act_counts2500.sum() > 0:
                    entropy, entropy_k = estimate_entropy(act_counts2500, act_counts5000, act_counts7500, act_counts)
                elif act_counts.sum() > 0:
                    entropy, entropy_k = estimate_entropy(act_counts)
                else:
                    entropy = 0
                    entropy_k = np.nan

                entropies[(system, npat)] = {
                    'k': act_counts.sum(),
                    'S': entropy_k,
                    'Shat': entropy,
                    'D': np.exp(entropy_k),
                    'Dhat': np.exp(entropy)
                    }

                # probs = act_counts / act_counts.sum()
                res[(system, npat, 'entropy')] = np.exp(entropy)

                first_act = [x[0] for x in splitted]

                xx = np.sort(first_act) / 100
                yy = np.arange(len(xx))

                xx_to_del = np.argwhere(np.diff(xx) == 0)
                xx = np.delete(xx, xx_to_del)
                yy = np.delete(yy, xx_to_del)

                new_yy = np.interp(new_xx, xx, yy, left=0)

                interpolations[(system, npat)] = new_yy

                number = np.argwhere((xx - measure_time) < 0).flatten().max()

                res[(system, npat, 'nunique')] = number

                sorted_pix = pattern_ixs[np.argsort(act_times)]
                sorted_times = np.sort(act_times)
                iais = np.diff(sorted_times)


                first = sorted_pix[:-1]
                second = sorted_pix[1:]

                mask = (first != second)  # leaving out occasions when the same pattern is reactivated again

                res[(system, npat, 'mean_freq')] = 100 / iais[mask].mean()
                res[(system, npat, 'mean_dur')] = durations.mean()
                inter_event_intervals[(system, npat)] = iais[mask]
                print(system, npat, number)
            else:
                res[(system, npat, 'nunique')] = 0
                res[(system, npat, 'mean_freq')] = 0
                res[(system, npat, 'mean_dur')] = np.nan
                res[(system, npat, 'entropy')] = 0

                inter_event_intervals[(system, npat)] = np.array([])

    pd.Series(res).unstack(level=[0, 2]).to_csv('plotting/data/activation_stats.csv')
    pd.DataFrame(interpolations, index=new_xx).to_csv('plotting/data/gradual.csv')

    df = pd.DataFrame(entropies).T.sort_index()
    df['k'] = df['k'].astype(int)
    print(df.to_latex(float_format="{:.2f}".format))

    with open('plotting/data/iais.pkl', 'wb') as f:
        pickle.dump(inter_event_intervals, f)