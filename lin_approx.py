import numpy as np
from tqdm import tqdm
import pickle
import yaml

from utils import underscore


folder = 'sp1'

with open('config/server_config.yaml') as f:
    config = yaml.safe_load(f)

path = config['data_path']


def get_activations(plast, npat, folder, prefix, suffix, postsynaptic, stim_len=40, skip=5):
    inh = ''
    n_neuron = 8000
    if postsynaptic == 'inh':
        inh = '_inh'
        n_neuron = 2000

    xx = np.linspace(-0.35, 0.35, 8)

    activations = {}

    for presynaptic in ['exc','inh']:
        activations[presynaptic] = []

        res_file_pert = f'{path}/{folder}/data/trained_{plast}_{prefix}{npat}_results_perturbed{underscore(suffix)}_{presynaptic}.pkl'

        with open(res_file_pert, 'rb') as file:
            results_pert = pickle.load(file)

        nstim = results_pert['params']['count']

        traces = results_pert['analysis'][f'spike_counts{inh}'][:,5:].reshape(n_neuron, nstim, stim_len).mean(axis=1)

        for trace in tqdm(traces):
            yy = trace[::skip]
            coefs = np.polyfit(xx, yy, deg=3)[::-1]
            activations[presynaptic].append([coefs[0], coefs[1]])

        activations[presynaptic] = np.array(activations[presynaptic])

    return activations

def run_lin(W_lin, patterns, rand=False, seed=42):
    np.random.seed(seed)

    rates = {}
    rates['pattern'] = []
    rates['rest'] = []
    rates['second'] = []
    rates['second_ix'] = []

    corrs = {}
    corrs['pattern'] = []
    corrs['rest'] = []
    corrs['second'] = []
    corrs['second_ix'] = []

    inhibs = {}
    inhibs['pattern'] = []
    inhibs['rest'] = []

    excits = {}
    excits['pattern'] = []
    excits['rest'] = []

    if rand:
        tmp_patterns = []
        for pat in patterns:
            tmp_patterns.append(np.random.permutation(pat))

        tmp_patterns = np.array(tmp_patterns)
    else:
        tmp_patterns = patterns

    cent_patterns = ((tmp_patterns.T - tmp_patterns.mean(axis=1)) / tmp_patterns.std(axis=1)).T
    pat_sizes = tmp_patterns.sum(axis=1)
    norm_patterns = (tmp_patterns.T / pat_sizes).T

    # for ix in tqdm(np.random.randint(low=0, high=1000, size=10)):
    for ix in tqdm(range(200)):
        full_pattern = np.concatenate([tmp_patterns[ix], np.zeros(2000)])

        x = full_pattern * (np.random.rand(10000) < 1) #+ np.random.randn(10000) * 1
        trace = []

        for i in (range(500)):
            trace.append(x)
            x = 0.999 * x + 0.001 * (W_lin @ x)

        trace = np.array(trace)

            # corrs.append(((trace[:, :8000].T - np.mean(trace[:, :8000], axis=1)) / np.std(trace[:, :8000], axis=1)).T @ (patterns[ix] - np.mean(patterns[ix])) / np.std(patterns[ix]))

        pat_avgs = (trace[:, :8000] @ norm_patterns.T).T

        rates['pattern'].append(pat_avgs[ix])
        pat_avgs = np.delete(pat_avgs, ix, axis=0)
        rates['rest'].append(pat_avgs.mean(axis=0))
        rates['second'].append(pat_avgs.max(axis=0))
        rates['second_ix'].append(np.argmax(pat_avgs, axis=0))

        exc_trace = trace[:,:8000]
        cent_trace = ((exc_trace.T - exc_trace.mean(axis=1)) / exc_trace.std(axis=1)).T

        pat_corrs = (cent_trace @ cent_patterns.T).T / 8000

        corrs['pattern'].append(pat_corrs[ix])
        pat_corrs = np.delete(pat_corrs, ix, axis=0)
        corrs['rest'].append(pat_corrs.mean(axis=0))
        corrs['second'].append(pat_corrs.max(axis=0))
        corrs['second_ix'].append(np.argmax(pat_corrs, axis=0))

        pat_inh = (norm_patterns @ (W_lin[:8000,8000:] @ trace[:,8000:].T))

        inhibs['pattern'].append(pat_inh[ix])
        pat_inh = np.delete(pat_inh, ix, axis=0)
        inhibs['rest'].append(pat_inh.mean(axis=0))

        pat_exc = (norm_patterns @ (W_lin[:8000,:8000] @ trace[:,:8000].T))

        excits['pattern'].append(pat_exc[ix])
        pat_inh = np.delete(pat_exc, ix, axis=0)
        excits['rest'].append(pat_exc.mean(axis=0))

    return rates, corrs, inhibs, excits

def get_linear_approximations(plast, npat, folder, prefix, suffix=''):
    mat_file = f'{path}/{folder}/connectivity/training_{plast}_{prefix}{npat}_matrix{underscore(suffix)}.pkl'

    with open(mat_file, 'rb') as file:
        Z, N_exc, patterns, exc_alpha, delays, params = pickle.load(file)

    act_exc = get_activations(plast, npat, folder, prefix, suffix, 'exc')
    act_inh = get_activations(plast, npat, folder, prefix, suffix, 'inh')

    pat_sizes = patterns.sum(axis=1)
    norm_patterns = (patterns.T / pat_sizes).T

    W_lin = np.copy(Z)

    W_lin[:8000, :8000] = (Z[:8000, :8000].T * act_exc['exc'][:,1]).T
    W_lin[:8000, 8000:] = (Z[:8000, 8000:].T * act_exc['inh'][:,1]).T
    W_lin[8000:, :8000] = (Z[8000:, :8000].T * act_inh['exc'][:,1]).T
    W_lin[8000:, 8000:] = (Z[8000:, 8000:].T * act_inh['inh'][:,1]).T

    return W_lin, patterns, act_exc, act_inh


if __name__ == '__main__':


    avg_rates = {}
    correlations = {}
    pattern_inhibs = {}

    avg_rates = {}
    correlations = {}
    pattern_inhibs = {}

    for plast in ['hebb','rate']:
        r, c, i = run_lin(plast, patterns)
        avg_rates[plast] = r
        correlations[plast] = c
        pattern_inhibs[plast] = i