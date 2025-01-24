import numpy as np
from tqdm import tqdm
import pickle


folder = 'sp1'
npat = 1000
path = f'/media/tomasbarta/DATA/StructuredInihibition/{folder}'


def get_activations(plast, ei='exc'):
    res_file_pert = f'{path}/data/trained_{plast}_nonburst{npat}_results_perturbed.pkl'
    # res_file_spont = f'{path}/data/trained_{plast}_nonburst{npat}_results_spont.pkl'

    with open(res_file_pert, 'rb') as file:
        results_pert = pickle.load(file)

    inh = ''
    n_neuron = 8000
    if ei == 'inh':
        inh = '_inh'
        n_neuron = 2000

    # res_corrected = np.concatenate([
    #     results_pert['analysis'][f'spike_counts{inh}'][:n_neuron,5:4005],
    #     results_pert['analysis'][f'spike_counts{inh}'][:n_neuron,8005:12005],
    # ], axis=1)

    xx = np.linspace(-0.35, 0.35, 8)
    stims = 500

    gains = {}

    for i, ei in enumerate(['exc', 'inh']):
        gains[ei] = []

        averaged = results_pert['analysis']['spike_counts'][:n_neuron, 5:40005].reshape((n_neuron, stims * 2, 40))[:,
                   i * stims:(i + 1) * stims, :].mean(axis=1)

        for trace in tqdm(averaged):
            yy = np.zeros(8)

            for i in range(8):
                yy[i] = trace[5 * i]

            gain = np.polyfit(xx, yy, 3)[-2]
            gains[ei].append(gain)

        gains[ei] = np.array(gains[ei])

    return gains

def run_lin(plast, patterns, linear_approximations, rand=False):
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

    Z_lin = linear_approximations[plast]

    # for ix in tqdm(np.random.randint(low=0, high=1000, size=10)):
    for ix in tqdm(range(10)):
        full_pattern = np.concatenate([tmp_patterns[ix], np.zeros(2000)])

        x = full_pattern * (np.random.rand(10000) < 0.1) #+ np.random.randn(10000) * 1
        trace = []

        for i in (range(500)):
            trace.append(x)
            x = 0.999 * x + 0.001 * (Z_lin @ x)

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

        pat_inh = (norm_patterns @ (Z_lin[:8000,8000:] @ trace[:,8000:].T))

        inhibs['pattern'].append(pat_inh[ix])
        pat_inh = np.delete(pat_inh, ix, axis=0)
        inhibs['rest'].append(pat_inh.mean(axis=0))

        pat_exc = (norm_patterns @ (Z_lin[:8000,:8000] @ trace[:,:8000].T))

        excits['pattern'].append(pat_exc[ix])
        pat_inh = np.delete(pat_exc, ix, axis=0)
        excits['rest'].append(pat_exc.mean(axis=0))

    return rates, corrs, inhibs, excits

def get_linear_approximations():
    act_exc = {}
    act_inh = {}

    act_exc['hebb'] = get_activations('hebb')
    act_exc['rate'] = get_activations('rate')
    act_inh['hebb'] = get_activations('hebb', ei='inh')
    act_inh['rate'] = get_activations('rate', ei='inh')

    linear_approximations = {}

    for plast in ['hebb','rate']:
        mat_file = f'{path}/connectivity/training_{plast}_nonburst{npat}_matrix.pkl'

        with open(mat_file, 'rb') as file:
            Z, N_exc, patterns, exc_alpha, delays, params = pickle.load(file)

        pat_sizes = patterns.sum(axis=1)
        norm_patterns = (patterns.T / pat_sizes).T

        Z_lin = np.copy(Z)

        Z_lin[:8000, :8000] = (Z[:8000, :8000].T * act_exc[plast]['exc']).T
        Z_lin[:8000, 8000:] = (Z[:8000, 8000:].T * act_exc[plast]['inh']).T
        Z_lin[8000:, :8000] = (Z[8000:, :8000].T * act_inh[plast]['exc']).T
        Z_lin[8000:, 8000:] = (Z[8000:, 8000:].T * act_inh[plast]['inh']).T

        linear_approximations[plast] = Z_lin

    return linear_approximations, patterns, act_exc, act_inh


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