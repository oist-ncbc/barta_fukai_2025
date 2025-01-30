import argparse
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from analysis import get_spike_counts

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-r', '--results', type=str, nargs='+')
    parser.add_argument('--plast', type=str, nargs='+')
    parser.add_argument('--patterns', type=int, nargs='+')
    parser.add_argument('--folder', type=str, default='')
    parser.add_argument('--img', type=str)
    parser.add_argument('--prefix', type=str)
    parser.add_argument('--suffix', type=str, nargs='+')

    args = parser.parse_args()

    with open('config/server_config.yaml') as f:
        config = yaml.safe_load(f)

    path = f"{config['data_path']}/{args.folder}"

    if len(args.suffix) == 0:
        suffices = ['']*len(args.patterns)
    else:
        suffices = []
        for i in range(len(args.patterns)):
            if args.suffix[i] == '_':
                suffices.append('')
            else:
                suffices.append('_' + args.suffix[i])

    # fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 6))
    fig, ax = plt.subplots()

    for ii, (plast, pat_num, suffix) in enumerate(zip(args.plast, args.patterns, suffices)):
        to_be_df = {}

        if args.folder != '':
            res_file = f'{path}/data/trained_{plast}_{args.prefix}{pat_num}_results_300stim{suffix}.pkl'

        with open(res_file, 'rb') as file:
            results = pickle.load(file)

        mat_file = results['params']['input']

        with open(mat_file, 'rb') as file:
            Z, N_exc, patterns, exc_alpha, delays, params = pickle.load(file)


        dt_fine = 0.01

        i, t = results['spikes']['exc']
        _, sc_fine = get_spike_counts(t, i, 102, dt=dt_fine)

        # axes[0,0].plot(np.arange(400)*dt_fine, sc_fine[:,0:400].mean(axis=0)/dt_fine)

        traces_stim = []
        traces_nonstim = []

        interval_before = int(0.15 / dt_fine)
        interval_after = int(0.5 / dt_fine)

        tt_stim = np.arange(interval_before+interval_after)*dt_fine - 0.15

        to_be_df['time'] = tt_stim

        for s in range(100):
            pattern = np.argwhere(patterns[s].astype(bool)).flatten()
            cutoff = int(len(pattern) / 10)
            # cutoff = 0

            traces_stim.append(sc_fine[pattern[:cutoff]].mean(axis=0)[(s+1)*100-interval_before:(s+1)*100+interval_after]/dt_fine)
            traces_nonstim.append(sc_fine[pattern[cutoff:]].mean(axis=0)[(s+1)*100-interval_before:(s+1)*100+interval_after]/dt_fine)

            to_be_df[f'pattern{s}'] = traces_nonstim[-1]

            # ax.plot(tt_stim, traces_nonstim[-1], color=f'C{ii}', lw=1, alpha=0.1)

        pd.DataFrame(to_be_df).to_csv(f'data/stim_{plast}{pat_num}.csv')

        # import pdb; pdb.set_trace()
        traces_stim = np.array(traces_stim)
        traces_nonstim = np.array(traces_nonstim)

        # axes[0,1].plot(np.arange(interval_before+interval_after)*dt_fine, traces_stim.mean(axis=0), color=f'C{ii}')
        ax.plot(tt_stim, traces_nonstim.mean(axis=0), color=f'C{ii}', lw=2)
        ax.plot(tt_stim, traces_stim.mean(axis=0), color=f'C{ii}', lw=2, linestyle=':')
        # ax.fill_between(tt_stim, np.percentile(traces_nonstim, q=2.5, axis=0), np.percentile(traces_nonstim, q=97.5, axis=0), color=f'C{ii}', alpha=0.5)
        ax.axvspan(0, 0.1, color='black', alpha=0.1)
        # ax.axvspan(1, 1.01, color='black', alpha=0.1)


        gap = 10

        # a = np.array([results['analysis']['activations'][i][(i+1)*gap] for i in range(99)])
        # b = np.array([results['analysis']['activations'][i][(i-1)*gap] for i in range(99)])
        # c = a-b
        #
        # print(a.mean())
        # print(b.mean())

        bins = np.linspace(0,1,20)
        # axes[1,0].hist(c, bins=bins, alpha=0.5)
        # axes[1,0].axvline(c.mean(), linestyle='dashed', color=f'C{ii}')

    plt.savefig(args.img)