import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pickle

from plotutils import *


if __name__ == '__main__':
    syslist = ['hebb', 'hebb_smooth_rate', 'rate']
    linestyles = ['solid','dashed','dotted']
    
    activation_stats = pd.read_csv(f'plotting/data/activation_stats.csv', header=[0,1], index_col=0)
    
    gradual = pd.read_csv(f'plotting/data/gradual.csv', header=[0,1], index_col=0)

    npat_list = activation_stats.index.values
    xx = gradual.index.values

    # fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(8,4))
    fig = plt.figure(figsize=(8, 4))
    gs_main = GridSpec(nrows=2, ncols=3, figure=fig, wspace=0.5, hspace=0.6)

    # SubGridSpec for the top-left cell (2 rows × 1 column)
    gs_sub = gs_main[0, 0].subgridspec(2, 1, hspace=0.3)
    ax_00a = fig.add_subplot(gs_sub[0, 0])
    ax_00b = fig.add_subplot(gs_sub[1, 0])

    # Regular axes for the rest
    ax_01 = fig.add_subplot(gs_main[0, 1])
    ax_02 = fig.add_subplot(gs_main[0, 2])
    ax_10 = fig.add_subplot(gs_main[1, 0])
    ax_11 = fig.add_subplot(gs_main[1, 1])
    ax_12 = fig.add_subplot(gs_main[1, 2])

    axes = [ax_00a, ax_00b, ax_01, ax_02, ax_10, ax_11, ax_12]

    activations = np.loadtxt('plotting/data/assembly_traces/activations.csv')
    rates = np.loadtxt('plotting/data/assembly_traces/rates.csv')

    tt = np.arange(500) / 100

    colors = mpl.colormaps['Dark2'].colors
    for rr, aa, c in zip(rates, activations, colors):
        ax_00a.plot(tt, rr, color=c)
        ax_00b.plot(tt, aa, color=c)

    ax_00b.axhline(0.9, lw=1, color='black', ls='dashed')

    ax_00a.set_ylabel('assembly\nrate (Hz)')
    ax_00b.set_ylabel('activation')
    ax_00b.set_xlabel('time (s)')

    despine_ax(ax_00a, 'trb')
    despine_ax(ax_00b, 'tr')

    with open(f'plotting/data/iais.pkl', 'rb') as f:
        inter_event_intervals = pickle.load(f)

    bins = np.linspace(0, 80, 15)

    for system in syslist:
        # axes[0,0].hist(inter_event_intervals[(system, 1000)], bins=bins, histtype='step')
        marker = rule_marker(system)
        color = rule_color(system)

        ax_01.plot(npat_list, activation_stats[system]['mean_freq'], marker=marker, color=color, label=rule_name(system))
        ax_02.plot(npat_list, activation_stats[system]['mean_dur']*10, marker=marker, color=color)

        for npat, ls in zip(['1000','1400','1800'], linestyles):
            ax_10.plot(xx / 60, gradual[system][npat], ls=ls, color=color)

        ax_11.plot(npat_list, activation_stats[system]['nunique'], marker=marker, color=color)
        ax_12.plot(npat_list, activation_stats[system]['entropy'], marker=marker, color=color)

    letters = ['A','','B','C','D','E','F']

    for ax, l in zip(axes, letters):
        ax.set_title(l, fontweight='bold', loc='left')

    ax_01.set_ylabel('replay freq. (Hz)')
    ax_01.set_xlabel('embedded assemblies')

    ax_02.set_ylabel('replay avg. dur (ms)')
    ax_02.set_xlabel('embedded assemblies')

    ax_10.set_ylabel('unique replayed\nassemblies (cumulative)')
    # ax_10.set_xticks([0, 30, 60, 90, 120, 150])
    # ax_10.set_xticklabels([0, '', 60, '', 120, ''])
    ax_10.set_xlabel('time (min)')

    ax_11.set_ylabel('unique replayed\nassemblies')
    ax_11.set_xlabel('embedded assemblies')

    ax_12.set_ylabel('diversity')
    ax_12.set_xlabel('embedded assemblies')

    # ax_02.set_ylim(0, 600)
    ax_01.legend(loc=(0.45, 0.6))


    # fig.align_ylabels([ax_00a, ax_00b, ax_10])
    ax_00b.yaxis.set_label_coords(-0.23, 0.5)
    ax_00a.yaxis.set_label_coords(-0.18, 0.5)
    fig.align_ylabels([ax_01, ax_11])
    fig.align_ylabels([ax_02, ax_12])

    # fig.tight_layout()
    plt.savefig(f'img/replay.png', bbox_inches='tight')


