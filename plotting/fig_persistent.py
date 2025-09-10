import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

from plotutils import *


if __name__ == '__main__':
    fig = plt.figure(figsize=(8,4))
    gs = mpl.gridspec.GridSpec(2, 2, height_ratios=[1, 1], width_ratios=[2,1], figure=fig)

    ax_traces = fig.add_subplot(gs[:,0])

    colors = mpl.colormaps['Dark2'].colors

    persistent_traces = pd.read_csv('plotting/data/persistent/persistent_traces.csv', index_col=None)

    for ix, c in zip([1, 2, 470, 979, 618, 235, 877, 785, 268], colors):
        ax_traces.plot(persistent_traces['xx'], 100*persistent_traces[str(ix)], color=c, lw=0.9)

    ax_traces.plot(persistent_traces['xx'], 100*persistent_traces['best'], color='black', lw=0.9, ls='dotted')

    ax_traces.scatter([11], [270], color=colors[0], s=15)
    ax_traces.scatter([21], [270], color=colors[1], s=15)

    ax_avg = fig.add_subplot(gs[0,1])
    ax_sfa = fig.add_subplot(gs[1,1])

    avg_persist = pd.read_csv('plotting/data/persistent/averaged.csv', index_col=None)
    avg_sfa = pd.read_csv('plotting/data/persistent/averaged_sfa.csv', index_col=None)

    xx = np.arange(len(avg_persist['stimulated'])) * 0.01
    ax_avg.plot(xx, 100*avg_persist['stimulated'], color='black')
    ax_avg.plot(xx, 100*avg_persist['other'], color='black', linestyle='dotted')

    xx = np.arange(len(avg_sfa['stimulated'])) * 0.01
    ax_sfa.plot(xx, 100*avg_sfa['stimulated'], color='black')
    ax_sfa.plot(xx, 100*avg_sfa['other'], color='black', linestyle='dotted')

    for ax in [ax_traces, ax_avg, ax_sfa]:
        despine_ax(ax, 'tr')
        ax.set_ylabel('firing rate (Hz)')
        ax.set_xlabel('time (s)')

    ax_traces.set_title('A', fontweight='bold', x=-0.13, y=0.93)
    ax_avg.set_title('B', fontweight='bold', x=0.1, y=0.8)
    ax_sfa.set_title('C', fontweight='bold', x=0.1, y=0.8)

    ax_avg.set_xlim(0,2)
    ax_avg.axvspan(1, 1.1, alpha=0.1, color='black')
    ax_sfa.set_xlim(0,2)
    ax_sfa.axvspan(0.5, 0.6, alpha=0.1, color='black')

    plt.savefig('plotting/img/persistent.png', dpi=300)
