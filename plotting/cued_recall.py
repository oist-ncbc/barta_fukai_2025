import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pickle
import numpy as np

from plotutils import *


if __name__ == '__main__':
    fig = plt.figure(figsize=(8, 7))

    syslist = ['rate','hebb_smooth_rate','hebb']

    # Main GridSpec with 3 rows, 3 columns
    gs_main = gridspec.GridSpec(nrows=3, ncols=3, width_ratios=[1, 1, 0.7], figure=fig, wspace=0.45, hspace=0.35)

    # Large plots in columns 0 and 1
    axes_large = []
    for row in range(3):
        axes_large.append([])
        for col in [0, 1]:
            ax = fig.add_subplot(gs_main[row, col])
            axes_large[-1].append(ax)

    # SubGridSpec for the rightmost column (3 main rows × 3 stacked plots each)
    right_axes = []
    for row in range(3):
        right_axes.append([])
        gs_sub = gs_main[row, 2].subgridspec(3, 2, hspace=0.2)
        for subrow in range(3):
            ax1 = fig.add_subplot(gs_sub[subrow, 0])
            ax2 = fig.add_subplot(gs_sub[subrow, 1])
            right_axes[-1].append([ax1, ax2])

    with open('plotting/data/retrieve2.pkl', 'rb') as f:
        res = pickle.load(f)


    letters1 = [
        ['A', 'B'],
        ['C', 'D'],
        ['E', 'F']
    ]

    letters2 = ['G', 'H', 'I']

    for i, npat in enumerate([1000, 1400, 2000]):
        for j, system in enumerate(syslist):
            c = rule_color(system)

            axes_large[i][0].set_title(f'{npat} emb. assemblies', loc='right', fontsize=8)

            nstim = len(res[npat][system]['nsact'])

            axes_large[i][0].hist(np.max(res[npat][system]['nsact'], axis=1), bins=np.linspace(0, 1., 30), **hist_params, color=c)
            axes_large[i][1].hist(np.max(res[npat][system]['nonstim'], axis=1)*100, bins=np.linspace(0, 150, 30), **hist_params, color=c)

            axes_large[i][0].set_ylabel('density')

            axes_large[i][1].set_ylabel('density')

            axes_large[i][0].set_title(letters1[i][0], loc='left', fontweight='bold')
            axes_large[i][1].set_title(letters1[i][1], loc='left', fontweight='bold')

            if j == 0:
                right_axes[i][j][0].set_title(letters2[i], loc='left', fontweight='bold')
            if j ==1:
                right_axes[i][j][0].set_ylabel('spontaneous replay count')

            right_axes[i][j][0].scatter(np.max(res[npat][system]['nsact'], axis=1),
                                     res[npat][system]['act_counts'][:nstim],
                                     color=c, s=1)

            right_axes[i][j][1].scatter(np.max(res[npat][system]['nonstim']*100, axis=1),
                                     res[npat][system]['act_counts'][:nstim],
                                     color=c, s=1)
            
            right_axes[i][j][1].set_yticklabels([])

            right_axes[i][j][0].set_xlim(0, 1.1)
            right_axes[i][j][1].set_xlim(0, 1.5*100)

            right_axes[i][j][0].tick_params(labelsize=8)
            right_axes[i][j][1].tick_params(labelsize=8)

            if i == 2:
                axes_large[i][0].set_xlabel('quality of pattern completion')
                axes_large[i][1].set_xlabel('peak firing rate (Hz)')
            else:
                axes_large[i][0].set_xticklabels([])
                axes_large[i][1].set_xticklabels([])
                right_axes[i][j][0].set_xticklabels([])
                right_axes[i][j][1].set_xticklabels([])

            if j != 2:
                right_axes[i][j][0].set_xticklabels([])
                right_axes[i][j][1].set_xticklabels([])

            if j == 2 and i == 2:
                right_axes[i][j][0].set_xlabel('activ.')
                right_axes[i][j][1].set_xlabel('f.r. (Hz)')
            
            # right_axes[i][j][0].set_yscale('log')
            # right_axes[i][j][1].set_yscale('log')

    fig.align_ylabels([right_axes[0][1][0], right_axes[1][1][0], right_axes[2][1][0]])

    plt.savefig('plotting/img/cued_recall.png', bbox_inches='tight', dpi=300)