import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pickle
import numpy as np

from plotutils import *


if __name__ == '__main__':
    fig = plt.figure(figsize=(8, 7.5))

    with open('plotting/data/retrieve2.pkl', 'rb') as f:
        res = pickle.load(f)

    syslist = ['rate','hebb_smooth_rate','hebb']

    # Main GridSpec with 3 rows, 3 columns
    gs_illustrate = gridspec.GridSpec(nrows=1, ncols=2, figure=fig, wspace=0.55, hspace=0.55, top=1, left=0.6, bottom=0.92)
    ax1 = fig.add_subplot(gs_illustrate[0])
    ax2 = fig.add_subplot(gs_illustrate[1])

    stimulus_color = '#fec44f'
    stimulus_color = 'red'
    assembly_id = 3
    npat = 2000
    ax1.plot(res[npat]['hebb']['sact'][assembly_id], color=stimulus_color)
    ax1.plot(res[npat]['hebb']['nsact'][assembly_id], color='black')
    ax1.axvspan(50, 60, color=stimulus_color, alpha=0.1)
    ax2.plot(res[npat]['hebb']['stim'][assembly_id]*100, color=stimulus_color)
    ax2.plot(res[npat]['hebb']['nonstim'][assembly_id]*100, color='black')
    ax1.set_ylabel('activated\nneurons\nratio')
    ax2.set_ylabel('firing rate (Hz)')
    ax1.axvspan(50, 60, color=stimulus_color, alpha=0.1)
    ax2.axvspan(50, 60, color=stimulus_color, alpha=0.1)
    ax1.text(55, -0.25, '100ms', horizontalalignment='center')

    ax1.set_title('A', loc='left', fontweight='bold')

    despine_ax(ax1, 'b')
    despine_ax(ax2, 'b')

    gs_main = gridspec.GridSpec(nrows=4, ncols=3, width_ratios=[1, 1, 1.5], figure=fig, wspace=0.55, hspace=0.55, top=0.85)

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
        gs_sub = gs_main[row, 2].subgridspec(2, 3, hspace=0.3, wspace=0.6)
        for subrow in range(3):
            ax1 = fig.add_subplot(gs_sub[0, subrow])
            ax2 = fig.add_subplot(gs_sub[1, subrow])
            right_axes[-1].append([ax1, ax2])


    letters1 = [
        ['B', 'E'],
        ['C', 'F'],
        ['D', 'G']
    ]

    letters2 = ['H', 'I', 'J']

    for i, npat in enumerate([1000, 1400, 2000]):
        for j, system in enumerate(syslist):
            c = rule_color(system)

            axes_large[i][0].set_title(f'{npat} emb. assemblies', loc='right', fontsize=8)

            nstim = len(res[npat][system]['nsact'])

            quality = np.max(res[npat][system]['nsact'], axis=1)
            peak = np.max(res[npat][system]['nonstim'], axis=1)*100

            axes_large[i][0].hist(quality, bins=np.linspace(0, 1., 30), **hist_params, color=c, label=rule_name(system))
            axes_large[i][1].hist(peak, bins=np.linspace(0, 150, 30), **hist_params, color=c)

            # axes_large[i][0].set_ylabel('density')

            # axes_large[i][1].set_ylabel('density')

            axes_large[i][0].set_title(letters1[i][0], loc='left', fontweight='bold', x=-0.2)
            axes_large[i][1].set_title(letters1[i][1], loc='left', fontweight='bold', x=-0.2)

            if j == 0:
                right_axes[i][j][0].set_title(letters2[i], loc='left', fontweight='bold')
            if i == 2 and j == 1:
                right_axes[i][j][1].set_xlabel('spontaneous replay count')

            right_axes[i][j][0].scatter(
                                     res[npat][system]['act_counts'][:nstim],
                                     quality,
                                     color=c, s=1, clip_on=False)

            right_axes[i][j][1].scatter(
                                     res[npat][system]['act_counts'][:nstim],
                                     peak,
                                     color=c, s=1)
            
            if j != 0:
                right_axes[i][j][0].set_yticklabels([])
                right_axes[i][j][1].set_yticklabels([])

    right_axes[1][0][0].set_ylabel('quality')
    right_axes[1][0][1].set_ylabel('peak\nf.r. (Hz)')
    axes_large[1][0].set_ylabel('density')
    axes_large[2][0].set_xlabel('pattern completion\nquality')
    axes_large[2][1].set_xlabel('pattern completion\nquality')
    axes_large[0][0].legend()

    for i in range(3):
        for j in range(3):
            right_axes[i][j][0].set_ylim(0, 1)
            right_axes[i][j][0].set_xticklabels([])
            if j != 0:
                right_axes[i][j][0].set_yticklabels([])

    for i, maxrate in enumerate([180, 130, 120]):
        for j in range(3):
            right_axes[i][j][1].set_ylim(0, maxrate)
            right_axes[i][j][1].set_yticks([0, maxrate])
            if j != 0:
                right_axes[i][j][1].set_yticklabels([])

            # right_axes[i][j][0].set_xlim(0, 1.1)
            # right_axes[i][j][1].set_xlim(0, 1.5*100)

            # right_axes[i][j][0].tick_params(labelsize=8)
            # right_axes[i][j][1].tick_params(labelsize=8)

            # if i == 2:
            #     axes_large[i][0].set_xlabel('quality of pattern completion')
            #     axes_large[i][1].set_xlabel('peak firing rate (Hz)')
            # else:
            #     axes_large[i][0].set_xticklabels([])
            #     axes_large[i][1].set_xticklabels([])
            #     right_axes[i][j][0].set_xticklabels([])
            #     right_axes[i][j][1].set_xticklabels([])

            # if j != 2:
            #     right_axes[i][j][0].set_xticklabels([])
            #     right_axes[i][j][1].set_xticklabels([])

            # if j == 2 and i == 2:
            #     right_axes[i][j][0].set_xlabel('activ.')
            #     right_axes[i][j][1].set_xlabel('f.r. (Hz)')

    fig.align_ylabels([right_axes[0][1][0], right_axes[1][1][0], right_axes[2][1][0]])

    plt.savefig('img/cued_recall.svg', bbox_inches='tight', dpi=300, transparent=True)