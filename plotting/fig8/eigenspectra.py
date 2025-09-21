import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import pickle

from utils import data_path
from plotutils import *
# from plotutils import rule_color, rule_name

def plot_eigenassembly(axes, assembly, system, multiple=1):
    dur = 50
    start = 490
    end = start+dur

    xx = np.arange(8000)

    cutoff = assembly['ev_cutoff']

    if cutoff > 0:
        mask = assembly['eigenvector'] < cutoff
    else:
        mask = assembly['eigenvector'] > cutoff

    axes[0].scatter(xx[mask], assembly['eigenvector'][mask]*multiple, color='black', s=1)
    axes[0].scatter(xx[~mask], assembly['eigenvector'][~mask]*multiple, color=rule_color(system), s=1)
    ylim = axes[0].get_ylim()

    if cutoff > 0:
        axes[0].axhspan(cutoff, ylim[1], color='black', alpha=0.1)
    else:
        axes[0].axhspan(ylim[0], cutoff, color='black', alpha=0.1)

    axes[1].plot(assembly['activation'][start:end], color=rule_color(system), label='eigenvalue-based\nassembly')

    for ii, activity in enumerate(assembly['overlaps_activity']):
        if ii == 0:
            label = 'assemblies with\nlarge overlap'
        else:
            label = None

        axes[1].plot(activity[start:end], color='black', lw=0.2, label=label)

    plot_table(axes[2], [assembly['neurons'], *assembly['overlapping_patterns']], colors=[rule_color(system)]+['black']*5)


if __name__ == '__main__':
    syslist = ['hebb','hebb_smooth_rate','rate'][::-1]

    # fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(8,8))
    fig = plt.figure(figsize=(8,9))

    outer_gs = GridSpec(4, 1, height_ratios=[3, 0.2, 1,1.], hspace=0.3, figure=fig)

    # Top 3x3 grid as a SubGridSpec
    top_gs = outer_gs[0].subgridspec(2, 3, wspace=0.2, hspace=0.1)
    axs_top = [[fig.add_subplot(top_gs[i, j]) for i in range(2)] for j in range(3)]

    # Bottom 1x3 grid for other plots
    bottom_gs1 = outer_gs[-2].subgridspec(1, 4, width_ratios=[1, 0.8, 0, 1], wspace=0.35)
    ax_bottom_left1 = fig.add_subplot(bottom_gs1[0, 0])
    ax_bottom_middle1 = fig.add_subplot(bottom_gs1[0, 1])
    ax_bottom_right1 = fig.add_subplot(bottom_gs1[0, 3])

    bottom_gs2 = outer_gs[-1].subgridspec(1, 4, width_ratios=[1, 0.8, 0, 1], wspace=0.35)
    ax_bottom_left2 = fig.add_subplot(bottom_gs2[0, 0])
    ax_bottom_middle2 = fig.add_subplot(bottom_gs2[0, 1])
    ax_bottom_right2 = fig.add_subplot(bottom_gs2[0, 3])

    for i, system in enumerate(syslist):
        for j, npat in enumerate([1000, 2000]):
            ev = np.loadtxt(f'{data_path()}/lognormal/eigensystem/{system}{npat}.csv', dtype=np.complex64) * 6 / 100

            ax = axs_top[i][j]
            ax.scatter(ev.real, ev.imag, s=0.5, color=rule_color(system))
            ax.set_xlim(-1.,1.)
            ax.set_ylim(-1,1)

            ax.grid(linewidth=0.2)

            # despine_ax(ax, 'tr')

            if j != 1:
                ax.set_xticklabels([])
                despine_ax(ax, 'b')
                # ax.set_xticks([-10, 0, 10])
            else:
                ax.set_xlabel(r'$\operatorname{Re}(\lambda)$')
            if i != 0:
                ax.set_yticklabels([])
                despine_ax(ax, 'l')
                # ax.set_yticks([-10, 0, 10])
            else:
                ax.set_ylabel(r'$\operatorname{Im}(\lambda)$')
                ax.text(-14*6/100, 12*6/100, f'{npat} embedded assemblies')

            if j == 0:
                ax.set_title(rule_name(system), loc='center', fontweight='bold')
    
    axs_top[0][0].set_title('A', fontweight='bold', loc='left')

    with open('plotting/data/overlap_activity.pkl', 'rb') as f:
        overlaps = pickle.load(f)

    system = 'rate'

    axes1 = [ax_bottom_left1, ax_bottom_right1, ax_bottom_middle1]
    axes2 = [ax_bottom_left2, ax_bottom_right2, ax_bottom_middle2]
    letter_sets = [['B','D','C'], ['E','G','F']]

    for ii, (system, axes, letters) in enumerate(zip(['rate','hebb'], [axes1, axes2], letter_sets)):
        overlap_res = overlaps[(2000,system)][0]  # the first assembly from several identified

        if ii == 0:
            multiple = 1
        else:
            multiple = 1

        plot_eigenassembly(axes, overlap_res, system, multiple=multiple)

        for ax, l in zip(axes, letters):
            ax.set_title(l, loc='left', fontweight='bold')

        axes[0].set_ylabel(r'$\operatorname{Re}(\mathbf{v})$')
        axes[1].set_ylabel('assembly activation')
        axes[1].set_ylim(0, 1)

        if ii == 1:
            axes[1].set_xlabel('time (s)')
            axes[0].set_xlabel('neuron index')
            axes[0].set_ylim(-0.013, 0.013)
        if ii == 0:
            axes[0].set_xticklabels([])
            axes[1].set_xticklabels([])

    axes2[0].yaxis.set_label_coords(-0.24, 0.5)

    axes1[1].legend(loc=(0.15, 0.9))
    axes2[1].legend(loc=(0.05, 0.6))



    fig.tight_layout()
    plt.savefig('img/eigenspectra.png', bbox_inches='tight')