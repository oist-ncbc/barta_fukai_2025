import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
import pickle
import numpy as np
import pandas as pd

from utils import plot_covariance_ellipse, data_path, load_patterns
from plotutils import *
from eigenvalues import get_W

def plot_responses(ax, responses):
    xx = np.arange(800)
    ax.plot(xx[:400], responses[0]*100, color=type_color['exc'])
    ax.plot(xx[399:], np.concatenate([[responses[0][-1]], responses[1]])*100, color=type_color['inh'])

    despine_ax(ax, 'trb')

def overlap_panel(ax, npat=1000):
    system = 'hebb'

    W = get_W(system, npat)
    patterns = load_patterns(npat)

    pattern_derivs = W[:8000,:8000] @ patterns.dense().T

    pix = 100

    patdev = pattern_derivs[:,pix]
    pat = patterns[pix]
    patsize = len(pat)

    order = np.argsort(patdev)[::-1]
    mask = np.isin(order, pat)

    xx = np.arange(8000)

    ax.scatter(xx[~mask], patdev[order][~mask], s=5, color='black')
    ax.scatter(xx[mask], patdev[order][mask], s=5, color='red')
    ax.axvspan(0, patsize, color='red', alpha=0.1)
    ax.set_xlim(-1, 120)

    ax2 = ax.twinx()
    ax2.plot(np.cumsum(~mask[:patsize]) / patsize, color=contamination_color)
    ax2.plot([patsize,120], [np.sum(~mask[:patsize]) / patsize]*2, linestyle='dotted', color=contamination_color)
    ax2.set_ylim(0, 1)

    for label in ax2.get_yticklabels():
        label.set_color(contamination_color)

    return ax, ax2

def plot_inputs(ax, text=True):
    stim_steps = 5

    pair = np.array([1]+[0]*(stim_steps-1))

    input_list = np.linspace(-0.35, 0.35, 8)
    init_steps = len(input_list)

    input_arr_exc = np.zeros(init_steps)
    input_arr_inh = np.zeros(init_steps)

    for ii in range(1):
        for inp in input_list:
            input_arr_exc = np.append(input_arr_exc, pair*inp)
            input_arr_inh = np.append(input_arr_inh, np.zeros(stim_steps))

        for inp in input_list:
            input_arr_inh = np.append(input_arr_inh, pair*inp)
            input_arr_exc = np.append(input_arr_exc, np.zeros(stim_steps))

    ax.plot(np.repeat(input_arr_exc, 100)-0.5, color=type_color['exc'])
    ax.plot(np.repeat(input_arr_inh, 100), color=type_color['inh'])

    ax.text(x=50, y=0.1, s=r'$g_{inh}$'+'\nperturbation', fontsize=8, color=type_color['inh'])
    ax.text(x=3500, y=-0.75, s=r'$g_{exc}$'+'\nperturbation', fontsize=8, color=type_color['exc'])

    # if text:
    #     ax.text(1000, 0.05, r'$g_\mathrm{exc}$')
    #     ax.text(1000, 0.05-0.5, r'$g_\mathrm{inh}$')

    despine_ax(ax)

    # n = 44
    # for i in range(n):
    #     ax.axvspan(2*i*100, (2*i+1)*100, alpha=0.1, lw=0)
    # for i in range(n):
    #     ax.axvspan((2*i+1)*100, (2*i+2)*100, alpha=0.01, lw=0)

if __name__ == '__main__':
    contamination_color = 'navy'
    syslist = ['hebb','hebb_smooth_rate','rate']

    fig = plt.figure(figsize=(8, 4))
    # gs = gridspec.GridSpec(6, 6, figure=fig, hspace=0.5, wspace=0.8)  # finer grid for flexibility

    # # Top row: 3 equal plots
    # ax1 = fig.add_subplot(gs[0:2, 0:2])
    # ax21 = fig.add_subplot(gs[0:1, 2:4])
    # ax22 = fig.add_subplot(gs[1:2, 2:4])
    # ax3 = fig.add_subplot(gs[0:2, 4:6])

    # # Second row: 1 wide plot
    # ax4 = fig.add_subplot(gs[2:4, 0:6])

    # # Bottom row: 2 medium plots + 2x2 small plots
    # ax5 = fig.add_subplot(gs[4:6, 0:3])
    # ax6 = fig.add_subplot(gs[4:6, 3:6])
    # ax7 = fig.add_subplot(gs[4, 4])
    # ax8 = fig.add_subplot(gs[4, 5])
    # ax9 = fig.add_subplot(gs[5, 4])
    # ax10 = fig.add_subplot(gs[5, 5])

    # Optional: add titles or content to test layout
    # for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10], 1):
    #     ax.set_title(f"Plot {i}")
    #     ax.set_xticks([])
    #     ax.set_yticks([])

    outer_gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1.2], figure=fig, hspace=0.5)

    # --- Top part: subgridspec with 3 columns ---
    top_gs = outer_gs[0].subgridspec(2, 8, wspace=0.1,
                                     width_ratios=[1, 0.1, 1, 0.1, 0.7, 1, 0.1, 1],
                                     height_ratios=[0.8, 0.2])
    ax1 = fig.add_subplot(top_gs[:,0])
    # ax21 = fig.add_subplot(top_gs[0, 1])
    # ax22 = fig.add_subplot(top_gs[1, 1])
    ax21 = fig.add_subplot(top_gs[:,2])
    ax22 = fig.add_subplot(top_gs[:,5])
    ax3 = fig.add_subplot(top_gs[0,7])

    ax_plus = fig.add_subplot(top_gs[:,1])
    ax_plus.set_xlim(0, 1)
    ax_plus.set_ylim(0, 1)
    ax_plus.text(0.5, 0.5, '+', fontweight='bold', fontsize=20,
                 horizontalalignment='center', verticalalignment='center')
    despine_ax(ax_plus)

    ax_arrow = fig.add_subplot(top_gs[:,3])
    ax_arrow.set_xlim(0, 1)
    ax_arrow.set_ylim(0, 1)
    ax_arrow.text(0.5, 0.5, r'$\rightarrow$', fontweight='bold', fontsize=20,
                 horizontalalignment='center', verticalalignment='center')
    despine_ax(ax_arrow)

    despine_ax(ax3, 'tr')
    ax3.set_yticks([])
    ax3.set_ylabel('neurons')
    ax3.set_xlabel(r'$\chi^{EE}$ (Hz/nS)')
    ax3.set_xticks([0, 0.5, 1])

    # --- Middle wide plot ---
    # ax4 = fig.add_subplot(outer_gs[1])

    # --- Bottom: flexible layout with multiple small axes ---
    bottom_gs = outer_gs[1].subgridspec(1, 2, wspace=0.5, hspace=0.4)
    ax5 = fig.add_subplot(bottom_gs[0])
    ax6 = fig.add_subplot(bottom_gs[1])

    time, ge, gi = np.loadtxt('plotting/data/ou.csv')
    ax1.plot(time, ge, color=type_color['exc'], lw=0.5)
    ax1.plot(time, gi, color=type_color['inh'], lw=0.5)

    despine_ax(ax1, 'trb')

    # ax1.set_xlabel(r'$g_\mathrm{exc}$ (nS)')
    ax1.set_ylabel(r'conductance (nS)')

    # with open(f'plotting/data/conductances.pkl', 'rb') as f:
    #     data = pickle.load(f)

    # ax1.scatter(data['ge'], data['gi'], s=0.5, color='black')
    # plot_covariance_ellipse(*data['cov_nrb'], ax=ax1, color='purple', label='non-robust')
    # plot_covariance_ellipse(*data['cov_rob'], ax=ax1, color='red', label='robust')

    plot_inputs(ax21)
    perturb_sc_responses = np.loadtxt('plotting/data/perturb_responses.csv')
    plot_responses(ax22, perturb_sc_responses)

    folder = f"{data_path()}/lognormal"
    
    bins = np.linspace(-0.2, 1.2, 30)

    for system in ['hebb','hebb_smooth_rate','rate']:
        filename = f"{folder}/linear_approx/{system}1000.csv"

        data = pd.read_csv(filename, index_col=0)
        ax3.hist(data.loc['exc']['activation_exc'], bins=bins, color=rule_color(system), **hist_params)

    # img = mpimg.imread("plotting/matrix.png")
    # ax4.imshow(img, aspect='equal')
    # despine_ax(ax4)

    ax5, ax51 = overlap_panel(ax5)
    ax5.set_ylabel('rate derivative (Hz/s)')
    ax51.set_ylabel('contamination', color=contamination_color)
    ax5.set_xlabel('ordered neuron index')
    despine_ax(ax5, 't')
    despine_ax(ax51, 't')

    # overlap_stats = pd.read_csv('plotting/data/overlaps/overlaps1000.csv', header=0)
    overlap_stats = pd.read_csv('plotting/data/overlaps/angles1000.csv', header=0)

    # bins = np.linspace(0,1,80)
    for system in syslist:
        # ax6.hist(1-overlap_stats[system], bins=bins, color=rule_color(system), label=rule_name(system), **hist_params)
        bins = np.linspace(40, 90, 80)
        ax6.hist(overlap_stats[system], bins=bins, color=rule_color(system), label=rule_name(system), **hist_params)

    # ax6.set_xlabel('assembly contamination')
    ax6.set_xlabel('angle')
    ax6.set_ylabel('density')
    ax6.legend(loc=(0.4, 0.8))
    despine_ax(ax6, 'tr')
    # ax6.set_yticks([])

    letters = ['A','','B','C','D']
    for ax, l in zip([ax1, ax21, ax3, ax5, ax6], letters):
        ax.set_title(l, loc='left', fontweight='bold')

    ax22.set_ylabel('firing\nrate (Hz)')

    # plt.tight_layout()
    plt.savefig('plotting/img/contamination.png', dpi=300, bbox_inches='tight')