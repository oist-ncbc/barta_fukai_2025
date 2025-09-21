import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from plotutils import *


def plot_rule(alpha, ax, modulation=1, tau=20):
    xx = np.linspace(-100, 100, 500)
    yy = np.zeros_like(xx)
    yy[:250] = np.exp((xx[:250])/tau)
    yy[250:] = np.exp(-(xx[250:])/tau)
    yy += alpha

    ax.plot(xx, yy*modulation, color='black', lw=1.5)
    ax.fill_between(xx, np.clip(yy*modulation, a_min=None, a_max=0), color='C3', lw=0, alpha=0.3)
    ax.fill_between(xx, np.clip(yy*modulation, a_min=0, a_max=None), color='C2', lw=0, alpha=0.3)
    ax.axvline(0, color='black', linestyle='dashed')
    ax.axis('off')

if __name__ == '__main__':
    npat = 1000
    systems = ['rate','hebb_smooth_rate','hebb']
    N_exc = 8000

    root = '/home/t/tomas-barta/StructuredInhibition/plotting'

    fig = plt.figure(figsize=(8,7.5))

    outer_gs = gridspec.GridSpec(5, 1, height_ratios=[1, 1, 1, 0, 0.8], figure=fig, hspace=0.35)

    ax_image = fig.add_subplot(outer_gs[0])
    ax_image.axis('off')

    img = plt.imread(f'plotting/fig1/network_scheme.png')
    ax_image.imshow(img)
    ax_image.set_title('A', loc='left', fontweight='bold', x=-0.11, y=0.9)

    # --- Top (6x3 grid within the top third of figure) ---
    gs_top = outer_gs[1].subgridspec(6, 3, wspace=0.4)
    ax_gsf1 = fig.add_subplot(gs_top[0:3, 0])
    ax_gsf2 = fig.add_subplot(gs_top[3:6, 0])
    ax_ship = fig.add_subplot(gs_top[1:5, 1])
    ax_fhip = fig.add_subplot(gs_top[1:5, 2])

    ax_help1 = fig.add_subplot(gs_top[0:3, 1])
    ax_help2 = fig.add_subplot(gs_top[0:3, 2])

    ax_help1.axis('off')
    ax_help2.axis('off')

    # --- Middle row: 1x3 ---
    gs_middle = outer_gs[2].subgridspec(1, 3, wspace=0.4)
    axes_middle = [fig.add_subplot(gs_middle[0, i]) for i in range(3)]

    # --- Bottom row: 1x3 ---
    gs_bottom = outer_gs[-1].subgridspec(1, 3, wspace=0.4)
    axes_bottom = [fig.add_subplot(gs_bottom[0, i]) for i in range(3)]

    text_y = 0.5
    text_fs = 8

    ax_gsf1.set_title('global', loc='right', fontweight='bold', fontsize=9, y=0.7)
    ax_help1.set_title('slow local', loc='right', fontweight='bold', fontsize=9, y=0.7)
    ax_help2.set_title('fast local', loc='right', fontweight='bold', fontsize=9, y=0.7)

    # ax_gsf1.text(-100, text_y, 'global', fontsize=text_fs, ha='left', fontweight='bold')
    ax_gsf1.text(20, text_y, r'$\tau=10$s', fontsize=text_fs)

    # ax_ship.text(-100, text_y, 'slow local', fontsize=text_fs, ha='left', fontweight='bold')
    ax_ship.text(20, text_y, r'$\tau=10$s', fontsize=text_fs)

    # ax_fhip.text(-100, text_y, 'fast local', fontsize=text_fs, ha='left', fontweight='bold')
    ax_fhip.text(20, text_y, r'$\tau=20$ms', fontsize=text_fs)

    plot_rule(0.05, ax_gsf1)
    plot_rule(0.05, ax_gsf2, modulation=-1)
    plot_rule(-0.1, ax_ship)
    plot_rule(-0.1, ax_fhip, tau=10)

    ax_ship.set_ylim(-0.8, 1.3)


    letters = ['B', '', 'C', 'D']
    for ax, l in zip([ax_gsf1, ax_gsf2, ax_help1, ax_help2], letters):
        despine_ax(ax)
        ax.set_title(l, loc='left', fontweight='bold', y=0.7)

    for i, (system, ax) in enumerate(zip(systems, axes_middle)):
        exc_spikes = np.loadtxt(f'plotting/data/rasters/{system}_excitatory.csv')
        inh_spikes = np.loadtxt(f'plotting/data/rasters/{system}_inhibitory.csv')

        exc_mask = exc_spikes[0] < 800
        ax.scatter(exc_spikes[1][exc_mask], exc_spikes[0][exc_mask], color=type_color['exc'], s=.1)
        ax.set_xlim(15, 20)

        inh_mask = inh_spikes[0] < 200
        ax.scatter(inh_spikes[1][inh_mask], inh_spikes[0][inh_mask]+800, color=type_color['inh'], alpha=0.2, s=.1)
        ax.set_xlim(15, 20)

        despine_ax(ax, 'trb')
        # ax.set_xlabel('time (s)')

        # ax.set_xticks([10, 12.5, 15, 17.5, 20])
        # ax.set_xticklabels([10,'',15,'',20])

        if i != 0:
            ax.set_yticklabels([])
            despine_ax(ax, 'l')
        else:
            ax.set_yticks([0, 800, 1000])
            ax.set_yticklabels([])
            ax.text(-0.19, 0.4, 'exc.', va='center', rotation='vertical', transform=ax.transAxes)
            ax.text(-0.19, 0.9, 'inh.', va='center', rotation='vertical', transform=ax.transAxes)

            y = -50
            x0 = 19
            x1 = 20
            ax.hlines(y, x0, x1, linewidth=2, color='black', clip_on=False)
            ax.text((x0 + x1)/2, y-130, "1 s", ha='center', va='bottom', fontsize=10)

        ax.set_ylim(0, 1000)

    axes_middle[0].set_title('E', loc='left', fontweight='bold')
    axes_middle[1].set_title('F', loc='left', fontweight='bold')
    axes_middle[2].set_title('G', loc='left', fontweight='bold')

    rates = {
        system: np.loadtxt(f'plotting/data/firing_rates/{system}{npat}.csv') for system in systems
    }

    bins = np.linspace(-2, 2, 100)

    for system in systems:
        axes_bottom[0].hist(np.log10(rates[system][:N_exc]), color=rule_color(system), label=rule_name(system), bins=bins, **hist_params)
        # axes_bottom[0].axvline(np.log10(rates[system][:N_exc].mean()), color=rule_color(system), linestyle='dashed')

    log_ticks(axes_bottom[0])

    # despine_ax(axes[0])

    weights = {
        system: np.loadtxt(f'plotting/data/ei_weights/{system}{npat}.csv') for system in systems
    }
    totinhs = {
        system: np.loadtxt(f'plotting/data/tot_inhib/{system}{npat}.csv') for system in systems
    }

    bins1 = np.linspace(0, 2, 50)
    bins2 = np.linspace(0, 600, 100)

    for system in systems:
        axes_bottom[1].hist(weights[system][:N_exc], color=rule_color(system), bins=bins1, label=rule_name(system), **hist_params)
        # axes_bottom[1].axvline(weights[system][:N_exc].mean(), color=rule_color(system), linestyle='dashed')

        axes_bottom[2].hist(totinhs[system], color=rule_color(system), bins=bins2, label=rule_name(system), **hist_params)
        # axes_bottom[2].axvline(totinhs[system].mean(), color=rule_color(system), linestyle='dashed')

    axes_bottom[0].set_title('H', loc='left', fontweight='bold')
    axes_bottom[1].set_title('I', loc='left', fontweight='bold')
    axes_bottom[2].set_title('J', loc='left', fontweight='bold')

    # axes_bottom[0].set_xscale('log')
    axes_bottom[0].set_ylabel('density')
    axes_bottom[0].set_xlabel('neuron firing rate (Hz)')

    axes_bottom[1].set_ylabel('density')
    axes_bottom[1].set_xlabel('synapse weight (nS)')

    axes_bottom[2].set_ylabel('density')
    axes_bottom[2].set_xlabel('sum of pre-syn. weights (nS)')

    axes_bottom[2].legend(loc=(0.35,0.6))

    plt.savefig(f'img/intro.png', dpi=300, bbox_inches='tight')