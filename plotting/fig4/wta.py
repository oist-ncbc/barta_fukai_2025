import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import stats
import pickle

from utils import plot_covariance_ellipse, data_path
from plotutils import *


def load_varstats():
    varstats = {}

    npat = 1000

    for system in ['hebb','rate','hebb_smooth_rate']:
        varstats[system] = pd.read_csv(f'{data_path()}/lognormal/var_stats/{system}_conductances{npat}_stats.csv', index_col=[0,1])

    return varstats


if __name__ == '__main__':
    syslist = ['rate','hebb_smooth_rate','hebb']
    # markers = ['o', 'v', 's']
    # linestyles = ['solid','dashed','dotted']

    npat = 1000
    npat_list = [800, 1000, 1200, 1400, 1600, 1800, 2000]
    data = pd.read_csv(f'plotting/data/totweights/totweights_{npat}.csv', header=[0,1,2], index_col=0)
    
    # gradual = pd.read_csv(f'{root}/data/gradual.csv', header=[0,1], index_col=0)

    # npat_list = activation_stats.index.values
    # xx = gradual.index.values

    fig = plt.figure(figsize=(8,6))

    outer_gs = GridSpec(2, 1, height_ratios=[1, 2], figure=fig)

    top_gs = outer_gs[0].subgridspec(1, 7, wspace=0.3, width_ratios=[1,1,1,0.4,1,1,1])
    ax1 = fig.add_subplot(top_gs[1:3])
    ax2 = fig.add_subplot(top_gs[4:6])

    with open(f'plotting/data/conductances.pkl', 'rb') as f:
        cov_data = pickle.load(f)

    ax1.scatter(cov_data['ge'], cov_data['gi'], s=0.5, color='black')
    plot_covariance_ellipse(*cov_data['cov_nrb'], ax=ax1, color='purple', label='non-robust')
    plot_covariance_ellipse(*cov_data['cov_rob'], ax=ax1, color='red', label='robust')

    ax1.legend(loc=(0.5, 0.9))

    varstats = load_varstats()

    for system in ['hebb_smooth_rate','hebb','rate']:
        x = varstats[system].loc['exc']['mean_e']
        y = varstats[system].loc['exc']['mean_i']
        ax2.scatter(x, y, s=1,
                    color=rule_color(system))
        
        statres = stats.pearsonr(x, y)
        print(system, statres[0], statres[1])

    ax1.set_xlabel(r'$g_\mathrm{exc}$ (nS)')
    ax1.set_ylabel(r'$g_\mathrm{inh}$ (nS)')

    ax2.set_xlabel(r'$\bar{g}_\mathrm{exc}$ (nS)')
    ax2.set_ylabel(r'$\bar{g}_\mathrm{inh}$ (nS)')

    ax1.set_title('A', fontweight='bold', x=0.1, y=0.95)
    ax2.set_title('B', fontweight='bold', x=0.1, y=0.95)

    despine_ax(ax1, 'tr')
    despine_ax(ax2, 'tr')

    bottom_gs = outer_gs[1].subgridspec(2, 6, wspace=0.9, hspace=0.6)
    axes_scatters = np.array([
        fig.add_subplot(bottom_gs[0,i*2:(i+1)*2]) for i in range(3)
    ])

    for ax, l in zip(axes_scatters, ['C','D','E']):
        despine_ax(ax, 'tr')
        ax.set_title(l, fontweight='bold', x=0.1, y=0.95)

    for ax, system in zip(axes_scatters, syslist):
        x = data[system,str(npat),'tot_excit']
        y = data[system,str(npat),'act_counts']
        ax.scatter(x[y > -1], y[y > -1], color=rule_color(system), s=2)
        # ax.set_yscale('log')

        ax.set_xlabel('mean neur. exc (nS)')

        spearman = stats.spearmanr(x[y > 0], y[y > 0])

        # ax.set_title(f'{spearman[0]:.2f}', loc='right')

    axes_scatters[0].set_ylabel('replay count')

    axes_corrs = np.array([
        fig.add_subplot(bottom_gs[1,1:3]), fig.add_subplot(bottom_gs[1,3:5])
    ])

    for ax, l in zip(axes_corrs, ['F','G']):
        despine_ax(ax, 'tr')
        ax.set_title(l, fontweight='bold', x=0.1, y=0.95)


    # for system, c in zip(syslist, colors):
    #     x = data[system,str(npat),'tot_excit']
    #     y = data[system,str(npat),'tot_inhib']
    #     axes[1,2].scatter(x, y, s=1, c=c)

    correlations_nonzero = {}
    correlations_binary = {}
    correlations_excinh = {}

    for npat in npat_list:
        data = pd.read_csv(f'plotting/data/totweights/totweights_{npat}.csv', header=[0,1,2], index_col=0)
        for system in syslist:
            x = data[system,str(npat),'tot_excit']
            y = data[system,str(npat),'act_counts']
            z = data[system,str(npat),'tot_inhib']

            spearman_nonzero = stats.spearmanr(x[y > 0], y[y > 0])
            # spearman_nonzero = stats.spearmanr(x, y)
            pearson_binary = stats.pearsonr(x, np.clip(y, a_min=0, a_max=1))
            pearson_excinh = stats.pearsonr(x, z)

            correlations_binary[(system, npat)] = pearson_binary[0]
            correlations_nonzero[(system, npat)] = spearman_nonzero[0]
            correlations_excinh[(system, npat)] = pearson_excinh[0]


    ser_sper = pd.Series(correlations_nonzero)
    ser_pear = pd.Series(correlations_binary)
    ser_exin = pd.Series(correlations_excinh)

    for system in syslist:
        m = rule_marker(system)
        c = rule_color(system)
        axes_corrs[0].plot(ser_sper.loc[system], marker=m, color=c)
        axes_corrs[1].plot(ser_pear.loc[system], marker=m, color=c, label=rule_name(system))
        # print(stats.spearmanr(x[y > 0], y[y > 0]))
        # print(stats.pearsonr(x, np.clip(y, a_min=0, a_max=1)))

    axes_corrs[1].legend(loc=(0.6, 0.6))

    # letters = ['A','B','C','D','E','F']

    # for ax, l in zip(axes.flat, letters):
    #     ax.set_title(l, fontweight='bold', loc='left')


    # for ax in axes[1,:2]:
    #     ax.set_xlabel('embedded assemblies')

    axes_corrs[0].set_ylabel('spearman r (r.c.>0)')
    axes_corrs[1].set_ylabel('pearson r (bin. r.c.)')

    axes_corrs[0].set_xlabel('embedded assemblies')
    axes_corrs[1].set_xlabel('embedded assemblies')

    fig.tight_layout()
    plt.savefig(f'img/wta.png')


