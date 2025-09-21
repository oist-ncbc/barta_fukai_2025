import numpy as np
import matplotlib.pyplot as plt

from utils import load_linear, load_conduct
from plotutils import *


if __name__ == '__main__':
    fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(8,7))

    gL = 10

    npat = 1000
    for system in ['hebb_smooth_rate','hebb','rate']:
        data_lin = load_linear(system, npat).loc['exc']
        data_con = load_conduct(system, npat).loc['exc']

        tot_cond = data_con['mean_e'] + data_con['mean_i']
        vR = (data_con['mean_i'] + gL) * (-80) / (tot_cond + 10)

        # axes[1,0].scatter(tot_cond, data_lin['activation_exc'], s=1)
        axes[1,1].scatter(data_con['mean_e'], vR, s=1, color=rule_color(system))
        axes[1,0].scatter(vR, data_lin['activation_exc'], s=1, color=rule_color(system))

    axes[1,1].set_ylabel(r'$\bar{V}$ (mV)')
    axes[1,0].set_ylabel(r'$\chi^{EE}$ (Hz/nS)')
    axes[1,1].set_xlabel(r'$\bar{g}_\mathrm{exc}$ (nS)')
    axes[1,0].set_xlabel(r'$\bar{V}$ (mV)')

    axes[1,0].text(s=r'$\bar{V}=\frac{g_LE_L + \bar{g}_\mathrm{inh}E_\mathrm{inh} + \bar{g}_\mathrm{exc}E_\mathrm{exc}}{g_L + \bar{g}_\mathrm{inh} + \bar{g}_\mathrm{exc}}$',
                   x=-68.5, y=0.8, fontsize=12)

    npat_list = [1000, 1200, 1400, 1600, 1800, 2000]

    for system in ['hebb_smooth_rate','hebb','rate']:
        mean_list = []
        cv_list = []
        for npat in npat_list:
            data_lin = load_linear(system, npat).loc['exc']
            data_con = load_conduct(system, npat).loc['exc']

            std = data_lin['activation_exc'].std()
            mean = data_lin['activation_exc'].mean()

            mean_list.append(mean)
            cv_list.append(std / mean)

        axes[0,0].plot(npat_list, mean_list, marker='o', color=rule_color(system))
        axes[0,1].plot(npat_list, cv_list, marker='o', color=rule_color(system))

    axes[0,0].set_xlabel('embedded assemblies')
    axes[0,0].set_ylabel('mean exc. susceptibility (Hz/nS)')

    axes[0,1].set_xlabel('embedded assemblies')
    axes[0,1].set_ylabel('susceptibility CV')

    for ax, l in zip(axes.flat, ['A','B','C','D']):
        ax.set_title(l, fontweight='bold', x=-0.15, y=0.95)

    fig.tight_layout()
    plt.savefig('plotting/img/supp_susc.png')

