import numpy as np
import matplotlib.pyplot as plt

from utils import load_linear, load_conduct, load_patterns
from plotutils import *


if __name__ == '__main__':
    fig, ax = plt.subplots(figsize=(3,2))

    gL = 10

    npat = 1000

    patterns = load_patterns(npat)

    for i, system in enumerate(['hebb_smooth_rate','hebb','rate']):
        data_lin = load_linear(system, npat).loc['exc']
        data_con = load_conduct(system, npat).loc['exc']

        ax.scatter(patterns.participation()+i*0.2, data_lin['activation_exc'], color=rule_color(system), s=2)

    ax.set_xlabel('number of assemblies\nneuron participates in')
    ax.set_ylabel('susceptibility to\nsynaptic input (Hz/nS)')

    # npat_list = [1000, 1200, 1400, 1600, 1800, 2000]

    # for system in ['hebb_smooth_rate','hebb','rate']:
    #     mean_list = []
    #     cv_list = []
    #     for npat in npat_list:
    #         data_lin = load_linear(system, npat).loc['exc']
    #         data_con = load_conduct(system, npat).loc['exc']

    #         std = data_lin['activation_exc'].std()
    #         mean = data_lin['activation_exc'].mean()

    #         mean_list.append(mean)
    #         cv_list.append(std / mean)

    #     axes[0,0].plot(npat_list, mean_list, marker='o', color=rule_color(system))
    #     axes[0,1].plot(npat_list, cv_list, marker='o', color=rule_color(system))

    # axes[0,0].set_xlabel('embedded assemblies')
    # axes[0,0].set_ylabel('mean exc. susceptibility (Hz/nS)')

    # axes[0,1].set_xlabel('embedded assemblies')
    # axes[0,1].set_ylabel('susceptibility CV')

    # for ax, l in zip(axes.flat, ['A','B','C','D']):
    #     ax.set_title(l, fontweight='bold', x=-0.15, y=0.95)

    fig.tight_layout()
    plt.savefig('img/participation_dependence.png')

