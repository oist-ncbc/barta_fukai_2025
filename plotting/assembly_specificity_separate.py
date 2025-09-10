import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import pickle

from plotutils import *


if __name__ == '__main__':
    npat = 1000
    syslist = ['rate','hebb_smooth_rate','hebb']

    with open(f'data/conductance_traces{npat}.pkl', 'rb') as f:
        conductance_traces = pickle.load(f)

    differences = {}

    for npat in [800, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400, 2600, 2800, 3000]:
        entropies_ass = pd.read_csv(f'plotting/data/inhibitory_specialization/entropies_assemblies_{npat}.csv', index_col=0)
        entropies_rand_ass = pd.read_csv(f'plotting/data/inhibitory_specialization/entropies_rand_assemblies_{npat}.csv', index_col=0)
        
        for system in ['rate','hebb_smooth_rate','hebb']:
            differences[(system, npat, 'mean')] = np.exp(entropies_ass[system]).mean()
            diff = np.exp(entropies_ass[system]).mean() - np.exp(entropies_rand_ass[system]).mean()
            differences[(system, npat, 'diff')] = diff / np.exp(entropies_rand_ass[system]).mean() * 100
            test_res = stats.ttest_rel(np.exp(entropies_ass[system]), np.exp(entropies_rand_ass[system]))
            differences[(system, npat, 'T')] = test_res[0]
            differences[(system, npat, 'p')] = test_res[1]

    df = pd.Series(differences).unstack()
    df['p'] = df['p'].apply(format_p)
    # print(df[['mean','diff','T','p']].columns)
    print(df[['mean','diff','T','p']].to_latex(float_format="%.2f", escape=False))

    fig = plt.figure(figsize=(8, 3.5), constrained_layout=True)

    spacing = 0.05
    rest = 1-spacing

    gs1 = fig.add_gridspec(nrows=2, ncols=6,
                            height_ratios=[2, 1],  # ↑ make top row taller
                            hspace=0.1,
                            wspace=0.2,
                            width_ratios=[spacing, rest]*3)
    
    img_ax1 = fig.add_subplot(gs1[0,0:2])
    img_ax2 = fig.add_subplot(gs1[0,2:4])
    img_ax3 = fig.add_subplot(gs1[0,4:6])
    img_axes = [img_ax1, img_ax2, img_ax3]

    hist_ax1 = fig.add_subplot(gs1[1,1])
    hist_ax2 = fig.add_subplot(gs1[1,3])
    hist_ax3 = fig.add_subplot(gs1[1,5])

    names = ['neuron_spec','assembly_spec','feedback']

    for name, ax in zip(names, img_axes):
        img = plt.imread(f'plotting/data/{name}.png')
        ax.imshow(img)
        despine_ax(ax)

    entropies = pd.read_csv('plotting/data/inhibitory_specialization/entropies_1000.csv', index_col=0)
    entropies_ass = pd.read_csv('plotting/data/inhibitory_specialization/entropies_assemblies_1000.csv', index_col=0)
    feedback = pd.read_csv('plotting/data/inhibitory_specialization/self_feedback_1000.csv', index_col=0)
    entropies_rand_ass = pd.read_csv('plotting/data/inhibitory_specialization/entropies_rand_assemblies_1000.csv', index_col=0)

    axes = [hist_ax1, hist_ax2, hist_ax3]

    for system in syslist:
        # import pdb; pdb.set_trace()
        axes[0].hist(np.exp(entropies[system]), color=rule_color(system), **hist_params, bins=np.linspace(50, 280, 50), label=rule_name(system))
        axes[1].hist(np.exp(entropies_ass[system]), color=rule_color(system), **hist_params, bins=np.linspace(800, 970, 50))
        axes[2].hist(feedback[system], color=rule_color(system), **hist_params, bins=np.linspace(-0.02, 0.12, 50))

    axes[0].legend(loc=(0.8,0.5))

    axes[0].set_xlabel(r'diversity of I$\rightarrow$E'+'\nconnections')
    axes[0].set_ylabel('density\n(I neurons)')
    axes[1].set_xlabel(r'diversity of I$\rightarrow$assembly'+'\nconnections')
    axes[1].set_ylabel('density\n(I neurons)')
    axes[2].set_xlabel('self-feedback\ncoefficient')
    axes[2].set_ylabel('density\n(assemblies)')

    letters = ['A','B','C']
    for ax, l in zip(img_axes, letters):
        ax.set_title(l, y=0.88, x=0.05, fontweight='bold')

    plt.savefig('plotting/img/feedback.svg', dpi=300, bbox_inches='tight')

    fig, axes = plt.subplots(ncols=3, figsize=(8,2.2))

    for system in syslist:
        c = rule_color(system)
        mean_exc = conductance_traces['traces_exc'][system].mean(axis=0)
        print(len(conductance_traces['traces_exc'][system]))
        trace_len = len(mean_exc)

        xx = np.linspace(-1, 1, trace_len)

        axes[0].plot(xx, mean_exc, color=c, label=rule_name(system))

        inh = conductance_traces['traces_inh'][system].mean(axis=0)
        inh_rest = conductance_traces['traces_inh_rest'][system].mean(axis=0)

        axes[1].plot(xx, inh, color=c)
        axes[1].plot(xx, inh_rest, color=c, linestyle='dashed')

        inh_norm = inh / inh[:10].mean()
        inh_rest_norm = inh_rest / inh_rest[:10].mean()

        axes[2].plot(xx, inh / inh_rest, color=c)

    # for ax in axes:
    #     ax.set_xlabel('time (s)')

    axes[0].set_ylabel('excitatory conductance (nS)')
    axes[1].set_ylabel('inhibitory conductance (nS)')
    axes[2].set_ylabel('active / non-active\ninhibition ratio')

    for ax in axes:
        despine_ax(ax, 'b')
        ax.axvline(0, color='black', linestyle='dashed', lw=0.5)

    axes[0].plot([-0.8, -0.3], [12, 12], color='black', lw=3)
    axes[0].text(-0.55, 12.5, '500ms', ha='center')

    axes[0].legend(loc=(0.6,0.6))

    for ax, l in zip(axes.flat, ['A','B','C']):
        ax.set_title(l, loc='left', fontweight='bold', x=0.05, y=0.88)


    fig.tight_layout()
    plt.savefig('plotting/img/assembly_specificity.png')


    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(8,2.5), width_ratios=[1.2,1,1])
    axD = axes[0]
    axhist = axes[1]
    axplot = axes[2]

    img = plt.imread(f'plotting/data/assembly_feedback_horizontal.png')
    axD.imshow(img)

    despine_ax(axD)

    bins = np.linspace(6.72, 6.85, 40)
    bins=50

    for system in ['hebb']:
        axhist.hist(np.exp(entropies_ass[system]), color=rule_color(system), **hist_params, bins=bins, label='embedded')
        axhist.hist(np.exp(entropies_rand_ass[system]), color=rule_color(system), **hist_params, bins=bins, linestyle='dotted', label='random')

    axhist.set_xlabel(r'diversity of I$\rightarrow$assembly connections')
    axhist.set_ylabel('density\n(assemblies)')

        # Get current position (in figure coordinates)

    for system in ['rate','hebb_smooth_rate','hebb']:
        axplot.plot(df['diff'].loc[system], marker=rule_marker(system), color=rule_color(system), label=rule_name(system))

    axplot.set_ylabel('relative difference (%)')
    axplot.set_xlabel('embedded assemblies')
    # axplot.set_xticks([800,1000,1200,1400,1600,1800,2000,2200,2400,2600,2800,3000])
    # axplot.set_xticklabels(['',1000,'',1400,'',1800,'',2200,'',2600,'',3000])
    axplot.legend()

    axD.set_title('A', fontweight='bold', x=0.05, y=1.05)
    axhist.set_title('B', fontweight='bold', x=-0.1, y=1)
    axplot.set_title('C', fontweight='bold', x=-0.1, y=1)

    fig.tight_layout()
    pos = axD.get_position()

    # Expand it (e.g. 20% wider and taller, centered)
    new_pos = [pos.x0, pos.y0 - 0.08, pos.width * 1.05, pos.height * 1.1]

    axD.set_position(new_pos)

    axhist.legend(loc=(0.6,0.9))

    plt.savefig('plotting/img/assembly_specificity_supplementary.png', dpi=300)