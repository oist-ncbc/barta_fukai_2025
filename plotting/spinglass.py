import pickle
import matplotlib.pyplot as plt

from plotutils import *


if __name__ == '__main__':
    with open('plotting/data/overlap_activity.pkl', 'rb') as f:
        overlaps = pickle.load(f)

    axes = plt.subplots(ncols=3, nrows=5)

    dur = 200
    start = 490
    end = start+dur

    xx = np.arange(8000)

    for npat, system in product([1000, 1400, 2000], ['rate','hebb']):
        
        cutoff = overlap_res['ev_cutoff']
        mask = overlap_res['eigenvector'] > cutoff
        axes[0].scatter(xx[mask], overlap_res['eigenvector'][mask], color='black', s=1)
        axes[0].scatter(xx[~mask], overlap_res['eigenvector'][~mask], color='red', s=1)
        ylim = axes[0].get_ylim()

        if cutoff > 0:
            axes[0].axhspan(cutoff, ylim[1], color='black', alpha=0.1)
        else:
            axes[0].axhspan(ylim[0], cutoff, color='black', alpha=0.1)

        axes[1].plot(overlap_res['activation'][start:end], color='red')

        for activity in overlap_res['overlaps_activity']:
            axes[1].plot(activity[start:end], color='black', lw=0.2)

        plot_table(axes[2], [overlap_res['neurons'], *overlap_res['overlapping_patterns']], colors=['red']+['black']*5)

        axes[0].set_ylabel('real')
        axes[0].set_xlabel('neuron index')

        axes[1].set_xlabel('time (s)')
        axes[1].set_ylabel('assembly activation')