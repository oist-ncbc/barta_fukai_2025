import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from plotutils import *


if __name__ == '__main__':
    fig, axes = plt.subplots(ncols=3, nrows=3, figsize=(8,6), sharex=True)

    errs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    bins = np.linspace(0, 1, 50)

    for err, ax in zip(errs, axes.flat):
        df = pd.read_csv(f'plotting/data/overlaps/completion_{err}_1000.csv', index_col=0)
        systems = list(df.columns)[:-1]

        for system in systems:
            ax.hist(df[system], color=rule_color(system), bins=bins, label=rule_name(system), **hist_params)
        
        ax.axvline(1-err, color='black', linestyle='dashed')
        ax.set_title(f'error: {err}', loc='left')

    axes[2,2].legend()
    for ax in axes[2]:
        ax.set_xlabel('overlap')

    for ax in axes[:,0]:
        ax.set_ylabel('density')

    fig.tight_layout()

    plt.savefig('img/linear_completion.png')