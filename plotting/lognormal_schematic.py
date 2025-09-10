from plotutils import *
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import lognorm

mpl.rcParams['axes.linewidth'] = 2


if __name__ == '__main__':
    fig, ax = plt.subplots(figsize=(1,1))

    xx = np.linspace(0,5,1000)
    ax.plot(xx, lognorm.pdf(xx, s=1), color='black', lw=2)
    ax.set_xticks([])
    ax.set_yticks([])

    plt.savefig('plotting/img/lognorm.png', dpi=300)