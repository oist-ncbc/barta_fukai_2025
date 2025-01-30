import numpy as np


def despine_ax(ax, where=None, remove_ticks=None):
    if where is None:
        where = 'trlb'
    if remove_ticks is None:
        remove_ticks = where

    if remove_ticks is not None:
        if 'b' in where:
            ax.set_xticks([])
            ax.set_xticklabels([])
        if 'l' in where:
            ax.set_yticks([])
            ax.set_yticklabels([])

    to_despine = []

    if 'r' in where:
        to_despine.append('right')
    if 't' in where:
        to_despine.append('top')
    if 'l' in where:
        to_despine.append('left')
    if 'b' in where:
        to_despine.append('bottom')

    for side in to_despine:
        ax.spines[side].set_visible(False)

def underscore(text):
    if len(text) > 0:
        return '_' + text
    else:
        return text
    
def lognorm_randvar(mean, sigma, size):
    E = mean
    Var = sigma ** 2

    sig = np.sqrt(np.log(Var/(E*E) + 1))
    mu = np.log(E) - sig**2 / 2

    return np.exp(np.random.randn(size)*sig + mu)

def sortby(x, key):
    return np.sort(x)[np.argsort(np.argsort(key))]