import numpy as np
import h5py


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

# Function to create dataset with the correct shape
def create_weight_dataset(group, name, data, dtype=None):
    """Creates a dataset with the correct shape based on the input data."""
    group.require_dataset(
        name,
        shape=(data.shape[0],),  # Set shape dynamically based on data size
        maxshape=(None,),  # Allow expansion
        dtype=dtype if dtype else data.dtype,  # Use given dtype or infer from data
        compression="gzip"  # Enable compression
    )[:] = data  # Assign data after creation

def load_connectivity(filename):
    connectivity = {}
    weights = {}
    delays = {}

    connectivity['weights'] = weights
    connectivity['delays'] = delays

    with h5py.File(filename, "r") as h5f:
        print(list(h5f.keys()))

        for pre in ['E','I']:
            for post in ['E','I']:
                label = f'{post}{pre}'

                delays[label] = h5f[f'delays/{label}'][:]

                weights[label] = {
                    'weights' : h5f[f'weights/{label}/weights'][:],
                    'sources' : h5f[f'weights/{label}/sources'][:],
                    'targets' : h5f[f'weights/{label}/targets'][:],
                }

        connectivity['exc_alpha'] = h5f['exc_alpha'][:]
        connectivity['N_exc'] = h5f.attrs['N_exc']
        connectivity['N_inh'] = h5f.attrs['N_inh']

    return connectivity