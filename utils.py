import numpy as np
import h5py
from scipy.stats import chi2
import matplotlib.pyplot as plt
import yaml


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

def data_path():
    with open('config/server_config.yaml') as f:
        server_config = yaml.safe_load(f)

    return server_config['data_path']

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
        grp = h5f.require_group('connectivity')
        for pre in ['E','I']:
            for post in ['E','I']:
                label = f'{post}{pre}'

                delays[label] = grp[f'delays/{label}'][:]

                weights[label] = {
                    'weights' : grp[f'weights/{label}/weights'][:],
                    'sources' : grp[f'weights/{label}/sources'][:],
                    'targets' : grp[f'weights/{label}/targets'][:],
                }

        connectivity['exc_alpha'] = grp['exc_alpha'][:]
        connectivity['N_exc'] = grp.attrs['N_exc']
        connectivity['N_inh'] = grp.attrs['N_inh']

    return connectivity



def plot_covariance_ellipse(mean, cov, ax=None, confidence=0.95, **line_kwargs):
    """
    Plots the confidence ellipse of a 2D normally distributed dataset.
    
    Parameters:
    mean (array-like): Mean of the distribution (2D vector).
    cov (array-like): 2x2 Covariance matrix.
    ax (matplotlib.axes.Axes, optional): Axes object to draw the ellipse on. If None, a new figure is created.
    confidence (float): Confidence level (default: 0.95).
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6,6))
    
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
    
    # Compute the angle and axes lengths
    theta = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])
    chi2_val = chi2.ppf(confidence, df=2)
    width, height = 2 * np.sqrt(chi2_val * eigenvalues)
    
    # Parametric equation for an ellipse
    t = np.linspace(0, 2 * np.pi, 300)
    x = width / 2 * np.cos(t)
    y = height / 2 * np.sin(t)
    
    # Rotation matrix
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    
    # Apply rotation and shift
    xy_rotated = R @ np.array([x, y])
    xy_rotated[0] += mean[0]
    xy_rotated[1] += mean[1]
    
    # Plot
    ax.plot(xy_rotated[0], xy_rotated[1], 'b', label=f'{confidence*100:.1f}% Confidence Ellipse', **line_kwargs)