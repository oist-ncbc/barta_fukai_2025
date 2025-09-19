"""General utility functions and classes for data loading, random variables, and pattern manipulation."""

import numpy as np
import pandas as pd
import h5py
from scipy.stats import chi2
from scipy import sparse
import matplotlib.pyplot as plt
import yaml
import psutil
import os

def memory_usage() -> str:
    """Return a formatted string summarising current memory usage of the process."""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return f"[MEMORY] RSS: {mem_info.rss / 1e6:.2f} MB, VMS: {mem_info.vms / 1e6:.2f} MB"

def is_numpy_float(x) -> bool:
    """Determine if a value is a NumPy floating point scalar."""
    return np.issubdtype(type(x), np.floating)

def data_path(namespace=None) -> str:
    """Return the configured base directory for data files."""
    with open('config/server_config.yaml') as f:
        server_config = yaml.safe_load(f)

    if namespace is None:
        return server_config['data_path']
    else:
        return f"{server_config['data_path']}/{namespace}"
    
def create_directory(folder_path):
    try:
        os.mkdir(folder_path)
        print(f"Created directory '{folder_path}'.")
    except FileExistsError:
        pass

def underscore(text: str) -> str:
    """Prefix a string with an underscore if it is non-empty."""
    return '_' + text if text else text

def lognorm_randvar(mean: float, sigma: float, size) -> np.ndarray:
    """Generate lognormal random variables given desired mean and standard deviation."""
    E = mean
    Var = sigma**2
    sig = np.sqrt(np.log(Var / (E * E) + 1))
    mu = np.log(E) - sig**2 / 2
    return np.exp(np.random.randn(size) * sig + mu)

def sortby(x, key):
    """Return an array sorted according to the order implied by a key."""
    return np.sort(x)[np.argsort(np.argsort(key))]

def create_weight_dataset(group: h5py.Group, name: str, data: np.ndarray, dtype=None):
    """Create a dataset with the correct shape based on the input data."""
    group.require_dataset(
        name,
        shape=(data.shape[0],),
        maxshape=(None,),
        dtype=dtype if dtype is not None else data.dtype,
        compression="gzip",
    )[:] = data

def load_connectivity(system: str, run: str, npat: int, namespace: str) -> dict:
    """
    Load connectivity data from an HDF5 file.

    Parameters
    ----------
    system : str
        System identifier.
    run : str
        Run identifier used in filenames.
    npat : int
        Number of patterns.
    namespace : str, optional
        Data folder name.

    Returns
    -------
    dict
        Dictionary containing weight arrays, delay arrays, excitatory adaptation parameters,
        and metadata such as the number of excitatory and inhibitory neurons.
    """
    path_to_folder = data_path(namespace)
    if run != '':
        filename = f"{path_to_folder}/{system}_{run}{npat}.h5"
    else:
        filename = f"{path_to_folder}/{system}{npat}.h5"

    connectivity = {'weights': {}, 'delays': {}}

    with h5py.File(filename, "r") as h5f:
        grp = h5f.require_group('connectivity')
        for pre in ['E', 'I']:
            for post in ['E', 'I']:
                label = f'{post}{pre}'
                connectivity['delays'][label] = grp[f'delays/{label}'][:]
                connectivity['weights'][label] = {
                    'weights': grp[f'weights/{label}/weights'][:],
                    'sources': grp[f'weights/{label}/sources'][:],
                    'targets': grp[f'weights/{label}/targets'][:],
                }
        connectivity['exc_alpha'] = grp['exc_alpha'][:]
        connectivity['N_exc'] = grp.attrs['N_exc']
        connectivity['N_inh'] = grp.attrs['N_inh']

    return connectivity

def plot_covariance_ellipse(mean, cov, ax=None, confidence: float = 0.95, **line_kwargs):
    """Plot the confidence ellipse of a 2D normally distributed dataset."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
    theta = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])
    chi2_val = chi2.ppf(confidence, df=2)
    width, height = 2 * np.sqrt(chi2_val * eigenvalues)
    t = np.linspace(0, 2 * np.pi, 300)
    x = width / 2 * np.cos(t)
    y = height / 2 * np.sin(t)
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    xy_rotated = R @ np.array([x, y])
    xy_rotated[0] += mean[0]
    xy_rotated[1] += mean[1]
    ax.plot(xy_rotated[0], xy_rotated[1], **line_kwargs)
    return ax

class Patterns:
    """
    Sparse pattern container storing concatenated indices and split pointers.

    Attributes
    ----------
    indices : array_like
        Concatenated indices for all patterns.
    splits : array_like
        Split points indicating pattern boundaries.
    pointers : numpy.ndarray
        Calculated start and end pointers for each pattern.
    n : int
        Number of patterns.
    neurons : int
        Total number of neurons (default 8000).
    """
    def __init__(self, indices, splits, neurons: int = 8000):
        self.indices = indices
        self.splits = splits
        self.pointers = np.concatenate([[0], self.splits, [len(self.indices)]])
        self.n = len(splits) + 1
        self.neurons = neurons

    def __getitem__(self, ix):
        if isinstance(ix, slice):
            start, stop, step = ix.indices(self.n)
            return [
                self.indices[self.pointers[i] : self.pointers[i + 1]]
                for i in range(start, stop, step)
            ]
        return self.indices[self.pointers[ix] : self.pointers[ix + 1]]

    def __len__(self) -> int:
        return self.n

    def sizes(self) -> np.ndarray:
        """Return the sizes of each pattern."""
        return np.diff(self.pointers)

    def participation(self) -> np.ndarray:
        """Return per-neuron counts across patterns."""
        ser = pd.Series(self.indices).value_counts().reindex(np.arange(self.neurons), fill_value=0)
        return ser.values

    def csr(self) -> sparse.csr_array:
        """Return a sparse CSR representation of the patterns."""
        data = np.ones(len(self.indices))
        return sparse.csr_array((data, self.indices, self.pointers), shape=(self.n, self.neurons))

    def dense(self) -> np.ndarray:
        """Return a dense array representation of the patterns."""
        return self.csr().toarray()

    def randomize(self) -> "Patterns":
        """Return a copy of patterns with indices randomly permuted across neurons."""
        new_indices = []
        for s in self.sizes():
            new_indices.extend(np.random.permutation(self.neurons)[: s])
        return Patterns(np.array(new_indices), self.splits, self.neurons)

    def overlap(self, a: int, b: int) -> int:
        """Return the number of common indices between patterns a and b."""
        return np.isin(self[a], self[b]).sum()

def load_patterns(npat: int, namespace: str) -> Patterns:
    """Load pattern indices and splits from an HDF5 file and return a Patterns object."""
    path_to_folder = data_path(namespace)
    filename = f"{path_to_folder}/init{npat}.h5"
    with h5py.File(filename, "r") as h5f:
        patterns = Patterns(
            h5f['connectivity/patterns/indices'][:],
            h5f['connectivity/patterns/splits'][:],
            neurons=h5f['connectivity'].attrs['N_exc']
        )
    return patterns

def load_linear(system: str, npat: int, namespace: str) -> pd.DataFrame:
    """Load linear approximation data for a given system and number of patterns."""
    path_to_folder = f"{data_path(namespace)}/linear_approx"
    filename = f"{path_to_folder}/{system}{npat}.csv"
    return pd.read_csv(filename, index_col=[0, 1])

def load_activation(system: str, npat: int, run: str = 'spontaneous', namespace: str = 'lognormal'):
    """Load activation times, durations, and pattern indices from an HDF5 file."""
    path_to_folder = data_path(namespace)
    filename = f"{path_to_folder}/{system}_{run}{npat}_activations.h5"
    with h5py.File(filename, "r", swmr=True) as h5f:
        act_times, durations, pattern_ixs = h5f['activations'][:]
    return act_times, durations, pattern_ixs

def create_stim_tuples(patterns, num_indices: int, nstim: int, duration: float = 0.1, spacing: float = 1.0, random: bool = False):
    """Create a list of stimulus tuples (start, end, indices) for pattern-driven simulations."""
    t = 1.0
    tuples = []
    for i in range(nstim):
        pat = patterns[i % len(patterns)]
        if num_indices is None:
            num_indices_current = len(pat)
        else:
            num_indices_current = num_indices
        if random:
            ind_ix = np.random.permutation(len(pat))[: num_indices_current]
        else:
            ind_ix = np.arange(num_indices_current, dtype=int)
        tuples.append((t, t + duration, pat[ind_ix]))
        t += spacing
    return tuples

def load_stim_file(filename: str, patterns, fraction: float):
    """Load stimulus specifications from a CSV file and construct stimulus tuples."""
    stims = pd.read_csv(filename, header=None, index_col=False).values
    tuples = []
    for x in stims:
        pt = patterns[int(x[2])]
        num_indices = int(len(pt) * fraction)
        ind_ix = np.random.permutation(len(pt))[:num_indices]
        if x[3] == False:
            tuples.append((x[0], x[1], pt[ind_ix]))
        else:
            rand_pt = np.random.permutation(8000)[: len(pt)]
            tuples.append((x[0], x[1], rand_pt[ind_ix]))
    return tuples

def load_conduct(system: str, npat: int):
    """Load conductance statistics from the var_stats directory."""
    return pd.read_csv(f'{data_path()}/lognormal/var_stats/{system}_conductances{npat}_stats.csv', index_col=[0, 1])
