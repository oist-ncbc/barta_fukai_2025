"""Script used for assembling connectivity matrix and calculating eigenvalues.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse

from utils import *  # data_path, load_connectivity, load_patterns, sparse, etc.


def get_W(system, npat, namespace, effective=False, exc_vals=True):
    """Assemble the connectivity matrix W with optional excitability scaling.

    Parameters
    ----------
    system : str
        System identifier (used in filenames).
    npat : int
        Number of patterns (used in filenames).
    effective : bool, default False
        If True, return a reduced "effective" E→E matrix (see below). If False,
        return the full 2×2 population block matrix:
        ``[[WEE, WEI], [WIE, WII]]``.
    exc_vals : bool, default True
        If True, scale presynaptic columns by activation‑derived excitabilities
        from the linear approximation CSV. If False, use uniform scaling (=1).

    Returns
    -------
    numpy.ndarray
        The assembled weight matrix `W` (either E‑only effective or full E+I block).

    Notes
    -----
    - The CSV at `{data_path()}/lognormal/linear_approx/{system}{npat}.csv`
      is indexed by population with rows like `('exc', ...)`, `('inh', ...)` and
      columns including `activation_exc` / `activation_inh`, used here as
      excitability multipliers for **presynaptic columns**.

    """
    if exc_vals:
        data = pd.read_csv(f'{data_path()}/lognormal/linear_approx/{system}{npat}.csv', index_col=[0,1])

    connectivity = load_connectivity(system, 'train', npat, namespace=namespace)

    # ---- E←E block ---------------------------------------------------------
    vals = connectivity['weights']['EE']['weights']
    sources = connectivity['weights']['EE']['sources']
    targets = connectivity['weights']['EE']['targets']

    coo = sparse.coo_array((vals, (targets, sources)))
    if exc_vals:
        excitability = data.loc['exc']['activation_exc'].values
    else:
        excitability = 1
    WEE = (coo.toarray().T * excitability).T  # scale presynaptic columns

    # ---- E←I block ---------------------------------------------------------
    vals = connectivity['weights']['EI']['weights']
    sources = connectivity['weights']['EI']['sources']
    targets = connectivity['weights']['EI']['targets']

    coo = sparse.coo_array((vals, (targets, sources)))
    if exc_vals:
        excitability = data.loc['exc']['activation_inh'].values
    else:
        excitability = 1
    WEI = (coo.toarray().T * excitability).T

    # ---- I←E block ---------------------------------------------------------
    vals = connectivity['weights']['IE']['weights']
    sources = connectivity['weights']['IE']['sources']
    targets = connectivity['weights']['IE']['targets']

    coo = sparse.coo_array((vals, (targets, sources)))
    if exc_vals:
        excitability = data.loc['inh']['activation_exc'].values
    else:
        excitability = 1
    WIE = (coo.toarray().T * excitability).T

    # ---- I←I block ---------------------------------------------------------
    vals = connectivity['weights']['II']['weights']
    sources = connectivity['weights']['II']['sources']
    targets = connectivity['weights']['II']['targets']

    coo = sparse.coo_array((vals, (targets, sources)))
    if exc_vals:
        excitability = data.loc['inh']['activation_inh'].values
    else:
        excitability = 1
    WII = (coo.toarray().T * excitability).T

    if effective:
        # Reduced effective E→E matrix
        W = WEE + WEI @ WIE
    else:
        # Full block: rows=[E; I], cols=[E, I]
        WEX = np.concatenate([WEE, WEI], axis=1)
        WIX = np.concatenate([WIE, WII], axis=1)
        W = np.concatenate([WEX, WIX], axis=0)

    return W


if __name__ == '__main__':
    # Parse command‑line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--system', type=str, required=True)
    parser.add_argument('--patterns', type=int, required=True)
    parser.add_argument('--namespace', type=str, default='lognormal')
    parser.add_argument('--effective', action='store_true')
    parser.add_argument('--exc', action='store_true')
    parser.add_argument('--vals_only', action='store_true')

    args = parser.parse_args()

    system = args.system
    npat = args.patterns

    # Build W as requested (effective or full)
    W = get_W(system, npat, namespace=args.namespace, effective=args.effective)

    # Compute eigensystem: excitatory submatrix or full matrix
    if args.exc:
        vals, vecs = np.linalg.eig(W[:8000,:8000])
    else:
        vals, vecs = np.linalg.eig(W)
    
    # Save either values alone or values + vectors (concatenated)

    create_directory(f'{data_path(args.namespace)}/eigensystem')

    if args.vals_only:
        np.savetxt(f'{data_path(args.namespace)}/eigensystem/{system}{npat}.csv', vals)
    else:
        np.savetxt(f'{data_path(args.namespace)}/eigensystem/{system}{npat}.csv', np.concatenate([[vals], vecs]))
