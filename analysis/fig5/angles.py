import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import load_patterns
from eigenvalues import get_W


if __name__ == '__main__':
    system = 'hebb'

    angle_res = {}
    pearsons = {}

    for npat in tqdm([1000,]):
    # for npat in tqdm([3000]):
        patterns = load_patterns(npat)
        dense_patterns = patterns.dense()

        angle_res[npat] = {}
        pearsons[npat] = {}

        for system in ['hebb', 'hebb_smooth_rate','rate','uniform']:
            if system == 'uniform':
                W = get_W('hebb', npat, exc_vals=False)
            else:
                W = get_W(system, npat, exc_vals=True)

            pattern_derivs = W[:8000,:8000] @ patterns.dense().T
            # pattern_derivs = W @ padded_patterns.T
            # pattern_derivs += W @ pattern_derivs

            pattern_derivs = pattern_derivs[:8000]

            norm_patterns = patterns.dense().T / np.linalg.norm(patterns.dense().T, axis=0)
            norm_derivs   = pattern_derivs / np.linalg.norm(pattern_derivs, axis=0)

            angles = np.arccos(np.sum(norm_patterns * norm_derivs, axis=0)) / np.pi * 180

        #     npsysoverlaps = np.zeros(npat)

        #     for pix in range(npat):
        #         pat = patterns[pix]
        #         order = np.argsort(pattern_derivs[:,pix])[::-1]
        #         patsize = len(pat)
        #         npsysoverlaps[pix] = np.isin(order[:patsize], pat).mean()

            angle_res[npat][system] = angles
    
        pd.DataFrame(angle_res[npat]).to_csv(f'plotting/data/overlaps/angles{npat}.csv')