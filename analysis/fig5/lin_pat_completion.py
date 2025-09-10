import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import load_patterns
from eigenvalues import get_W


if __name__ == '__main__':
    system = 'hebb'

    overlaps = {}
    pearsons = {}

    for err in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        mask = (np.random.rand(8000) < (1-err)).astype(float)

        for npat in tqdm([1000, 1200, 1400, 1600, 1800, 2000]):
        # for npat in tqdm([3000]):
            patterns = load_patterns(npat)
            dense_patterns = patterns.dense()

            dense_corrupted = dense_patterns * mask

            overlaps[npat] = {}
            pearsons[npat] = {}

            for system in ['hebb', 'hebb_smooth_rate','rate','uniform']:
                if system == 'uniform':
                    W = get_W('hebb', npat, exc_vals=False)
                else:
                    W = get_W(system, npat, exc_vals=True)

                pattern_derivs = W[:8000,:8000] @ dense_corrupted.T
                # pattern_derivs = W @ padded_patterns.T
                # pattern_derivs += W @ pattern_derivs

                pattern_derivs = pattern_derivs[:8000]

                npsysoverlaps = np.zeros(npat)

                for pix in range(npat):
                    pat = patterns[pix]
                    order = np.argsort(pattern_derivs[:,pix])[::-1]
                    patsize = len(pat)
                    npsysoverlaps[pix] = np.isin(order[:patsize], pat).mean()

                overlaps[npat][system] = npsysoverlaps
        
            pd.DataFrame(overlaps[npat]).to_csv(f'plotting/data/overlaps/completion_{err:.1f}_{npat}.csv')