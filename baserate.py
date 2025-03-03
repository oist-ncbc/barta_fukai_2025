import pandas as pd
import argparse

from utils import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-o', '--ordered', action='store_true')
    parser.add_argument('-r', '--reverse', action='store_true')
    parser.add_argument('-p', '--patterns', type=int, nargs='+')

    args = parser.parse_args()

    for npat in args.patterns:
        data = pd.read_csv(f'{data_path()}/lognormal/linear_approx/hebb{npat}.csv', index_col=[0,1])
        rates = lognorm_randvar(3, 1, 8000)

        if args.reverse:
            sorted_rates = sortby(rates, data.loc['exc']['rate'].values[::-1])
            np.savetxt(f'config/rates/br_v1r_{npat}.csv', sorted_rates, fmt='%10.5f')
        if args.ordered:
            sorted_rates = sortby(rates, data.loc['exc']['rate'].values)
            np.savetxt(f'config/rates/br_v1o_{npat}.csv', sorted_rates, fmt='%10.5f')

        