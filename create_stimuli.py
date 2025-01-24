import argparse

import numpy as np
import pandas as pd


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--nstim', type=int, default=1000)
    parser.add_argument('--duration', type=float, default=0.1)
    parser.add_argument('--spacing', type=int, default=1)

    args = parser.parse_args()

    stimuli = []

    for i in range(args.nstim):
        stimuli.append((args.spacing*i+1, args.spacing*i+1 + args.duration, i, False))

    pd.DataFrame(stimuli).to_csv('config/stimuli_all_01.csv', index=False, header=False)