import argparse
import numpy as np
import time

from utils import *
from analysis import *
from eigenvalues import get_W

import logging
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    start_time = time.time()
    
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('--system', type=str)
    parser.add_argument('--npat', type=int)
    args = parser.parse_args()


    logger.info(f"Arguments received: system={args.system}, npat={args.npat}")

    W = get_W(system=args.system, npat=args.npat, exc_vals=False)

    Wei = W[:8000,8000:]
    wei = Wei[Wei != 0]
    totinhib = Wei.sum(axis=1)

    Wee = W[:8000,:8000]
    totexcit = Wee.sum(axis=1)

    np.savetxt(f'plotting/data/ei_weights/{args.system}{args.npat}.csv', wei)
    np.savetxt(f'plotting/data/tot_inhib/{args.system}{args.npat}.csv', totinhib)
    np.savetxt(f'plotting/data/tot_excit/{args.system}{args.npat}.csv', totinhib)
    logger.info("Weights saved.")