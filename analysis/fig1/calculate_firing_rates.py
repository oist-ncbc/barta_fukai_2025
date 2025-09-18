import argparse
import numpy as np
import time

from utils import *
from analysis import *

import logging
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    start_time = time.time()
    
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('--system', type=str)
    parser.add_argument('--namespace', type=str, default='lognormal')
    parser.add_argument('--npat', type=int)
    args = parser.parse_args()

    logger.info(f"Arguments received: system={args.system}, npat={args.npat}")

    fr = get_firing_rates(args.system, args.npat, namespace=args.namespace)
    np.savetxt(f'plotting/data/firing_rates/{args.system}{args.npat}.csv', np.concatenate([fr['exc'], fr['inh']]))
    logger.info("Firing rates saved.")

    duration = time.time() - start_time
    logger.info(f"Execution completed in {duration:.2f} seconds.")