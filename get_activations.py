import h5py
import argparse
from multiprocessing import Pool, get_logger
import logging

from utils import *
from analysis import get_spike_counts


# def get_activations(offset):
#     _, sc = get_spike_counts(*spikes, max_t, dt=0.1, offset=offset*0.01)

#     activations = []
#     for i in range(npat):
#         activations.append((sc[patterns[i]] > 0).mean(axis=0))

#     return np.array(activations)

import gc

def get_activations(offset):
    logger = logging.getLogger()
    logger.info(f"Processing offset {offset}, memory: {memory_usage()}")

    # patterns = load_patterns('hebb', npat)

    try:
        _, sc = get_spike_counts(*spikes, max_t, dt=0.1, offset=offset * 0.1)

        activations = []
        for i in range(npat):
            activations.append((sc[patterns[i]] > 0).mean(axis=0))

            if i % 100 == 0:
                logger.info(f"Offset {offset}: Processed {i} patterns")

        result = np.array(activations)

        # Free memory
        del activations
        gc.collect()

        logger.info(f"Completed offset {offset}, memory: {memory_usage()}")
        return result

    except Exception as e:
        logger.error(f"Error processing offset {offset}: {e}", exc_info=True)
        return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-p', '--patterns', type=int)
    parser.add_argument('-r', '--run', type=str)
    parser.add_argument('-s', '--system', type=str)

    stabilization_offset = 10

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s")

    folder = f"{data_path()}/lognormal"
    filename = f"{folder}/{args.system}_{args.run}{args.patterns}.h5"

    logging.info(
        f"Loading data. {memory_usage()}"
    )

    patterns = load_patterns('hebb', args.patterns)

    with h5py.File(filename, "r", swmr=True) as h5f:
        spikes = h5f['spikes_exc'][:].T
        max_t = h5f.attrs['simulation_time']

    spikes = spikes[:,spikes[1] > stabilization_offset]
    spikes[1] = spikes[1] - stabilization_offset
    max_t -= stabilization_offset

    logging.info(
        f"Data loaded. {memory_usage()}"
    )

    npat = args.patterns

    activations = np.array(Pool(processes=5).map(get_activations, np.arange(10)[::-1]))
    activations = np.transpose(activations, (1,2,0)).reshape((args.patterns,-1))

    logging.info(
        f"Activations calculated. {memory_usage()}"
    )

    act_times = []
    durations = []
    pattern_ixs = []

    for i in range(args.patterns):
        change = np.diff((activations[i] > 0.9).astype(float))
        starts = np.argwhere(change == 1).flatten()
        ends = np.argwhere(change == -1).flatten()

        if ends.size > 0:
            starts = starts[starts < ends[-1]]
            if starts.size > 0:
                ends = ends[ends > starts[0]]
        
        if starts.size > 0:
            act_times.extend(starts)
            pattern_ixs.extend([i]*starts.size)
            durations.extend(ends - starts)

    try:
        activations_sparse = np.array([
            act_times,
            durations,
            pattern_ixs
        ])
    except:
        import pdb; pdb.set_trace()

    output_file = f"{folder}/{args.system}_{args.run}{args.patterns}_activations.h5"

    logging.info(
        f"Saving results to {output_file}. {memory_usage()}"
    )

    with h5py.File(output_file, "w") as h5f:
        h5f.create_dataset(
            'traces',
            shape=activations.shape,
            compression='gzip'
        )[:] = activations
        h5f.create_dataset(
            'activations',
            shape=activations_sparse.shape,
            compression='gzip'
        )[:] = activations_sparse