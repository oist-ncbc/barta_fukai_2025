"""Detect assembly activations using overlapping 100 ms windows with phase sharding.

This script computes per-pattern activation traces (fraction of member neurons
that fired at least once) from spike data using **100 ms windows** with large
overlap. To recover fine temporal resolution efficiently, it uses
**multiprocessing with phase offsets**:

- We launch N parallel jobs (here N=10) with `offset` values `9..0`.
- Each job calls :func:`analysis.get_spike_counts` with `dt=0.1` (100 ms) and a
  **phase shift** of `offset * 0.1` seconds (i.e., 0.0 s, 0.1 s, ..., 0.9 s).
- Each job returns a `(patterns × time)` array for *its* phase.
- We then stack all phase matrices and **interleave/reshape** columns to obtain
  a high-resolution activation timeline without changing the core counting code.
"""

import h5py
import argparse
from multiprocessing import Pool, get_logger
import logging

from utils import *
from analysis import get_spike_counts

import gc


def get_activations(offset):
    """Compute activation fractions for one **phase offset**.

    Parameters
    ----------
    offset : int or float
        Phase index passed from the process pool. This function multiplies it by
        `0.1` (seconds) when calling :func:`get_spike_counts`, yielding phase
        shifts of 0.0 s, 0.1 s, ..., 0.9 s for offsets 0..9.

    Returns
    -------
    numpy.ndarray or None
        A `(npat, T_phase)` float array with fraction-active per pattern for this
        phase. Returns ``None`` if an exception occurs (logged at ERROR level).
    """
    logger = logging.getLogger()
    logger.info(f"Processing offset {offset}, memory: {memory_usage()}")

    # patterns, spikes, max_t, npat are provided by the main block (globals here)
    patterns = load_patterns(npat)

    try:
        # dt=0.1 → 100 ms windows; offset*0.1 → phase shift in seconds
        _, sc = get_spike_counts(*spikes, max_t, dt=0.1, offset=offset * 0.1)

        activations = []
        for i in range(npat):
            # Fraction of members that spiked at least once per window
            activations.append((sc[patterns[i]] > 0).mean(axis=0))

            if i % 100 == 0:
                logger.info(f"Offset {offset}: Processed {i} patterns")

        result = np.array(activations)

        # Free memory in this worker
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
    parser.add_argument('--system', type=str, required=True)
    parser.add_argument('--namespace', type=str, required=True)
    parser.add_argument('--offset', type=float, default=10)

    args = parser.parse_args()
    stabilization_offset = args.offset  # seconds of data to discard from the start

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s")

    folder_path = data_path(args.namespace)
    filename = f"{folder_path}/{args.system}_{args.run}{args.patterns}.h5"

    logging.info(f"Loading data. {memory_usage()}")

    # Load spikes (excitatory) and simulation end time
    with h5py.File(filename, "r", swmr=True) as h5f:
        spikes = h5f['spikes_exc'][:].T  # shape (2, M) → [indices; times]
        max_t = h5f.attrs['simulation_time']

    # Discard early transients: keep only spikes after stabilization_offset
    spikes = spikes[:, spikes[1] > stabilization_offset]
    spikes[1] = spikes[1] - stabilization_offset
    max_t -= stabilization_offset

    logging.info(f"Data loaded. {memory_usage()}")

    # Number of patterns is also used in the worker (global)
    npat = args.patterns

    # Launch 5 worker processes across 10 phase offsets (9..0)
    activations = np.array(Pool(processes=5).map(get_activations, np.arange(10)[::-1]))

    # Shape now: (10, patterns, T_phase). Reorder to (patterns, T_phase, 10) then flatten time
    activations = np.transpose(activations, (1, 2, 0)).reshape((args.patterns, -1))

    logging.info(f"Activations calculated. {memory_usage()}")

    # Convert traces into sparse activation events by thresholding and run-lengths
    act_times = []
    durations = []
    pattern_ixs = []

    for i in range(args.patterns):
        change = np.diff((activations[i] > 0.9).astype(float))
        starts = np.argwhere(change == 1).flatten()
        ends = np.argwhere(change == -1).flatten()

        # Ensure proper start/end pairing within bounds
        if ends.size > 0:
            starts = starts[starts < ends[-1]]
            if starts.size > 0:
                ends = ends[ends > starts[0]]

            if starts.size > 0:
                act_times.extend(starts)
                pattern_ixs.extend([i] * starts.size)
                durations.extend(ends - starts)

        if len(durations) != len(act_times):
            import pdb; pdb.set_trace()

    try:
        activations_sparse = np.array([
            act_times,
            durations,
            pattern_ixs
        ])
    except Exception:
        import pdb; pdb.set_trace()

    output_file = f"{folder_path}/{args.system}_{args.run}{args.patterns}_activations.h5"

    logging.info(f"Saving results to {output_file}. {memory_usage()}")

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
