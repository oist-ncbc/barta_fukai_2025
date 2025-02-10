import numpy as np
from tqdm import tqdm
from itertools import combinations, product
import argparse
import yaml
import h5py
from scipy.sparse import csr_array

from utils import create_weight_dataset


def genconn(N, N_exc, P, f, sparsity, i_factor, circular=False, rescale=True,
            var_factor=0.5, fix_size=False, spread=0, seed=42, distribution='lognormal'):
    """
    Generate connectivity matrix by calculating Hebbian terms in EE connectivity, selecting the highest ones
    and overlaying lognormal weight distribution.
    :param N: number of neurons (exc + inh)
    :param N_exc: number of excitatory neurons
    :param P: number of patterns
    :param f: sparsity of patterns
    :param sparsity: sparsity of network. tuple of floats (sparse_ee, sparse_ie, sparse_ii, sparse_ei)
    :param circular: whether connections should be made between two subsequent patterns
    :param normalize: if post-synaptic weights should be rescaled based on how many patterns the pre-synaptic neurons
        participates in
    :param seed: seed for reproducibility
    :return: returns connectivity weight matrix (ndarray NxN), and the embedded patterns (ndarray PxN_exc)
    """
    np.random.seed(seed)

    if not fix_size:
        patterns = (np.random.rand(P, N_exc) < f).astype(float)
    else:
        patterns = []
        pat_len = int(N_exc * f)

        for ii in range(P):
            pattern = np.random.permutation(np.concatenate([np.ones(pat_len), np.zeros(N_exc-pat_len)]))
            patterns.append(pattern)

        patterns = np.array(patterns)

    Z0 = np.zeros((N,N))

    spread_arr = np.linspace(-1, 1, len(patterns)) * spread + 1

    for pattern, q in tqdm(zip(patterns, spread_arr), total=len(patterns)):
        pairs = np.array(list(combinations(np.argwhere(pattern).flatten(), 2)))
        x, y = pairs.T
        Z0[x,y] += 1 * q
        Z0[y,x] += 1 * q

    if circular:
        for pattern1, pattern2 in tqdm(zip(patterns, np.roll(patterns, shift=1, axis=0)), total=len(patterns)):
            pairs = np.array(list(product(np.argwhere(pattern1).flatten(), np.argwhere(pattern2).flatten())))
            x, y = pairs.T
            Z0[x,y] += 0.5
            Z0[y,x] += 0.5

        for pattern1, pattern2 in tqdm(zip(patterns, np.roll(patterns, shift=-1, axis=0)), total=len(patterns)):
            pairs = np.array(list(product(np.argwhere(pattern1).flatten(), np.argwhere(pattern2).flatten())))
            x, y = pairs.T
            Z0[x,y] += 0.5
            Z0[y,x] += 0.5

    # delete self-synapses
    for i in range(N):
        Z0[i,i] = 0

    sparse_ee, sparse_ie, sparse_ii, sparse_ei = sparsity

    Z = np.copy(Z0)

    Z_slice = Z[:N_exc,:N_exc]
    Z_slice += np.random.rand(N_exc, N_exc)
    limit = np.percentile(Z_slice, 100-100*sparse_ee)
    Z_slice[Z_slice < limit] = 0
    weights_ee = (Z[:N_exc,:N_exc][Z[:N_exc,:N_exc] != 0])

    E = 0.25
    Var = E*var_factor

    sig = np.sqrt(np.log(Var/(E*E) + 1))
    mu = np.log(E) - sig**2 / 2

    if distribution == 'lognormal':
        rvs_exc = np.sort(np.exp(np.random.randn(len(weights_ee))*sig + mu))
    elif distribution == 'normal':
        rvs_exc = np.sort(np.random.randn(len(weights_ee))*np.sqrt(Var) + E)
    else:
        raise ValueError(f'Distribution "{distribution}" is not supported.')

    Z[:N_exc,:N_exc][Z[:N_exc,:N_exc] != 0] = rvs_exc[np.argsort(np.argsort(weights_ee))]

    slices = [
        (N_exc, N, 0, N_exc),
        (N_exc, N, N_exc, N),
        (0, N_exc, N_exc, N)
    ]

    for slice, sp_val in zip(slices, [sparse_ie, sparse_ii, sparse_ei]):
        Z_slice = Z[slice[0]:slice[1],slice[2]:slice[3]]
        Z[slice[0]:slice[1],slice[2]:slice[3]] = (np.random.rand(*Z_slice.shape) < sp_val).astype(float) * E * i_factor

    if rescale:
        pattern_participation = patterns.sum(axis=0)

        for ii in range(N_exc):
            factor = np.exp((pattern_participation[ii] - P*f) / P*f)
            Z[:,ii] = Z[:,ii] / factor

    Z[N_exc:,N_exc:] = Z[N_exc:,N_exc:] * 6
    Z[:N_exc,N_exc:] = Z[:N_exc,N_exc:] * 2
    Z[N_exc:,:N_exc] = Z[N_exc:,:N_exc] * 2

    # if rescale:
    #     for i in tqdm(range(N_exc)):
    #         Z[i,:8000] = Z[i,:8000] / Z[i,:8000].sum()

    return Z, patterns

def get_aux_prop(Z):
    """
    Generates synaptic delays between neurons and a random neuronal parameter exc_alpha for each excitatory neuron
    :param Z: synaptic weight matrix
    :return: tuple of ndarrays with delays for each existing synapse (delays_ee, delays_ie, delays_ii, delays_ei)
        and an ndarray with a random number for each excitatory neuron
    """
    exc_alpha = np.random.randn(8000)
    delays = np.random.rand(*Z.shape) * 2
    delays[:,:8000] += 1

    delays_ee = delays[:8000,:8000][Z[:8000,:8000] != 0]
    delays_ie = delays[8000:,:8000][Z[8000:,:8000] != 0]
    delays_ii = delays[8000:,8000:][Z[8000:,8000:] != 0]
    delays_ei = delays[:8000,8000:][Z[:8000,8000:] != 0]

    delays_tuple = (delays_ee, delays_ie, delays_ii, delays_ei)

    return delays_tuple, exc_alpha


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-n', '--neurons', type=int, default=10000)
    parser.add_argument('--nexc', type=int, default=8000)
    parser.add_argument('-p', '--patterns', type=int, default=2000)
    parser.add_argument('--pattern_sparsity', type=float, default=0.01)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--circular', action='store_true')
    parser.add_argument('--fix_size', action='store_true')
    parser.add_argument('--ee_sparse', type=float, default=0.05)
    parser.add_argument('--i_sparse', type=float, default=0.1)
    parser.add_argument('--i_factor', type=float, default=1)
    parser.add_argument('--folder', type=str, default='')
    parser.add_argument('--name', type=str, default='')
    parser.add_argument('--var', type=float, default=0.5)
    parser.add_argument('--spread', type=float, default=0)

    args = parser.parse_args()

    N_exc = args.nexc

    network_sparsity = (args.ee_sparse, args.i_sparse, args.i_sparse, args.i_sparse)

    Z, patterns = genconn(N=args.neurons, N_exc=N_exc, P=args.patterns, f=args.pattern_sparsity,
                sparsity=network_sparsity, circular=args.circular, seed=args.seed, i_factor=args.i_factor,
                          var_factor=args.var, fix_size=args.fix_size, spread=args.spread)

    delays_tuple, exc_alpha = get_aux_prop(Z)

    csr_patterns = csr_array(patterns) 

    limits = {
        'E': (0,8000),
        'I': (8000,10000)
    }

    with open('config/server_config.yaml') as f:
        server_config = yaml.safe_load(f)

    data_path = f"{server_config['data_path']}"

    output_filename = f'{data_path}/{args.folder}/init{args.patterns}.h5'

    with h5py.File(output_filename, "w") as h5f:  # Open file in append mode
        connectivity_group = h5f.require_group("connectivity")
        weights_group = connectivity_group.require_group("weights")  # Ensure "weights" group exists

        for pre in ['E','I']:
            for post in ['E','I']:
                cut = Z[limits[post][0]:limits[post][1], limits[pre][0]:limits[pre][1]].T
                sources, targets = np.nonzero(cut)
                weights = cut[cut != 0]

                label = f'{post}{pre}'
                create_weight_dataset(weights_group.require_group(label), "sources", sources, dtype=np.uint16)
                create_weight_dataset(weights_group.require_group(label), "targets", targets, dtype=np.uint16)
                create_weight_dataset(weights_group.require_group(label), "weights", weights)
                

        create_weight_dataset(connectivity_group, 'exc_alpha', exc_alpha)

        delays_group = connectivity_group.require_group('delays')
        create_weight_dataset(delays_group, "EE", delays_tuple[0], dtype=np.float32)
        create_weight_dataset(delays_group, "IE", delays_tuple[1], dtype=np.float32)
        create_weight_dataset(delays_group, "II", delays_tuple[2], dtype=np.float32)
        create_weight_dataset(delays_group, "EI", delays_tuple[3], dtype=np.float32)

        patterns_group = connectivity_group.require_group('patterns')
        create_weight_dataset(patterns_group, "indices", csr_patterns.indices)
        create_weight_dataset(patterns_group, "splits", csr_patterns.indptr[1:-1])

        connectivity_group.attrs['N_exc'] = N_exc
        connectivity_group.attrs['N_inh'] = args.neurons - N_exc
        connectivity_group.attrs['N_patterns'] = args.patterns


    # ZEE = Z[:N_exc,:N_exc].T
    # sources, targets = np.nonzero(ZEE)
    # weights = ZEE[ZEE != 0]

    # with open(args.output, 'wb') as file:
    #     pickle.dump((Z, N_exc, patterns, exc_alpha, delays_tuple, vars(args)), file)