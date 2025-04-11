import pickle
import argparse
import h5py

from utils import create_stim_tuples, Patterns, data_path


if __name__ == '__main__':
    npat = 1000

    filename = f'{data_path()}/lognormal/init{npat}.h5'

    with h5py.File(filename, "r") as h5f:
        patterns = Patterns(
            h5f['connectivity/patterns/indices'][:],
            h5f['connectivity/patterns/splits'][:],
            neurons=h5f['connectivity'].attrs['N_exc']
        )
    
    tuples = create_stim_tuples(patterns=patterns, fraction=0.1, nstim=1000)

    with open('data/stimuli/frac10pc1000stim.pkl', 'wb') as f:
        pickle.dump(tuples, f)