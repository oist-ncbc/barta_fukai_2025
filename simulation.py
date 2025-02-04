import argparse
import yaml
import pickle
from pandas import read_csv

from network import run_n_save, load_stim_file
from single_neuron import run_n_save as sn_run_n_save


if __name__ == '__main__':
    default_neuron = 'config/neurons/basic.yaml'

    parser = argparse.ArgumentParser()

    parser.add_argument('--system', type=str, required=True)
    parser.add_argument('--run', type=str, required=True)
    parser.add_argument('--patterns', type=int, required=True)
    parser.add_argument('--isolate', type=str, default=None)

    args = parser.parse_args()

    with open(args.system) as f:
        system = yaml.safe_load(f)

    with open(args.run) as f:
        run = yaml.safe_load(f)

    with open('config/server_config.yaml') as f:
        server_config = yaml.safe_load(f)

    folder_path = f"{server_config['data_path']}/{system['folder']}"

    if run['init_matrix'] == 'naive':
        matrix_file = f"{folder_path}/connectivity/{system['prefix']}{args.patterns}.pkl"
    else:
        matrix_file = f"{folder_path}/connectivity/{run['init_matrix']}_matrix.pkl"

    output_file = f"{folder_path}/data/{system['name']}_{run['name']}{args.patterns}.pkl"

    if run['save_matrix']:
        matrix_out = f"{folder_path}/connectivity/{system['name']}_{run['name']}{args.patterns}_matrix.pkl"
    else:
        matrix_out = None

    with open(matrix_file, 'rb') as file:
        matrix = dict()
        Z, N_exc, patterns, exc_alpha, delays, _ = pickle.load(file)

    if type(system['target_rate']) == str:
        with open(system['target_rate'],'rb') as f:
            target_rate = pickle.load(f)
    else:
        target_rate = system['target_rate']

    if run['stimulus'] is not None:
        stimulus_tuples = load_stim_file(
            run['stimulus']['file'],
            matrix['patterns'],
            randstim=run['stimulus']['random'],
            fraction=run['stimulus']['fraction'])
    else:
        stimulus_tuples = None


    simulation_params = dict(
        Z=Z,
        exc_alpha=exc_alpha,
        delays=delays,
        N_exc=N_exc,
        target_rate=target_rate,
        **system['background'],
        **system['neuron'],
        **run['run'],
        stimuli=stimulus_tuples,
    )

    if args.isolate:
        exc_stim, inh_stim = False, False
        if 'exc' in args.isolate:
            exc_stim = True
        if 'inh' in args.isolate:
            inh_stim = True

        varstats_e = read_csv(run['varstats_e'], index_col=None)
        varstats_i = read_csv(run['varstats_i'], index_col=None)

        simulation_params['exc_stim'] = exc_stim
        simulation_params['inh_stim'] = inh_stim
        simulation_params['varstats_e'] = varstats_e
        simulation_params['varstats_i'] = varstats_i

        sn_run_n_save(simulation_params, args, matrix_file=matrix_file, output=output_file, matrix_out=matrix_out)
    else:
        run_n_save(simulation_params, args, matrix_file=matrix_file, output=output_file, matrix_out=matrix_out)