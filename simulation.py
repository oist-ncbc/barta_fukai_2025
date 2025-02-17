import argparse
import yaml
import pickle
from pandas import read_csv
import h5py

from network_ch import run_network, load_stim_file
from single_neuron import run_network as sn_run
from utils import load_connectivity


if __name__ == '__main__':
    default_neuron = 'config/neurons/basic.yaml'

    parser = argparse.ArgumentParser()

    parser.add_argument('--system', type=str, required=True)
    parser.add_argument('--run', type=str, required=True)
    parser.add_argument('--patterns', type=int, required=True)

    args = parser.parse_args()

    with open(args.system) as f:
        system = yaml.safe_load(f)

    with open(args.run) as f:
        run = yaml.safe_load(f)

    with open('config/server_config.yaml') as f:
        server_config = yaml.safe_load(f)

    folder_path = f"{server_config['data_path']}/{run['folder']}"

    input_file  = f"{folder_path}/{run['init_matrix']}{args.patterns}.h5"
    output_file = f"{folder_path}/{system['name']}_{run['name']}{args.patterns}.h5"

    with h5py.File(input_file, "r", swmr=True) as src, h5py.File(output_file, "w") as dest:
        # Copy a group from source to destination
        src.copy("/connectivity", dest)  # Copies to the root of destination

    connectivity = load_connectivity(output_file)

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
        **connectivity,
        target_rate=target_rate,
        **system['background'],
        **system['neuron'],
        **run['run'],
        stimuli=stimulus_tuples,
        output_file=output_file
    )

    if 'isolate' in run:
        var_stats_filename = f"{folder_path}/var_stats/{system['name']}_conductances{args.patterns}_stats.csv"
        run['isolate']['var_stats'] = read_csv(var_stats_filename, index_col=[0,1], header=0)

        run_network(**simulation_params, isolate=run['isolate'])
    else:
        run_network(**simulation_params)