import argparse
import os
import yaml


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
    
    name = '_'.join([system['name'], run['name'], str(args.patterns)])
    
    bashrc = 'source ~/.bashrc'
    conda = 'conda activate StructuredInhibition'
    python = 'python simulation.py --system config/systems/hebb.yaml --run config/runtypes/default_train.yaml --patterns 1000'
    exit = 'exit'

    command = ' && '.join([bashrc, conda, python, exit])

    os.system(f"""
    screen -dmS {name} bash -c '{command}'
    """)