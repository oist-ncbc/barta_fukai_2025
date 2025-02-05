import argparse
import yaml
import subprocess


def run_locally(screen, python, log_file, conda_env="StructuredInhibition"):
    """Runs the script locally inside a screen session with a Conda environment and logs errors."""
    # command = [
    #     "screen", "-S", screen, "bash", "-c",
    #     f"'source ~/miniconda3/etc/profile.d/conda.sh && conda activate {conda_env} && {python} > {log_file} 2>&1 && exec bash'"
    # ]
    command = ["screen", "-dmS", screen, "bash", "-l", "-c",
               f"source ~/miniconda3/etc/profile.d/conda.sh && conda activate StructuredInhibition && {python} > {log_file} 2>&1"]
    print(f"Running locally: {' '.join(command)}")

    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running local script: {e}")

def run_on_ssh(screen, python, log_file, server_address, remote_folder):
    """Runs the script remotely inside a screen session and logs errors."""
    ssh_command = f"cd {remote_folder} && screen -dmS {screen} bash -c 'source ~/.bashrc && conda activate StructuredInhibition && {python} > {log_file} 2>&1'"
    full_command = ["ssh", server_address, ssh_command]

    print(f"Running on SSH: {' '.join(full_command)}")

    try:
        subprocess.run(full_command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running script on SSH: {e}")

if __name__ == '__main__':
    default_neuron = 'config/neurons/basic.yaml'

    parser = argparse.ArgumentParser()

    parser.add_argument('--system', type=str, required=True)
    parser.add_argument('--run', type=str, required=True)
    parser.add_argument('--patterns', type=int, required=True)
    parser.add_argument('--isolate', type=str, default=None)
    parser.add_argument('--ssh', type=str, default=None)

    args = parser.parse_args()

    with open(args.system) as f:
        system = yaml.safe_load(f)

    with open(args.run) as f:
        run = yaml.safe_load(f)
    
    name = '_'.join([system['name'], run['name'], str(args.patterns)])
    log_file = f'logs/{name}.txt'
    
    bashrc = 'source ~/.bashrc'
    conda = 'conda activate StructuredInhibition'

    if args.isolate is not None:
        python = f'python simulation.py --system {args.system} --run {args.run} --patterns {args.patterns} --isolate {args.isolate}'
    else:
        python = f'python simulation.py --system {args.system} --run {args.run} --patterns {args.patterns}'

    exit_screen = 'exit'
    command = ' && '.join([bashrc, conda, python, exit_screen])

    if args.ssh is not None:
        server_address = f"tomas-barta@{args.ssh}.oist.jp"
        remote_folder = "StructuredInhibition"

        run_on_ssh(name, python, log_file, server_address, remote_folder)
    else:
        run_locally(name, python, log_file)