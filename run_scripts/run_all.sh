#!/bin/bash

# Ensure an argument is provided
if [[ -z "$1" ]]; then
    echo "Error: Missing argument for patterns."
    exit 1
fi

python simulation.py --system config/systems/hebb.yml --run config/runtypes/default_train.yml --patterns "$1"
python simulation.py --system config/systems/hebb.yml --run config/runtypes/conductances.yml --patterns "$1"
python gstats_multiproc.py --name "hebb_conductances_$1.py" --folder lognormal --patterns "$1"
python simulation.py --system config/systems/hebb.yml --run config/runtypes/perturbation.yml --patterns "$1"
