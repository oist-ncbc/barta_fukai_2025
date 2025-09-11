"""Simulation entry point

Loads YAML configuration files, prepares connectivity and stimuli, and calls
`run_network` (or the single-neuron variant) to execute simulations. This file
is designed as a thin orchestrator: it **does not** implement neuron or synapse
models itself; those live in `network_ch.py`.

Expected YAMLs
--------------
- **system.yaml**: identifies the type of network.
- **run.yaml**: what kind of simulation should be ran with the network.

Command-line overview
---------------------
This script accepts paths to those YAMLs, plus a few convenience flags to
control stimuli and output file naming.
"""


import argparse
import pickle
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml
from pandas import read_csv

from network_ch import run_network
from utils import (
    data_path,
    load_connectivity,
    load_patterns,
    create_stim_tuples,
)


# ----------------------------
# Helpers
# ----------------------------

def _load_yaml(path: Path) -> Dict[str, Any]:
    """Load a YAML file and return a dictionary.

    Parameters
    ----------
    path : Path
        Path to a YAML file.

    Returns
    -------
    dict
        Parsed YAML as a Python dictionary.
    """
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _build_stimuli(
    patterns,
    stim_file: Path | None,
    fraction: float,
    nstim: int,
    duration: float,
    spacing: float,
    random: bool,
):
    """Construct stimulus tuples `(start, end, indices)`.

    If `stim_file` is provided, it should point to a CSV with rows describing
    `(start, end, pattern_id, is_random)`; otherwise we synthesize stimuli from
    the first `nstim` patterns via :func:`utils.create_stim_tuples`.
    """
    if stim_file is not None and stim_file.exists():
        import pandas as pd

        stims = pd.read_csv(stim_file, header=None, index_col=False).values
        tuples = []
        for x in stims:
            pt = patterns[int(x[2])]
            num_indices = int(len(pt) * fraction)
            ind_ix = np.random.permutation(len(pt))[:num_indices]
            if not bool(x[3]):
                tuples.append((float(x[0]), float(x[1]), pt[ind_ix]))
            else:
                rand_pt = np.random.permutation(patterns.neurons)[: len(pt)]
                tuples.append((float(x[0]), float(x[1]), rand_pt[ind_ix]))
        return tuples

    # Default: synthesize from patterns
    return create_stim_tuples(
        patterns=patterns,
        num_indices=None if fraction >= 1 else int(max(1, fraction * patterns.sizes().max())),
        nstim=nstim,
        duration=duration,
        spacing=spacing,
        random=random,
    )


# ----------------------------
# CLI
# ----------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line options.

    Returns
    -------
    argparse.Namespace
        Resolved arguments.
    """
    p = argparse.ArgumentParser(description="Run network simulation from YAML configs.")

    # Config paths
    p.add_argument("--system", type=Path, required=True, help="Path to system YAML")
    p.add_argument("--run", type=Path, required=True, help="Path to run YAML")
    p.add_argument(
        "--neuron",
        type=Path,
        default=Path("config/neurons/basic.yaml"),
        help="Path to neuron YAML (overrides)",
    )

    # Stimuli
    p.add_argument("--stimulus", type=Path, default=None, help="CSV stimulus file (optional)")
    p.add_argument("--stimfrac", type=float, default=1.0, help="Fraction of pattern to stimulate")
    p.add_argument("--nstim", type=int, default=10, help="Number of stimuli to synthesize if no file")
    p.add_argument("--stimdur", type=float, default=0.1, help="Stimulus duration (s)")
    p.add_argument("--spacing", type=float, default=1.0, help="Inter-stimulus spacing (s)")
    p.add_argument("--randstim", action="store_true", help="Randomize indices inside each pattern")

    # Output
    p.add_argument("--patterns", type=int, required=True, help="Number of patterns (npat)")
    p.add_argument("--output", type=Path, required=True, help="Output HDF5 path")

    # Mode
    p.add_argument("--single-neuron", action="store_true", help="Run single-neuron variant")

    return p.parse_args()


def main() -> None:
    """Entry point for running simulations from configuration files.

    This function:
    1. Loads YAMLs for system/run.
    2. Loads connectivity and patterns for the specified `system` and `npat`.
    3. Builds stimulus tuples from a CSV or synthesized schedule.
    4. Assembles keyword arguments for :func:`network_ch.run_network`.
    5. Executes the simulation and writes outputs to HDF5.
    """
    args = parse_args()

    # Load configs
    system = _load_yaml(args.system)
    run = _load_yaml(args.run)

    # Namespace/folder for this dataset
    namespace = system.get("namespace", "lognormal")
    folder_path = Path(data_path()) / namespace

    # Connectivity & patterns
    conn = load_connectivity(system["name"], run.get("name", "train"), args.patterns, folder=namespace)
    patterns = load_patterns(args.patterns, system=system["name"], run=run.get("name", "train"), folder=namespace)

    # Stimulus tuples (CSV or synthesized)
    stimulus_tuples = _build_stimuli(
        patterns=patterns,
        stim_file=args.stimulus,
        fraction=float(args.stimfrac),
        nstim=int(args.nstim),
        duration=float(args.stimdur),
        spacing=float(args.spacing),
        random=bool(args.randstim),
    )

    # Target rate can be scalar or a vector
    target_rate = run.get("target_rate", 3.0)
    rate_file = run.get("rate_file")
    if rate_file:
        with open(rate_file, "rb") as f:
            target_rate = np.array(pickle.load(f))

    # Thresholds optional
    thresholds = None
    thr_file = run.get("thr_file")
    if thr_file:
        with open(thr_file, "rb") as f:
            thresholds = pickle.load(f)

    # Assemble kwargs for run_network
    simulation_params = dict(
        weights=conn["weights"],
        exc_alpha=conn["exc_alpha"],
        delays=conn["delays"],
        N_exc=conn["N_exc"],
        N_inh=conn["N_inh"],
        target_rate=target_rate,
        thresholds=thresholds,
        **system.get("background", {}),
        **system.get("neuron", {}),
        **run.get("run", {}),
        stimuli=stimulus_tuples,
        output_file=str(args.output),
    )

    # Optional isolated mode
    if "isolate" in run:
        # Attach var_stats for isolation (per E/I) if present on disk
        var_stats_filename = folder_path / "var_stats" / f"{system['name']}_conductances{args.patterns}_stats.csv"
        if var_stats_filename.exists():
            run["isolate"]["var_stats"] = read_csv(var_stats_filename, index_col=[0, 1], header=0)

        run_network(**simulation_params, isolate=run["isolate"])  # noqa: E501
    else:
        run_network(**simulation_params)


if __name__ == "__main__":
    main()
