# Inhibitory Plasticity and Memory Replay Simulations

This repository contains code to reproduce simulations of spiking neural networks stabilized with different inhibitory plasticity rules.
The workflow includes generating network connectivity, running spiking simulations, detecting assembly activations, and approximating effective connectivity with linearized neuron sensitivities.

---

## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/oist-ncbc/barta_fukai_2025.git
   cd barta_fukai_2025
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate   # on Linux/Mac
   venv\Scripts\activate      # on Windows
   ```

3. Install dependencies from `requirements.txt`:

   ```bash
   pip install -r requirements.txt
   ```

---

## Workflow

The following steps reproduce a typical experiment with **1000 memory assemblies**.
Adjust the number of assemblies with `--patterns` as needed.

### 1. Generate connectivity matrix

Hebbian terms in excitatory–excitary connections are computed, and the strongest ones are selected. Lognormal weights are assigned to create the base connectivity.

```bash
python genconn.py --patterns 1000
```

---

### 2. Train the network with fLHP inhibitory plasticity

Run training for **2000 s** with the Vogels & Sprekeler rule (`fLHP`).

```bash
python simulation.py \
  --system config/systems/hebb.yml \
  --run config/runtypes/default_train.yml \
  --patterns 1000
```

---

### 3. Spontaneous activity

Run the network without external training input for **10,000 s** to obtain spontaneous replay.

```bash
python simulation.py \
  --system config/systems/hebb.yml \
  --run config/runtypes/spontaneous.yml \
  --patterns 1000
```

---

### 4. Detect assembly activations

Identify transient replays of assemblies from the spontaneous activity.

```bash
python get_activations.py \
  --patterns 1000 \
  --system hebb \
  --run spontaneous
```

---

### 5. Save excitatory and inhibitory conductances

Run a simulation that records excitatory (`g_e`) and inhibitory (`g_i`) synaptic conductances.

```bash
python simulation.py \
  --system config/systems/hebb.yml \
  --run config/runtypes/conductances.yml \
  --patterns 1000
```

---

### 6. Extract robust conductance statistics

Use the Minimum Covariance Determinant (MCD) estimator to extract robust statistics of synaptic input.

```bash
python gstats_multiproc.py \
  --name hebb_conductances \
  --folder lognormal \
  --patterns 1000
```

---

### 7. Perturbation simulations

Run the network with isolated neurons and small perturbations to excitatory and inhibitory input.

```bash
python simulation.py \
  --system config/systems/hebb.yml \
  --run config/runtypes/perturbation.yml \
  --patterns 1000
```

---

### 8. Estimate neuron sensitivities

Analyze perturbation results to compute each neuron's sensitivity to external input.
Modify parameters directly inside:

```bash
python linear_sensitivity.py
```

---

## Notes

* Configurations are organized under `config/systems/` (network setup) and `config/runtypes/` (simulation protocols).
* Simulation logs and outputs are stored in HDF5 format for efficient handling of large datasets.
* The `analysis/` folder contains scripts to process the raw data as well as plotting scripts to generate the figures used in the manuscript.
* Figures and further analyses (e.g., eigenvalue spectra, replay diversity) can be generated using the analysis scripts in this repository.

---
