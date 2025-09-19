"""Spiking network simulation with Brian2

This module defines utilities and a high-level `run_network` routine to simulate a
mixed excitatory/inhibitory spiking network under several inhibitory plasticity
rules (Hebbian, rate-based, iDIP, and threshold adaptation). Results (spikes,
state variables, and optionally weights) are appended to an HDF5 file so long
runs can be chunked without exhausting memory.

Conventions
-----------
- Units follow Brian2; where arrays are exported to HDF5 they are stored as raw
  floats after dividing by the specified `default_units` (see `default_units`).
- Time is in seconds unless noted; the simulation time-step is set globally via
  `defaultclock.dt = 0.1*ms`.
- Connectivity is passed in `weights`/`delays` dicts with keys "EE", "IE",
  "EI", "II" mapping to arrays of sources, targets, and weights (nS) / delays (ms).
"""

from brian2 import *
import numpy as np
import argparse
import pickle
import pandas as pd
import h5py
import logging
import time

import os
import psutil
import gc

from analysis import get_spike_counts


def memory_usage():
    """Return a short human-readable string summarizing current process memory.

    The string includes resident set size (RSS) and virtual memory size (VMS)
    in megabytes for quick logging during long simulations.

    Returns
    -------
    str
        Formatted memory summary, e.g. ``"[MEMORY] RSS: 512.34 MB, VMS: 2048.12 MB"``.
    """
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return f"[MEMORY] RSS: {mem_info.rss / 1e6:.2f} MB, VMS: {mem_info.vms / 1e6:.2f} MB"


def seconds_to_hms(seconds):
    """Convert seconds to ``HH:MM:SS`` string (UTC-based via ``time.gmtime``).

    Parameters
    ----------
    seconds : float
        Duration in seconds.

    Returns
    -------
    str
        Time formatted as ``"HH:MM:SS"``.
    """
    return time.strftime("%H:%M:%S", time.gmtime(seconds))


def get_stim_matrix(stimuli, N_exc, simulation_time, dt=0.1):
    """Build a binary stimulus matrix for excitatory neurons over time.

    Parameters
    ----------
    stimuli : list[tuple[float, float, array_like]]
        Each tuple is ``(start, end, neurons)`` in seconds, where ``neurons`` is
        an iterable of excitatory neuron indices to stimulate in the interval
        ``[start, end)``.
    N_exc : int
        Number of excitatory neurons.
    simulation_time : float
        Total simulation duration in seconds; the matrix spans ``[0, simulation_time]``.
    dt : float, optional
        Temporal resolution in seconds for the stimulus matrix (default 0.1 s).

    Returns
    -------
    ndarray, shape (N_exc, T)
        Binary matrix where ``stim_matrix[i, t] = 1`` if neuron ``i`` is
        stimulated at time index ``t``.
    """
    time_arr = np.arange(0, simulation_time + 1e-5, dt)
    stim_matrix = np.zeros((N_exc, len(time_arr)), dtype=float)

    for start, end, neurons in stimuli:
        for ix in neurons:
            mask = (time_arr >= start) & (time_arr < end)
            stim_matrix[ix][mask] = 1.0

    return stim_matrix


def ei_plasticity_eqs(plasticity, learning_rate):
    """Return Brian2 pre/post event code snippets for inhibitory → excitatory plasticity.

    This helper produces strings used in ``Synapses(..., on_pre=..., on_post=...)``
    to implement several plasticity modes. When ``learning_rate <= 0``, updates
    are disabled and only synaptic current is applied.

    Parameters
    ----------
    plasticity : {"threshold", "rate", "hebb", "idip"}
        Plasticity mechanism to use.
    learning_rate : float
        Learning rate controlling the magnitude of weight/threshold updates.
        If ``<= 0``, plasticity is effectively off.

    Returns
    -------
    tuple[str, str | None]
        ``(pre_eqs_inh, post_eqs_inh)`` code strings. ``post_eqs_inh`` may be
        ``None`` for one-sided rules.

    Raises
    ------
    ValueError
        If an unknown ``plasticity`` name is supplied.
    """
    if learning_rate > 0:
        if plasticity == 'rate':
            pre_eqs_inh = '''
                    gi += w*nS
                    w = w + learning_rate * (network_rate-target_rate) * (x_post + 1)'''

            post_eqs_inh = '''
                    w = w + learning_rate * (network_rate-target_rate) * x_pre'''

        elif plasticity == 'hebb':
            pre_eqs_inh = '''
                    gi += w*nS
                    alpha = neuron_target_rate_post*Hz*tau_stdp*2
                    w = clip(w+(x_post-alpha)*learning_rate*20/(tau_stdp/ms), 0, 100)'''
            # factor 20/tau_stdp normalizes learning rate

            post_eqs_inh = '''
                    w = clip(w+x_pre*learning_rate*20/(tau_stdp/ms), 0, 100)'''

        elif plasticity == 'idip':
            pre_eqs_inh = '''
                    gi += w*nS
                    w = clip(w+(y_pre-17)*learning_rate, 0, 100)'''
            post_eqs_inh = None
        elif plasticity == 'threshold':
            pre_eqs_inh = '''
                    gi += w*nS
            '''
            post_eqs_inh = None
        else:
            raise ValueError(
                f"plasticity can be 'threshold', 'rate', 'hebb', or 'idip'. Received '{plasticity}' instead."
            )
    else:
        pre_eqs_inh = '''
                    gi += w*nS
            '''
        post_eqs_inh = None

    return pre_eqs_inh, post_eqs_inh


def run_network(
    weights,
    exc_alpha,
    delays,
    N_exc,
    N_inh,
    alpha1,
    alpha2,
    reset_potential,
    target_rate,
    plasticity,
    background_poisson,
    poisson_amplitude,
    output_file,
    simulation_time,
    learning_rate,
    state_variables=None,
    stimuli=None,
    thresholds=None,
    seed_num=42,
    tau_stdp_ms=20,
    recharge=0,
    save_weights=False,
    isolate=None,
    chunk_size=None,
    plast_ii=False,
    inhf=None,
    shuffle=False,
):
    """Simulate a spiking E/I network with optional plasticity and stimuli.

    The function supports (a) a connected network with Poisson background input
    and optional stimulus drive, and (b) an **isolation mode** where neurons are
    disconnected from network synapses and receive parameterized external noise
    (Ornstein–Uhlenbeck-like conductance drive) with optional perturbations.
    Results are appended to an HDF5 file to allow long runs in time chunks.

    Parameters
    ----------
    weights : dict
        Mapping of connection labels to connectivity arrays. For each label in
        {"EE", "IE", "EI", "II"}, the value is a dict with keys:
        ``{"sources", "targets", "weights"}`` (all 1D arrays). ``weights`` are
        in nanosiemens (nS). Example::

            weights = {
                'EE': {'sources': np.array([...]), 'targets': np.array([...]), 'weights': np.array([...])},
                'IE': {...}, 'EI': {...}, 'II': {...}
            }

    exc_alpha : ndarray, shape (N_exc,)
        Per-neuron parameter controlling the fast adaptation increment for
        excitatory cells (used when ``alpha1 is True``).
    delays : dict
        Mapping of connection labels {"EE", "IE", "EI", "II"} to delay arrays in
        milliseconds; each array length must match the number of synapses for
        the corresponding connection.
    N_exc : int
        Number of excitatory neurons.
    N_inh : int
        Number of inhibitory neurons.
    alpha1 : bool
        Enable (True) or disable (False) the short-timescale adaptation for
        excitatory cells.
    alpha2 : float
        Slow adaptation increment (mV) added to ``H2`` on spike.
    reset_potential : bool
        If True, reset membrane potential to ``EL`` on spike.
    target_rate : float or ndarray
        Target firing rate (Hz). A scalar applies to all excitatory neurons; or
        an array of shape (N_exc,) sets per-neuron targets.
    plasticity : {"hebb", "rate", "idip", "threshold"}
        Inhibitory→excitatory plasticity mode. ``"threshold"`` adjusts neuron
        thresholds toward the target rate instead of synaptic weights.
    background_poisson : float
        Background Poisson input rate (kHz) per neuron (applied to ``ge``).
    poisson_amplitude : float
        Conductance increment (nS) per background/stimulus spike.
    output_file : str
        Path to an HDF5 (``.h5``) file where spikes, state variables, and
        optional weights are appended.
    simulation_time : float
        Total simulation duration in seconds.
    learning_rate : float
        Plasticity learning-rate hyperparameter. If ``<= 0``, plasticity is off.
    state_variables : list[str], optional
        Neuron state variable names to record (e.g., ``["v", "ge", "gi"]``).
    stimuli : list[tuple[float, float, array_like]] | None, optional
        Optional stimulus schedule for excitatory neurons as ``(start, end, idxs)``.
    thresholds : ndarray | None, optional
        Initial thresholds (mV) for excitatory neurons when ``plasticity='threshold'``.
    seed_num : int, default 42
        Random seed for reproducibility.
    tau_stdp_ms : float, default 20
        STDP time constant (ms) used in plasticity equations.
    recharge : float, default 0
        Modulates spike-triggered ``x`` update: ``rech>0`` downweights bursts,
        ``rech<0`` favors coincidence detection.
    save_weights : bool, default False
        If True, append final EI weights (``Sei.w``) to the output file.
    isolate : dict | None, optional
        If provided, switches to isolation mode with keys like
        ``{"exc_stim": bool, "inh_stim": bool, "stim_count": int, "var_stats": DataFrame, "strength": {"weak"|"strong"}}``.
        See inline comments for details.
    chunk_size : float | None, optional
        Wall-clock chunk length in seconds for simulation segments. Defaults to
        10 s (connected mode) or to the initial perturbation duration (isolation).
    plast_ii : bool, default False
        If True, enable Hebbian-like i→i plasticity (``Sii``) instead of fixed ``gi``.
    inhf : float | None, optional
        Global scaling factor for inhibitory conductance on excitatory cells
        (``inhf`` field in equations). If None, set to 1.
    shuffle : bool, default False
        If True, shuffle EI weight vector before simulation (diagnostics).

    Returns
    -------
    None
        Data are written to ``output_file``; nothing is returned in memory.

    Notes
    -----
    - A helper "rate unit" integrates population rate and shares it with
      excitatory cells (via a summed field) to support heterosynaptic rules.
    - HDF5 datasets for spikes are created once and extended (resized) after
      each chunk; this avoids keeping all spikes in RAM.
    - ``default_units`` defines scaling used when writing state variables.

    Examples
    --------
    Run a 100 s network simulation and persist spikes/state variables::

        run_network(
            weights=weights_dict,
            exc_alpha=alpha,
            delays=delays_dict,
            N_exc=8000,
            N_inh=2000,
            alpha1=True,
            alpha2=2,
            reset_potential=False,
            target_rate=5.0,
            plasticity='hebb',
            background_poisson=1.5,
            poisson_amplitude=0.3,
            output_file='simulation_results.h5',
            simulation_time=100,
            learning_rate=1e-3,
            state_variables=['v', 'ge', 'gi'],
            save_weights=True,
        )
    """

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    meta_eta = 0  # global modifier for burst-driven target-rate adaptation (used in a run_on_event below)

    np.random.seed(seed_num)

    # Use a 0.1 ms time step for numerical integration across the model
    defaultclock.dt = 0.1 * ms

    # Neural model parameters
    # _______________________
    C = 200 * pF
    EL = -80 * mV
    refrac = 2 * ms

    # MAT threshold parameters
    omega = -55 * mV
    tau_th1 = 10 * ms
    tau_th2 = 200 * ms

    # Synaptic parameters
    taue = 6 * ms   # excitatory synapse time constant
    taui = 6 * ms   # inhibitory synapse time constant
    Ee = 0 * mV     # excitatory reversal potential
    Ei = -80 * mV   # inhibitory reversal potential

    tau_stdp = tau_stdp_ms * ms  # STDP time constant
    tidip = 160 * ms
    tau_rate = 10 * second  # very long integration for smooth rate estimate
    tau_coincidence = 150 * ms
    # ________________________

    # Build membrane/synapse equations; in isolation mode we inject OU-like drive
    if isolate is not None:
        Isyn = 'Isyn = -(ge + gext(t))*(v-Ee)-(gi + gext_inh(t))*(v-Ei) : amp'
        dgedt = 'dge/dt = (mu_e - ge) / taue + sigma_e * sqrt(2 / taue) * xi_e : siemens'
        dgidt = 'dgi/dt = (mu_i - gi) / taui + sigma_i * sqrt(2 / taui) * (rho*xi_e+sqrt(1-rho**2)*xi_i) : siemens'
        mu_e = 'mu_e : siemens'
        mu_i = 'mu_i : siemens'
        sigma_e = 'sigma_e : siemens'
        sigma_i = 'sigma_i : siemens'
        rho = 'rho : 1'

        # Perturbative input schedule in isolation mode
        # --------------------------------------------
        if isolate['strength'] == 'weak':
            stim_steps = 5
            step_time = 0.1
            input_list = np.linspace(-0.35, 0.35, 8)
        elif isolate['strength'] == 'strong':
            stim_steps = 100
            step_time = 0.01
            input_list = np.array([10, 20])

        pair = np.array([1] + [0] * (stim_steps - 1))
        init_steps = stim_steps * len(input_list)

        input_arr_exc = np.zeros(init_steps)
        input_arr_inh = np.zeros(init_steps)

        for ii in range(isolate['stim_count']):
            for inp in input_list:
                if isolate['exc_stim']:
                    input_arr_exc = np.append(input_arr_exc, pair * inp)
                    input_arr_inh = np.append(input_arr_inh, np.zeros(stim_steps))

            for inp in input_list:
                if isolate['inh_stim']:
                    input_arr_inh = np.append(input_arr_inh, pair * inp * 2)
                    input_arr_exc = np.append(input_arr_exc, np.zeros(stim_steps))

        simulation_time = len(input_arr_exc) * step_time

        if chunk_size is None:
            chunk_size = init_steps * step_time

        gext = TimedArray(input_arr_exc * nS, dt=step_time * second)
        gext_inh = TimedArray(input_arr_inh * nS, dt=step_time * second)
        # --------------------------------------------
    else:
        Isyn = 'Isyn = -(ge)*(v-Ee)-inhf*(gi)*(v-Ei) : amp'
        dgedt = 'dge/dt = -ge/taue : siemens'
        dgidt = 'dgi/dt = -gi/taue : siemens'
        mu_e = ''
        mu_i = ''
        sigma_e = ''
        sigma_i = ''
        rho = ''

        if chunk_size is None:
            chunk_size = 10

    # Full neuron model string assembled for Brian2
    eqs_str = f'''
    dv/dt = (-gL*(v-EL) + Isyn) / C : volt (unless refractory)
    {Isyn}
    
    dx/dt = -x/tau_stdp : 1
    dq/dt = -q/tau_coincidence : 1
    
    {dgedt}
    {dgidt}
    
    dy/dt = (-y + ge/nS)/tidip : 1
    
    dz/dt = -z/tau_rate : 1
    
    basethr : volt
    
    dH1/dt = -H1/tau_th1 : volt
    dH2/dt = -H2/tau_th2 : volt
    theta = basethr + H1 + H2 : volt
    
    a1 : volt
    a2 : volt
    gL : siemens
    network_rate : 1
    neuron_target_rate : 1
    rech : 1
    inhf : 1
    max_theta : volt
    {mu_e}
    {mu_i}
    {sigma_e}
    {sigma_i}
    {rho}
    '''

    # Spike reset rules; `recharge` modifies the spike-triggered increment of x
    if recharge >= 0:
        if reset_potential:
            reset = '''
            H1 += a1
            H2 += a2
            x += (1-q)**8 * rech + (1-rech)
            q = 1
            v = EL
            '''
        else:
            reset = '''
            H1 += a1
            H2 += a2
            x += (1-q)**8 * rech + (1-rech)
            q = 1
            '''
    else:
        if reset_potential:
            reset = '''
            H1 += a1
            H2 += a2
            x += (1-q)**8 * rech + 1
            q = 1
            v = EL
            '''
        else:
            reset = '''
            H1 += a1
            H2 += a2
            x += (1-q)**8 * rech + 1
            q = 1
            '''

    if plasticity == 'threshold':
        # threshold adaptation toward target rate
        reset_exc = reset + '''
        basethr = basethr + learning_rate*(z-target_rate)*mV
        z += second/tau_rate
        '''
    else:
        reset_exc = reset

    # Instantiate neuron groups
    eqs = Equations(eqs_str)
    updater = 'euler' if isolate is not None else 'exponential_euler'

    G_exc = NeuronGroup(
        N_exc,
        eqs,
        threshold='v > clip(theta, -100*mV, max_theta)',
        reset=reset_exc,
        method=updater,
        refractory=refrac,
    )
    G_inh = NeuronGroup(
        N_inh,
        eqs,
        threshold='v > clip(theta, -100*mV, max_theta)',
        reset=reset,
        method=updater,
        refractory=refrac,
    )

    # Initial thresholds; `max_theta` can cap runaway thresholds if desired
    G_exc.max_theta = 0 * mV
    G_inh.max_theta = 0 * mV

    # Target rates used by some rules; inhibitory target is fixed here
    G_exc.neuron_target_rate = target_rate
    G_inh.neuron_target_rate = 10

    # Recharge controls (see reset rules)
    G_exc.rech = recharge
    G_inh.rech = 0

    # Fast (a1) and slow (a2) threshold increments per spike
    if alpha1 is True:
        G_exc.a1 = (exc_alpha * 0.25 + 2) * mV  # reproducible because exc_alpha is supplied
        G_inh.a1 = 2 * mV
    else:
        G_exc.a1 = 0
        G_inh.a1 = 0

    G_exc.a2 = alpha2 * mV
    G_inh.a2 = alpha2 * mV

    # Leak conductance and initial voltages
    G_exc.gL = 10 * nS
    G_inh.gL = 10 * nS

    G_exc.v = (np.random.rand(N_exc) * 10) * mV + EL
    G_inh.v = (np.random.rand(N_inh) * 10) * mV + EL

    # External noise parameters in isolation mode
    if isolate is not None:
        for group, ei in zip([G_exc, G_inh], ['exc', 'inh']):
            group.mu_e = isolate['var_stats'].loc[ei, 'mean_e'].values * nS
            group.mu_i = isolate['var_stats'].loc[ei, 'mean_i'].values * nS
            group.sigma_e = isolate['var_stats'].loc[ei, 'std_e'].values * nS
            group.sigma_i = isolate['var_stats'].loc[ei, 'std_i'].values * nS
            group.rho = isolate['var_stats'].loc[ei, 'pearsonr'].values

    # Baseline thresholds
    if thresholds is None:
        G_exc.basethr = omega
    else:
        G_exc.basethr = thresholds * mV
    G_inh.basethr = omega

    # Inhibitory scaling factor (EI only)
    if inhf is None:
        G_exc.inhf = 1
        G_inh.inhf = 1
    else:
        G_exc.inhf = inhf
        G_inh.inhf = 1

    # Connected-network-only components
    if isolate is None:
        # Background drive to both populations
        P1 = PoissonInput(G_exc, 'ge', 1000, background_poisson * Hz, weight=poisson_amplitude * nS)
        P2 = PoissonInput(G_inh, 'ge', 1000, background_poisson * Hz, weight=poisson_amplitude * nS)

        # Population rate integration to support heterosynaptic rules
        eqs_rate_mon = '''
            dr/dt = -r/tau_rate : 1
            '''
        rateunit = NeuronGroup(1, Equations(eqs_rate_mon), method='exponential_euler')
        Sre = Synapses(G_exc, rateunit, model='w : 1', on_pre='r += w')
        Ser = Synapses(rateunit, G_exc, 'network_rate_post = r_pre : 1 (summed)')
        Sre.connect(p=1)
        Ser.connect(p=1)
        Sre.w = (second / tau_rate) / N_exc

        # Inhibitory plasticity (EI) and optional II plasticity
        pre_eqs_inh, post_eqs_inh = ei_plasticity_eqs(plasticity, learning_rate)
        pre_eqs_ii, post_eqs_ii = ei_plasticity_eqs('hebb', learning_rate)

        # Synapse objects per pathway
        model_ei = '''
            w : 1
            alpha : 1
            '''

        See = Synapses(G_exc, G_exc, model='w : 1', on_pre='ge += w*nS', method='exponential_euler')
        Sie = Synapses(G_exc, G_inh, model='w : 1', on_pre='ge += w*nS', method='exponential_euler')
        Sei = Synapses(G_inh, G_exc, model=model_ei, on_pre=pre_eqs_inh, on_post=post_eqs_inh, method='exponential_euler')

        if plast_ii:
            Sii = Synapses(G_inh, G_inh, model=model_ei, on_pre=pre_eqs_ii, on_post=post_eqs_ii, method='exponential_euler')
        else:
            Sii = Synapses(G_inh, G_inh, model='w : 1', on_pre='gi += w*nS', method='exponential_euler')

        # Connect synapses from arrays
        synapses = {'EE': See, 'IE': Sie, 'EI': Sei, 'II': Sii}
        for pre in ['E', 'I']:
            for post in ['E', 'I']:
                label = f'{post}{pre}'
                synapses[label].connect(
                    i=weights[label]['sources'].astype(int),
                    j=weights[label]['targets'].astype(int),
                )

                if (label == 'EI') and (shuffle is True):
                    synapses[label].w = np.random.permutation(weights[label]['weights'])
                else:
                    synapses[label].w = weights[label]['weights']
                
                synapses[label].delay = delays[label] * ms

    # ---------------------------------
    # Recording & HDF5 dataset scaffolding
    # ---------------------------------
    default_units = {
        'v': mV,
        'Isyn': nA,
        'ge': nS,
        'gi': nS,
        'y': 1,
        'theta': mV,
        'H1': mV,
        'H2': mV,
        'x': 1,
    }

    with h5py.File(output_file, "a") as h5f:
        h5f.create_dataset("spikes_exc", (0, 2), maxshape=(None, 2), dtype="float32", chunks=True)
        h5f.create_dataset("spikes_inh", (0, 2), maxshape=(None, 2), dtype="float32", chunks=True)

        if state_variables is not None:
            h5f.create_group('state')
            h5f['state'].create_group('exc')
            h5f['state'].create_group('inh')
            for variable in state_variables:
                h5f['state/exc'].create_dataset(
                    variable,
                    (0, N_exc),
                    maxshape=(None, N_exc),
                    dtype="float32",
                    chunks=True,
                    compression='gzip',
                )
                h5f['state/inh'].create_dataset(
                    variable,
                    (0, N_inh),
                    maxshape=(None, N_inh),
                    dtype="float32",
                    chunks=True,
                    compression='gzip',
                )

    elapsed_time = 0
    elapsed_real_time = 0

    num_chunks = int(np.ceil(simulation_time / chunk_size))

    # Optional time-varying stimulus to E population
    if stimuli is not None:
        logging.info("Setting up stimulus.")
        stim_dt = 0.1
        rm = get_stim_matrix(stimuli, N_exc, simulation_time, dt=stim_dt) * 10
        ta = TimedArray(rm.T * kHz, dt=stim_dt * second)
        G_ext = PoissonGroup(N_exc, rates='ta(t,i)')
        Syn_ext = Synapses(G_ext, G_exc, model='w : 1', on_pre='ge += w*nS')
        Syn_ext.connect(i='j')
        Syn_ext.w = poisson_amplitude
    else:
        logging.info("No stimulus specified.")

    net = Network(collect())

    logging.info(
        f"Starting network simulation for {simulation_time}s in {num_chunks} chunks of {chunk_size}s each."
    )

    for ii in range(num_chunks):
        start_time = time.time()

        # Allocate monitors per chunk to keep memory bounded
        spikes_exc_mon = SpikeMonitor(G_exc)
        spikes_inh_mon = SpikeMonitor(G_inh)
        net.add(spikes_exc_mon, spikes_inh_mon)

        if state_variables is not None:
            state_exc_mon = StateMonitor(G_exc[:], state_variables, record=True, dt=1 * ms)
            state_inh_mon = StateMonitor(G_inh[:], state_variables, record=True, dt=1 * ms)
            net.add(state_exc_mon, state_inh_mon)

        net.run(chunk_size * second)

        elapsed_time += chunk_size

        # Progress + ETA logging
        chunk_real_time = time.time() - start_time
        elapsed_real_time += chunk_real_time
        per_chunk_real_time = elapsed_real_time / (ii + 1)
        remaining_chunks = num_chunks - (ii + 1)
        remaining_real_time = remaining_chunks * chunk_real_time

        logging.info(
            f"Chunk {ii+1} completed in {seconds_to_hms(chunk_real_time)}. "
            f"Elapsed time: {seconds_to_hms(elapsed_real_time)}. Estimated remaining: {seconds_to_hms(remaining_real_time)}. {memory_usage()}"
        )

        # Convert spike trains to (neuron_i, time_s)
        exc_spikes = np.column_stack((spikes_exc_mon.i, spikes_exc_mon.t / second))
        inh_spikes = np.column_stack((spikes_inh_mon.i, spikes_inh_mon.t / second))

        # Compute coarse mean rates per chunk for quick health-check logging
        _, sc_exc = get_spike_counts(
            exc_spikes[:, 0], exc_spikes[:, 1] - elapsed_time + chunk_size, t_max=chunk_size, N=N_exc, dt=1
        )
        _, sc_inh = get_spike_counts(
            inh_spikes[:, 0], inh_spikes[:, 1] - elapsed_time + chunk_size, t_max=chunk_size, N=N_inh, dt=1
        )

        mean_rate_exc = sc_exc.mean()
        mean_rate_inh = sc_inh.mean()
        rate_std_exc = sc_exc.mean(axis=1).std(axis=0)
        rate_std_inh = sc_inh.mean(axis=1).std(axis=0)

        logging.info(
            f"Excitatory neurons firing rate during chunk: ({mean_rate_exc} +/- {rate_std_exc})Hz"
        )
        logging.info(
            f"Inhibitory neurons firing rate during chunk: ({mean_rate_inh} +/- {rate_std_inh})Hz"
        )

        # Append this chunk to HDF5
        with h5py.File(output_file, "a") as h5f:
            h5f.attrs["simulation_time"] = elapsed_time

            h5f["spikes_exc"].resize((h5f["spikes_exc"].shape[0] + exc_spikes.shape[0]), axis=0)
            h5f["spikes_inh"].resize((h5f["spikes_inh"].shape[0] + inh_spikes.shape[0]), axis=0)

            h5f["spikes_exc"][-exc_spikes.shape[0] :] = exc_spikes
            h5f["spikes_inh"][-inh_spikes.shape[0] :] = inh_spikes

            if state_variables is not None:
                for variable in state_variables:
                    variable_exc_data = np.array(
                        state_exc_mon.get_states([variable])[variable] / default_units[variable]
                    )
                    variable_inh_data = np.array(
                        state_inh_mon.get_states([variable])[variable] / default_units[variable]
                    )

                    h5f[f"state/exc/{variable}"].resize(
                        (h5f[f"state/exc/{variable}"].shape[0] + variable_exc_data.shape[0]), axis=0
                    )
                    h5f[f"state/inh/{variable}"].resize(
                        (h5f[f"state/inh/{variable}"].shape[0] + variable_inh_data.shape[0]), axis=0
                    )

                    h5f[f"state/exc/{variable}"][-variable_exc_data.shape[0] :] = variable_exc_data
                    h5f[f"state/inh/{variable}"][-variable_inh_data.shape[0] :] = variable_inh_data

            if save_weights:
                # Save EI weights under /connectivity/weights/EI/weights (overwrite safely)
                grp = h5f.require_group("connectivity/weights/EI")
                if "weights" in grp:
                    del grp["weights"]
                grp.create_dataset("weights", data=Sei.w[:], compression="gzip")

        # Clean up per-chunk monitors to keep memory bounded
        net.remove(spikes_exc_mon, spikes_inh_mon)
        del spikes_exc_mon
        del spikes_inh_mon

        if state_variables is not None:
            net.remove(state_exc_mon, state_inh_mon)
            del state_exc_mon
            del state_inh_mon

        gc.collect()