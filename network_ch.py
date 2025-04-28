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
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return f"[MEMORY] RSS: {mem_info.rss / 1e6:.2f} MB, VMS: {mem_info.vms / 1e6:.2f} MB"

def seconds_to_hms(seconds):
    return time.strftime("%H:%M:%S", time.gmtime(seconds))

def get_stim_matrix(stimuli, N_exc, simulation_time, dt=0.1):
    time_arr = np.arange(0, simulation_time+1e-5, dt)

    stim_matrix = np.zeros((N_exc, len(time_arr)), dtype=float)

    for start, end, neurons in stimuli:
        for ix in neurons:
            mask = (time_arr >= start) & (time_arr < end)
            stim_matrix[ix][mask] = 1.

    return stim_matrix

def ei_plasticity_eqs(plasticity, learning_rate):
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
                    w = clip(w+(x_post-alpha)*learning_rate*20/(tau_stdp/ms), 0, 100)'''  # factor 20/tau_stdp normalizes learning rate

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
            raise ValueError(f"plasticity can be 'threshold', 'rate', 'hebb', or 'idip'. Received '{plasticity}' instead.")
    else:
            pre_eqs_inh = '''
                    gi += w*nS
            '''
            post_eqs_inh = None
    
    return pre_eqs_inh, post_eqs_inh

def run_network(weights, exc_alpha, delays, N_exc, N_inh, alpha1, alpha2, reset_potential,
                target_rate, plasticity, background_poisson, poisson_amplitude, output_file,
                simulation_time, learning_rate, state_variables=None, stimuli=None,
                thresholds=None, seed_num=42, tau_stdp_ms=20, recharge=0, save_weights=False,
                isolate=None, chunk_size=None, plast_ii=False):
    """
    Simulates a spiking neural network with customizable plasticity rules and input stimuli.

    This function can simulate neurons in both a **connected network state** and an **isolated state** 
    (where neurons are disconnected from each other but still receive external noise and perturbations). 
    The simulation results, including spikes, state variables, and synaptic weights, are stored in an HDF5 (.h5) file.

    Parameters
    ----------
    weights : dict
        A dictionary containing synaptic connectivity data for different neuron types.
        Each key represents a synaptic connection type and maps to a dictionary with:
        - `'sources'` (ndarray): Pre-synaptic neuron indices.
        - `'targets'` (ndarray): Post-synaptic neuron indices.
        - `'weights'` (ndarray): Synaptic weight values (in nanosiemens).
        
        The keys in `weights` should be:
        - `'EE'` : Excitatory → Excitatory
        - `'IE'` : Excitatory → Inhibitory
        - `'EI'` : Inhibitory → Excitatory
        - `'II'` : Inhibitory → Inhibitory
        
        Example structure:
        ```python
        weights = {
            'EE': {'sources': np.array([...]), 'targets': np.array([...]), 'weights': np.array([...])},
            'IE': {'sources': np.array([...]), 'targets': np.array([...]), 'weights': np.array([...])},
            'EI': {'sources': np.array([...]), 'targets': np.array([...]), 'weights': np.array([...])},
            'II': {'sources': np.array([...]), 'targets': np.array([...]), 'weights': np.array([...])},
        }
        ```
    exc_alpha : ndarray
        Excitatory adaptation parameter (`alpha1`) for each excitatory neuron, shape (N_exc,).
    delays : dict
        A dictionary containing synaptic delays (in milliseconds) for different connection types.
        The keys must match the `weights` dictionary:
        - `'EE'`: Excitatory → Excitatory
        - `'IE'`: Excitatory → Inhibitory
        - `'EI'`: Inhibitory → Excitatory
        - `'II'`: Inhibitory → Inhibitory

        Example:
        ```python
        delays = {
            'EE': np.array([...]),
            'IE': np.array([...]),
            'EI': np.array([...]),
            'II': np.array([...]),
        }
        ```
    N_exc : int
        Number of excitatory neurons.
    N_inh : int
        Number of inhibitory neurons.
    alpha1 : bool, default=True
        Enables or disables short timescale adaptation for excitatory neurons.
    alpha2 : float, default=2
        MAT parameter for slow adaptation of neurons (in mV).
    reset_potential : bool, default=False
        If `True`, resets the membrane potential to EL after a spike.
    target_rate : float or ndarray
        Target firing rate in Hz. Can be:
        - A scalar applied to all neurons.
        - An array of shape (N_exc,) specifying individual target rates.
    plasticity : str
        Synaptic plasticity rule:
        - `'hebb'` : Hebbian learning.
        - `'rate'` : Heterosynaptic plasticity to match target rates.
        - `'idip'` : Balances excitatory input for homeostasis.
        - `'threshold'` : Adjusts neuron thresholds to achieve target rate.
    background_poisson : float
        Background Poisson input rate (in kHz) applied to all neurons.
    poisson_amplitude : float
        Synaptic weight (in nS) of each background Poisson input spike.
    output_file : str
        Path to the HDF5 (.h5) file where the simulation results will be stored.
    simulation_time : float
        Total simulation duration (in seconds).
    learning_rate : float
        Learning rate for inhibitory synaptic plasticity.
    state_variables : list of str, optional (default: None)
        List of neuron state variables to record.
    stimuli : list of tuples, optional (default: None)
        Each tuple represents a stimulus and follows the format: 
        `(stimulus start, stimulus end, subset of stimulated neurons)`.
    thresholds : ndarray, optional (default: None)
        Initial threshold values for excitatory neurons (if `plasticity='threshold'`).
    seed_num : int, default=42
        Random seed for reproducibility.
    tau_stdp_ms : float, default=20
        STDP time constant in milliseconds.
    recharge : float, default=0
        Affects the calculation of x for excitatory neurons. Positive value leads to ignoring bursts, negative value leads to coincidence detection.
    save_weights : bool, default=False
        If `True`, saves final synaptic weights to the `.h5` file.
    isolate : dict, optional (default: None)
        If provided, simulates neurons in an **isolated state** (disconnected from the network).
        The dictionary must contain:
        - `'exc_stim'` (bool): Whether to apply perturbative input to excitatory neurons.
        - `'inh_stim'` (bool): Whether to apply perturbative input to inhibitory neurons.
        - `'stim_count'` (int): Number of stimulus repetitions.
        - `'var_stats'` (DataFrame): Statistical parameters for external noise (mean, std, correlations).
    chunk_size : float, optional (default: None)
        If provided, runs the simulation in chunks of `chunk_size` seconds to avoid memory overflow.
        If `None`, defaults to 10 seconds unless `isolate` mode is used, in which case it adapts to stimulus settings.

    Returns
    -------
    None
        The function does not return a dictionary of results but saves all relevant data in `output_file`.

    Notes
    -----
    - If `isolate` is provided, neurons **do not** receive synaptic input from the network, but instead receive **external noise**.
    - The function **appends to the HDF5 file** if it already exists, avoiding data loss.
    - Weights are stored in a hierarchical format inside the `"weights"` group.
    - If `state_variables` is provided, their data is recorded separately for excitatory and inhibitory populations.
    - If `save_weights=True`, synaptic weights are stored at the end of the simulation.

    Example
    -------
    **Standard network simulation**
    ```python
    run_network(weights=weights_dict, exc_alpha=alpha, delays=delays_dict,
                N_exc=8000, N_inh=2000, alpha1=True, alpha2=2, reset_potential=False,
                target_rate=5.0, plasticity='hebb', background_poisson=1.5, poisson_amplitude=0.3,
                output_file="simulation_results.h5", simulation_time=100, learning_rate=0.001,
                state_variables=['v', 'ge', 'gi'], save_weights=True)
    ```
    """

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s")

    meta_eta = 0

    np.random.seed(seed_num)

    defaultclock.dt = 0.1*ms  # set time step to 0.1ms

    # Neural model parameters
    # _______________________
    C = 200*pF
    EL = -80*mV
    refrac = 2*ms

    # MAT threshold parameters
    omega = -55*mV
    tau_th1 = 10*ms
    tau_th2 = 200*ms

    # synaptic parameters
    taue = 6*ms  # excitatory synapse time constant
    taui = 6*ms  # inhibitory synapse time constant
    Ee = 0*mV    # excitatory reversal potential
    Ei = -80*mV  # inhibitory reversal potential

    tau_stdp = tau_stdp_ms*ms  # STDP time constant
    tau_bdec = 100*ms
    tau_bmon = 20*second
    tidip = 160*ms
    tau_rate = 10*second  # set very long rate integration for smooth rate estimate
    tau_coincidence = 150*ms
    # ________________________

    if isolate is not None:
        Isyn = 'Isyn = -(ge + gext(t))*(v-Ee)-(gi + gext_inh(t))*(v-Ei) : amp'
        dgedt = 'dge/dt = (mu_e - ge) / taue + sigma_e * sqrt(2 / taue) * xi_e : siemens'
        dgidt = 'dgi/dt = (mu_i - gi) / taui + sigma_i * sqrt(2 / taui) * (rho*xi_e+sqrt(1-rho**2)*xi_i) : siemens'
        mu_e = 'mu_e : siemens'
        mu_i = 'mu_i : siemens'
        sigma_e = 'sigma_e : siemens'
        sigma_i = 'sigma_i : siemens'
        rho = 'rho : 1'
        # Perturbative input
        #________________________

        stim_steps = 5
        step_time = 0.1

        pair = np.array([1]+[0]*(stim_steps-1))

        input_list = np.linspace(-0.35, 0.35, 8)
        init_steps = stim_steps * len(input_list)

        input_arr_exc = np.zeros(init_steps)
        input_arr_inh = np.zeros(init_steps)

        for ii in range(isolate['stim_count']):
            for inp in input_list:
                if isolate['exc_stim']:
                    input_arr_exc = np.append(input_arr_exc, pair*inp)
                    input_arr_inh = np.append(input_arr_inh, np.zeros(stim_steps))

            for inp in input_list:
                if isolate['inh_stim']:
                    input_arr_inh = np.append(input_arr_inh, pair*inp*2)
                    input_arr_exc = np.append(input_arr_exc, np.zeros(stim_steps))

        simulation_time = len(input_arr_exc) * step_time
        
        if chunk_size is None:
            chunk_size = init_steps * step_time

        gext = TimedArray(input_arr_exc * nS, dt=step_time*second)
        gext_inh = TimedArray(input_arr_inh * nS, dt=step_time*second)
        # _________________________________
    else:
        Isyn = 'Isyn = -(ge)*(v-Ee)-(gi)*(v-Ei) : amp'
        dgedt = 'dge/dt = -ge/taue : siemens'
        dgidt = 'dgi/dt = -gi/taue : siemens'
        mu_e = ''
        mu_i = ''
        sigma_e = ''
        sigma_i = ''
        rho = ''

        if chunk_size is None:
            chunk_size = 10

    # Neural model equations

    eqs_str = f'''
    dv/dt = (-gL*(v-EL) + Isyn) / C : volt (unless refractory)
    {Isyn}
    
    dx/dt = -x/tau_stdp : 1
    dq/dt = -q/tau_coincidence : 1
    db_dec/dt = -b_dec/tau_bdec : 1
    db_mon/dt = -b_mon/tau_bmon : 1
    
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
    {mu_e}
    {mu_i}
    {sigma_e}
    {sigma_i}
    {rho}
    '''


    # Define reset rules
    if recharge >= 0:
        if reset_potential:
            reset = '''
            H1 += a1
            H2 += a2
            b_dec += 1
            x += (1-q)**8 * rech + (1-rech)
            q = 1
            v = EL
            '''
        else:
            reset = '''
            H1 += a1
            H2 += a2
            b_dec += 1
            x += (1-q)**8 * rech + (1-rech)
            q = 1
            '''
    else:
        if reset_potential:
            reset = '''
            H1 += a1
            H2 += a2
            b_dec += 1
            x += (1-q)**8 * rech + 1
            q = 1
            v = EL
            '''
        else:
            reset = '''
            H1 += a1
            H2 += a2
            b_dec += 1
            x += (1-q)**8 * rech + 1
            q = 1
            '''

    if plasticity == 'threshold':
        reset_exc = reset + '''
        basethr = basethr + learning_rate*(z-target_rate)*mV
        z += second/tau_rate
        '''
    else:
        reset_exc = reset

    # Initiate neurons
    # ________________

    eqs = Equations(eqs_str)

    if isolate is not None:
        updater = 'euler'
    else:
        updater = 'exponential_euler'

    G_exc = NeuronGroup(N_exc, eqs, threshold='v > theta', reset=reset_exc, method=updater, refractory=refrac,
                        events={'burst': 'b_dec > 3'})
    G_inh = NeuronGroup(N_inh, eqs, threshold='v > theta', reset=reset, method=updater, refractory=refrac)

    G_exc.neuron_target_rate = target_rate
    G_inh.neuron_target_rate = 10

    G_exc.rech = recharge
    G_inh.rech = 0

    target_b_mon = 0.1*Hz*tau_bmon*2

    G_exc.run_on_event('burst', """
        b_dec =  0
        b_mon += 1
        neuron_target_rate = clip(neuron_target_rate+(target_b_mon - b_mon)*meta_eta, 0, 100)
    """)

    if alpha1 is True:
        G_exc.a1 = (exc_alpha*0.25 + 2)*mV  # exc_alpha needs to be supplied to ensure reproducibility
        G_inh.a1 = 2*mV
    else:
        G_exc.a1 = 0
        G_inh.a1 = 0

    G_exc.a2 = alpha2*mV

    # G_inh.a1 = 3*mV
    G_inh.a2 = alpha2*mV

    G_exc.gL = 10*nS
    G_inh.gL = 10*nS
    # G_inh.gL = 20*nS

    G_exc.v = (np.random.rand(N_exc)*10)*mV + EL
    G_inh.v = (np.random.rand(N_inh)*10)*mV + EL

    # G_exc.z = target_rate

    if isolate is not None:
        for group, ei in zip([G_exc, G_inh], ['exc','inh']):
            group.mu_e = isolate['var_stats'].loc[ei,'mean_e'].values * nS
            group.mu_i = isolate['var_stats'].loc[ei,'mean_i'].values * nS
            group.sigma_e = isolate['var_stats'].loc[ei,'std_e'].values * nS
            group.sigma_i = isolate['var_stats'].loc[ei,'std_i'].values * nS
            group.rho = isolate['var_stats'].loc[ei,'pearsonr'].values

    if thresholds is None:
        G_exc.basethr = omega
    else:
        G_exc.basethr = thresholds*mV

    G_inh.basethr = omega
    # ________________


    if isolate is None:
        # Initiate background input
        # _________________________________
        P1 = PoissonInput(G_exc, 'ge', 1000, background_poisson*Hz, weight=poisson_amplitude*nS)
        P2 = PoissonInput(G_inh, 'ge', 1000, background_poisson*Hz, weight=poisson_amplitude*nS)
        # _________________________________

        # ___________________________

        # Rate calculation
        # _________________________
        # In order to use heterosynaptic plasticity, a dummy unit is created, receiving input from all neurons
        # and integrating it as a rate. The rate is then passed on the neurons through "gap junctions".

        eqs_rate_mon = '''
            dr/dt = -r/tau_rate : 1
            '''

        rateunit = NeuronGroup(1, Equations(eqs_rate_mon), method='exponential_euler')
        Sre = Synapses(G_exc, rateunit, model='w : 1', on_pre='r += w')
        Ser = Synapses(rateunit, G_exc, 'network_rate_post = r_pre : 1 (summed)')

        Sre.connect(p=1)
        Ser.connect(p=1)

        Sre.w = (second/tau_rate) / N_exc

        # _________________________

        # Define E<-I plasticity equations
        # ___________________________________

        pre_eqs_inh, post_eqs_inh = ei_plasticity_eqs(plasticity, learning_rate)
        pre_eqs_ii, post_eqs_ii = ei_plasticity_eqs('hebb', learning_rate)

        # Initiate synapses
        # _________________________

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

        # connect synapses
        synapses = {
            'EE': See,
            'IE': Sie,
            'EI': Sei,
            'II': Sii
        }

        for pre in ['E','I']:
            for post in ['E','I']:
                label = f'{post}{pre}'
                synapses[label].connect(i=weights[label]['sources'].astype(int), j=weights[label]['targets'].astype(int))
                synapses[label].w = weights[label]['weights']
                synapses[label].delay = delays[label] * ms

    # Run simulation
    # __________________________

    default_units = {
        'v': mV,
        'Isyn': nA,
        'ge': nS,
        'gi': nS,
        'y': 1,
        'theta': mV,
        'x': 1,
        'b_dec': 1,
        'b_mon': 1
    }


    with h5py.File(output_file, "a") as h5f:
        h5f.create_dataset("spikes_exc", (0, 2), maxshape=(None, 2), dtype="float32", chunks=True)
        h5f.create_dataset("spikes_inh", (0, 2), maxshape=(None, 2), dtype="float32", chunks=True)
        
        if state_variables is not None:
            h5f.create_group('state')
            h5f['state'].create_group('exc')
            h5f['state'].create_group('inh')
            for variable in state_variables:
                h5f['state/exc'].create_dataset(variable, (0,N_exc), maxshape=(None,N_exc), dtype="float32", chunks=True, compression='gzip')
                h5f['state/inh'].create_dataset(variable, (0,N_inh), maxshape=(None,N_inh), dtype="float32", chunks=True, compression='gzip')

    elapsed_time = 0
    elapsed_real_time = 0

    num_chunks = int(np.ceil(simulation_time / chunk_size))

    

    # Activate patterns
    # ___________________________

    if stimuli is not None:
        logging.info(f"Setting up stimulus.")
        stim_dt = 0.1
        rm = get_stim_matrix(stimuli, N_exc, simulation_time, dt=stim_dt) * 10
        ta = TimedArray(rm.T*kHz, dt=stim_dt*second)
        G_ext = PoissonGroup(N_exc, rates='ta(t,i)')
        Syn_ext = Synapses(G_ext, G_exc, model='w : 1', on_pre='ge += w*nS')
        Syn_ext.connect(i='j')
        Syn_ext.w = poisson_amplitude
    else:
        logging.info(f"No stimulus specified.")

    net = Network(collect())

    logging.info(f"Starting network simulation for {simulation_time}s in {num_chunks} chunks of {chunk_size}s each.")

    for ii in range(num_chunks):
        start_time = time.time()

        spikes_exc_mon = SpikeMonitor(G_exc)
        spikes_inh_mon = SpikeMonitor(G_inh)
        net.add(spikes_exc_mon, spikes_inh_mon)
        
        if state_variables is not None:
            state_exc_mon = StateMonitor(G_exc[:], state_variables, record=True, dt=1*ms)
            state_inh_mon = StateMonitor(G_inh[:], state_variables, record=True, dt=1*ms)
            net.add(state_exc_mon, state_inh_mon)


        net.run(chunk_size*second)

        elapsed_time += chunk_size
        
        chunk_real_time = time.time() - start_time
        elapsed_real_time += chunk_real_time
        per_chunk_real_time = elapsed_real_time / (ii+1)
        remaining_chunks = num_chunks - (ii+1)
        remaining_real_time = remaining_chunks * chunk_real_time

        logging.info(
            f"Chunk {ii+1} completed in {seconds_to_hms(chunk_real_time)}. "
            f"Elapsed time: {seconds_to_hms(elapsed_real_time)}. Estimated remaining: {seconds_to_hms(remaining_real_time)}. {memory_usage()}"
        )

        exc_spikes = np.column_stack((spikes_exc_mon.i, spikes_exc_mon.t / second))
        inh_spikes = np.column_stack((spikes_inh_mon.i, spikes_inh_mon.t / second))

        _, sc_exc = get_spike_counts(exc_spikes[:,0], exc_spikes[:,1]-elapsed_time+chunk_size, t_max=chunk_size, N=N_exc, dt=1)
        _, sc_inh = get_spike_counts(inh_spikes[:,0], inh_spikes[:,1]-elapsed_time+chunk_size, t_max=chunk_size, N=N_inh, dt=1)

        mean_rate_exc = sc_exc.mean()
        mean_rate_inh = sc_inh.mean()
        rate_std_exc = sc_exc.mean(axis=1).std(axis=0)
        rate_std_inh = sc_inh.mean(axis=1).std(axis=0)

        logging.info(f"Excitatory neurons firing rate during chunk: ({mean_rate_exc} +/- {rate_std_exc})Hz")
        logging.info(f"Inhibitory neurons firing rate during chunk: ({mean_rate_inh} +/- {rate_std_inh})Hz")

        # if recharge != 0 and learning_rate > 0:
        #     rate_diff = target_rate - mean_rate_exc
        #     G_exc.neuron_target_rate = np.clip(G_exc.neuron_target_rate + rate_diff*0.1, a_min=0, a_max=None)
        #     logging.info(f"Target rate adjusted to {np.mean(G_exc.neuron_target_rate):.2f}")

        # Append to HDF5
        with h5py.File(output_file, "a") as h5f:
            h5f.attrs["simulation_time"] = elapsed_time

            h5f["spikes_exc"].resize((h5f["spikes_exc"].shape[0] + exc_spikes.shape[0]), axis=0)
            h5f["spikes_inh"].resize((h5f["spikes_inh"].shape[0] + inh_spikes.shape[0]), axis=0)

            h5f["spikes_exc"][-exc_spikes.shape[0]:] = exc_spikes
            h5f["spikes_inh"][-inh_spikes.shape[0]:] = inh_spikes

            if state_variables is not None:
                for variable in state_variables:
                    variable_exc_data = np.array(state_exc_mon.get_states([variable])[variable] / default_units[variable])
                    variable_inh_data = np.array(state_inh_mon.get_states([variable])[variable] / default_units[variable])

                    h5f[f"state/exc/{variable}"].resize((h5f[f"state/exc/{variable}"].shape[0] + variable_exc_data.shape[0]), axis=0)
                    h5f[f"state/inh/{variable}"].resize((h5f[f"state/inh/{variable}"].shape[0] + variable_inh_data.shape[0]), axis=0)

                    h5f[f"state/exc/{variable}"][-variable_exc_data.shape[0]:] = variable_exc_data
                    h5f[f"state/inh/{variable}"][-variable_inh_data.shape[0]:] = variable_inh_data

                del state_exc_mon
                del state_inh_mon

            if save_weights:
            # with h5py.File(output_file, "a") as h5f:  # Open file in append mode
                grp = h5f.require_group("connectivity/weights/EI")

                if "weights" in grp:
                    del grp["weights"]  # Delete existing dataset before overwriting
                grp.create_dataset("weights", data=Sei.w[:], compression="gzip")  # Optional compression

        net.remove(spikes_exc_mon, spikes_inh_mon)
        del spikes_exc_mon
        del spikes_inh_mon

        # if stimuli is not None:
        #     net.remove(G_ext, Syn_ext, ta)
        #     del G_ext
        #     del Syn_ext
        #     del ta

        gc.collect()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-f', '--input', type=str)
    parser.add_argument('-t', '--time', type=float, default=10)
    parser.add_argument('--rate_file', type=str)
    parser.add_argument('--thr_file', type=str)
    parser.add_argument('--target_rate', type=float, default=3.)
    parser.add_argument('--trstd', type=float, default=0.)
    parser.add_argument('--trdistr', type=str, default='lognorm')
    parser.add_argument('--eta', type=float, default=1e-3)
    parser.add_argument('--plasticity', type=str, default='hebb')
    parser.add_argument('--eiplast', action='store_true')
    parser.add_argument('--eeplast', action='store_true')
    parser.add_argument('--bcg_rate', type=float, default=2.)
    parser.add_argument('--bcg_ampl', type=float, default=0.3)
    parser.add_argument('-o', '--output', type=str)
    parser.add_argument('--matrix', type=str)
    parser.add_argument('--reset', action='store_true')
    parser.add_argument('--a1_off', action='store_true')
    parser.add_argument('--alpha2', type=float, default=2)
    parser.add_argument('--record', type=str, nargs='*')
    parser.add_argument('--stimulus', type=str)
    parser.add_argument('--stimfrac', type=float, default=10)
    parser.add_argument('--randstim', action='store_true')
    parser.add_argument('--tau_stdp', type=float, default=20)
    parser.add_argument('--meta_eta', type=float, default=0)
    parser.add_argument('--recharge', type=float, default=0)

    args = parser.parse_args()

    with open(args.input, 'rb') as file:
        Z, N_exc, patterns, exc_alpha, delays, _ = pickle.load(file)

    if args.rate_file is not None:
        with open(args.rate_file, 'rb') as file:
            rates = pickle.load(file)
            target_rate = np.array(rates)
    else:
        target_rate = args.target_rate

    if args.thr_file is not None:
        with open(args.thr_file, 'rb') as file:
            thresholds = pickle.load(file)
    else:
        thresholds = None

    if args.stimulus is not None:
        stimulus_tuples = load_stim_file(args.stimulus, patterns, randstim=args.randstim, fraction=args.stimfrac)
    else:
        stimulus_tuples = None

    if args.a1_off:
        alpha1 = False
    else:
        alpha1 = True

    print(args.record)

    simulation_params = dict(
        Z=Z,
        exc_alpha=exc_alpha,
        delays=delays,
        target_rate=target_rate,
        plasticity=args.plasticity,
        background_poisson=args.bcg_rate,
        poisson_amplitude=args.bcg_ampl,
        simulation_time=args.time,
        learning_rate=args.eta,
        N_exc=N_exc,
        target_rate_std=args.trstd,
        target_distr=args.trdistr,
        state_variables=args.record,
        plast_ie=args.eiplast,
        plast_ee=args.eeplast,
        reset_potential=args.reset,
        alpha1=alpha1,
        alpha2=args.alpha2,
        stimuli=stimulus_tuples,
        thresholds=thresholds,
        tau_stdp_ms=args.tau_stdp,
        meta_eta=args.meta_eta,
        recharge=args.recharge
    )

    run_n_save(simulation_params, args, matrix_file=args.input, output=args.output, matrix_out=args.matrix)


    # Facilitating synapses