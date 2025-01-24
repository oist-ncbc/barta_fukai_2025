from brian2 import *
import numpy as np
import argparse
import pickle
import pandas as pd

from network import update_matrix


def get_stim_matrix(stimuli, N_exc, simulation_time, dt=0.1):
    time_arr = np.arange(0, simulation_time + 1e-5, dt)

    stim_matrix = np.zeros((N_exc, len(time_arr)), dtype=float)

    for start, end, neurons in stimuli:
        for ix in neurons:
            mask = (time_arr >= start) & (time_arr < end)
            stim_matrix[ix][mask] = 1.

    return stim_matrix


def run_network(Z, exc_alpha, delays, target_rate, plasticity, background_poisson, poisson_amplitude,
                simulation_time, learning_rate, varstats_e, varstats_i, plast_ie=False, plast_ee=False, report=True, state_variables=None,
                state_subset=0.1,
                stimuli=None, N_exc=8000, alpha1=True, alpha2=2, reset_potential=False, target_rate_std=0,
                target_distr='lognorm',
                thresholds=None, seed_num=42, tau_stdp_ms=20, meta_eta=0):
    """
    Run network of neurons with or without inhibitory plasticity.
    :param Z: connection matrix with weights in nS
    :param exc_alpha: vector of length N_exc containing the alpha1 parameters for excitatory neurons
    :param delays: tuple of 4 vectors with synaptic delays (delays_ee, delays_ie, delays_ii, delays_ei)
    :param target_rate: can be either scalar for all neurons, or an array of length N_exc
    :param plasticity: if 'hebb', Hebbian learning is used. if 'rate', heterosynaptic learning rule is used
        to achieve the desired firing rate. if 'idip', learning rule to equalize mean excitatory input to a neuron
        is used. if 'threshold', the threshold of each neuron is modified until desired firing rate is achieved
    :param background_poisson: background input rate in kHz. received by both excitatory and inhibitory populations
    :param poisson_amplitude: amplitude of background input spikes in nS
    :param simulation_time: total simulation time in seconds
    :param learning_rate: inhibitory plasticity learning rate
    :param plast_ie: if exc to inh plasticity should be turned on
    :param plast_ee: if exc to exc plasticity should be turned on
    :param report: whether progress should be printed out during simulation
    :param state_variables: either None (default), or a list of variables that should be recorded
    :param state_subset: if float, given fraction of neurons is selected by random.
        if a list of indexes (<N_exc), specified subset is recorded.
    :param stimuli: list of tuples in the format (stimulus start, stimulus end, subset of stimulated neurons)
    :param N_exc: number of excitatory neurons
    :param alpha2: MAT parameter for slow adaptation of neurons
    :param reset_potential: whether membrane potential should be reset to EL when a spike is fired
    :param target_rate_std: standard deviation of target rate. if > 0, each neuron will have a different
        target rate assigned from a log-normal distribution
    :param target_distr: 'lognorm' or 'gamma' supported for distribution of firing rates
    :param seed_num: seed for reproducibility
    :param alpha1: whether short timescale adaptation should be used
    :param chunks: duration of chunks in seconds in which network is simulated. If None (default), network is simulated
        in one go
    """
    np.random.seed(seed_num)

    defaultclock.dt = 0.1 * ms  # set time step to 0.1ms

    N_inh = Z.shape[0] - N_exc  # number of inhibitory neurons

    # Neural model parameters
    # _______________________
    C = 200 * pF
    EL = -80 * mV
    refrac = 2 * ms

    # MAT threshold parameters
    omega = -55 * mV
    tau_th1 = 10 * ms
    tau_th2 = 200 * ms

    # synaptic parameters
    taue = 6 * ms  # excitatory synapse time constant
    taui = 6 * ms  # inhibitory synapse time constant
    Ee = 0 * mV  # excitatory reversal potential
    Ei = -80 * mV  # inhibitory reversal potential

    tau_stdp = tau_stdp_ms * ms  # STDP time constant
    tau_bdec = 100 * ms
    tau_bmon = 20 * second
    tidip = 160 * ms
    tau_rate = 10 * second  # set very long rate integration for smooth rate estimate
    # ________________________

    # Perturbative input
    #________________________
    input_arr_exc = np.zeros(5)
    input_arr_inh = np.zeros(5)
    pair = np.array([1]+[0]*4)

    # input_list = np.arange(0, 1, 0.1) - 0.45
    input_list = np.linspace(-0.35, 0.35, 8)

    for ii in range(500):
        for inp in input_list:
            input_arr_exc = np.append(input_arr_exc, pair*inp)
            input_arr_inh = np.append(input_arr_inh, np.zeros(5))

    for ii in range(500):
        for inp in input_list:
            input_arr_inh = np.append(input_arr_inh, pair*inp)
            input_arr_exc = np.append(input_arr_exc, np.zeros(5))

    gext = TimedArray(input_arr_exc * nS, dt=0.1*second)
    gext_inh = TimedArray(input_arr_inh * nS, dt=0.1*second)

    # Neural model equations

    # dge/dt = -ge/taue : siemens
    # dgi/dt = -gi/taui : siemens

    eqs_str = f'''
    dv/dt = (-gL*(v-EL) + Isyn) / C : volt (unless refractory)
    Isyn = -(ge + gext(t))*(v-Ee)-(gi + gext_inh(t))*(v-Ei) : amp

    dx/dt = -x/tau_stdp : 1

    dge/dt = (mu_e - ge) / taue + sigma_e * sqrt(2 / taue) * xi_e : siemens
    dgi/dt = (mu_i - gi) / taui + sigma_i * sqrt(2 / taui) * (rho*xi_e+sqrt(1-rho**2)*xi_i) : siemens

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
    
    mu_e : siemens
    mu_i : siemens
    sigma_e: siemens
    sigma_i: siemens
    rho : 1
    '''

    # Define reset rules

    if reset_potential:
        reset = '''
        H1 += a1
        H2 += a2
        x += 1
        v = EL
        '''
    else:
        reset = '''
        H1 += a1
        H2 += a2
        x += 1
        '''

    if plasticity == 'threshold':
        reset_exc = reset + '''
        basethr = basethr + clip(learning_rate*(z-target_rate), 0, 1)*mV
        z += second/tau_rate
        '''
    else:
        reset_exc = reset

    # Initiate neurons
    # ________________

    eqs = Equations(eqs_str)

    G_exc = NeuronGroup(N_exc, eqs, threshold='v > theta', reset=reset_exc, method='euler',
                        refractory=refrac)
    G_inh = NeuronGroup(N_inh, eqs, threshold='v > theta', reset=reset, method='euler', refractory=refrac)

    G_exc.neuron_target_rate = target_rate

    G_exc.mu_e = varstats_e['mean_e'].values * nS
    G_exc.mu_i = varstats_e['mean_i'].values * nS
    sig_e = np.sqrt(varstats_e['var_e'].values)
    sig_i = np.sqrt(varstats_e['var_i'].values)
    rho = varstats_e['cov'].values / (sig_e*sig_i)
    G_exc.sigma_e = sig_e * nS
    G_exc.sigma_i = sig_i * nS
    G_exc.rho = rho

    G_inh.mu_e = varstats_i['mean_e'].values * nS
    G_inh.mu_i = varstats_i['mean_i'].values * nS
    sig_e = np.sqrt(varstats_i['var_e'].values)
    sig_i = np.sqrt(varstats_i['var_i'].values)
    rho = varstats_i['cov'].values / (sig_e*sig_i)
    G_inh.sigma_e = sig_e * nS
    G_inh.sigma_i = sig_i * nS
    G_inh.rho = rho

    if alpha1 is True:
        G_exc.a1 = (exc_alpha * 0.25 + 2) * mV  # exc_alpha needs to be supplied to ensure reproducibility
        G_inh.a1 = 2 * mV
    else:
        G_exc.a1 = 0
        G_inh.a1 = 0

    G_exc.a2 = alpha2 * mV

    # G_inh.a1 = 3*mV
    G_inh.a2 = alpha2 * mV

    G_exc.gL = 10 * nS
    G_inh.gL = 10 * nS
    # G_inh.gL = 20*nS

    G_exc.v = (np.random.rand(N_exc) * 10) * mV + EL
    G_inh.v = (np.random.rand(N_inh) * 10) * mV + EL

    G_exc.z = target_rate

    if thresholds is None:
        G_exc.basethr = omega
    else:
        G_exc.basethr = thresholds * mV

    G_inh.basethr = omega
    # ________________

    # # Initiate background input
    # # _________________________________
    # P1 = PoissonInput(G_exc, 'ge', 1000, background_poisson * Hz, weight=poisson_amplitude * nS)
    # P2 = PoissonInput(G_inh, 'ge', 1000, background_poisson * Hz, weight=poisson_amplitude * nS)
    # # _________________________________

    # # Input perturbation
    # # ___________________________
    #
    # input_arr = np.zeros(10)
    # pair = np.array([1]+[0]*9)
    #
    # input_list = np.arange(0, 1, 0.1) - 0.45
    #
    # for inp in input_list:
    #     input_arr = np.append(input_arr, pair*inp)
    #
    # ta = TimedArray(input_arr * kHz, dt=0.1*second)
    # G_ext_exc = PoissonGroup(N_exc, rates='ta(t)')
    # G_ext_inh = PoissonGroup(N_inh, rates='ta(t)')
    #
    # Syn_ext_exc = Synapses(G_ext_exc, G_exc, model='w : 1', on_pre='ge += w*nS')
    # Syn_ext_inh = Synapses(G_ext_inh, G_inh, model='w : 1', on_pre='ge += w*nS')
    #
    # Syn_ext_exc.w = poisson_amplitude
    # Syn_ext_inh.w = poisson_amplitude
    #
    # # ___________________________

    # Run simulation
    # __________________________

    net = Network(collect())

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

    spike_monitors = [SpikeMonitor(G_exc), SpikeMonitor(G_inh)]
    net.add(spike_monitors)

    if state_variables is not None:
        print(state_variables)
        # define subset of neurons with state being recorded
        if type(state_subset) == float:
            n_state = int(state_subset * N_exc)
            subset_ix = np.random.permutation(N_exc)[:n_state]

        state_monitors = [StateMonitor(G_exc[:], state_variables, record=True)]
        net.add(state_monitors)

    if report:
        report_status = 'stderr'
    else:
        report_status = None


    simulation_time = len(input_arr_exc) * 0.1 * second
    net.run(simulation_time, report=report_status)

    # __________________________

    results = {'spikes': {
        'exc': (np.array(spike_monitors[0].i), np.array(spike_monitors[0].t / second)),
        'inh': (np.array(spike_monitors[1].i), np.array(spike_monitors[1].t / second))
    }, 'weights': {}}

    if meta_eta > 0:
        results['target_rates'] = np.array(G_exc.neuron_target_rate)

    if plasticity == 'threshold':
        results['thresholds'] = np.array(G_exc.basethr / mV)

    if state_variables is not None:
        results['state'] = {}

        # import pdb;pdb.set_trace()

        for variable in state_variables:
            variable_full_data = np.array(
                state_monitors[0].get_states([variable])[variable] / default_units[variable])
            print(variable_full_data.mean())
            results['state'][variable] = variable_full_data.reshape((-1, 10, 8000)).mean(axis=1)
            print(results['state'][variable].mean())

    return results

def run_n_save(simulation_params, args, matrix_file):
    results = run_network(**simulation_params)

    results['params'] = vars(args)
    results['simulation_params'] = simulation_params

    if 'thresholds' in results:
        with open('data/thresholds.pkl', 'wb') as file:
            pickle.dump(results['thresholds'], file)

    with open(matrix_file, 'rb') as file:
        Z, N_exc, patterns, exc_alpha, delays, _ = pickle.load(file)

    if args.matrix is not None:
        Z_new, delays_new = update_matrix(Z, N_exc, delays, results['weights'],
                                          plast_ie=simulation_params['plast_ie'],
                                          plast_ee=simulation_params['plast_ee'])

        with open(args.matrix, 'wb') as file:
            savetuple = (Z_new, N_exc, patterns, exc_alpha, delays_new, vars(args))
            pickle.dump(savetuple, file)

    with open(args.output, 'wb') as file:
        pickle.dump(results, file)


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
    parser.add_argument('--vardata_e', type=str)
    parser.add_argument('--vardata_i', type=str)

    args = parser.parse_args()

    vardata_e = pd.read_csv(args.vardata_e, index_col=None)
    vardata_i = pd.read_csv(args.vardata_i, index_col=None)

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
        varstats_e=vardata_e,
        varstats_i=vardata_i,
    )

    run_n_save(simulation_params, args, matrix_file=args.input)