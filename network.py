from brian2 import *
import numpy as np
import argparse
import pickle
import pandas as pd

def get_stim_matrix(stimuli, N_exc, simulation_time, dt=0.1):
    time_arr = np.arange(0, simulation_time+1e-5, dt)

    stim_matrix = np.zeros((N_exc, len(time_arr)), dtype=float)

    for start, end, neurons in stimuli:
        for ix in neurons:
            mask = (time_arr >= start) & (time_arr < end)
            stim_matrix[ix][mask] = 1.

    return stim_matrix

def run_network(Z, exc_alpha, delays, target_rate, plasticity, background_poisson, poisson_amplitude,
                simulation_time, learning_rate, plast_ie=False, plast_ee=False, report=True, state_variables=None, state_subset=0.1,
                stimuli=None, N_exc=8000, alpha1=True, alpha2=2, reset_potential=False, target_rate_std=0, target_distr='lognorm',
                thresholds=None, seed_num=42, tau_stdp_ms=20, meta_eta=0, chunks=None):
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

    defaultclock.dt = 0.1*ms  # set time step to 0.1ms

    N_inh = Z.shape[0] - N_exc  # number of inhibitory neurons

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
    # ________________________

    # Neural model equations

    eqs_str = f'''
    dv/dt = (-gL*(v-EL) + Isyn) / C : volt (unless refractory)
    Isyn = -(ge)*(v-Ee)-(gi)*(v-Ei) : amp
    
    dx/dt = -x/tau_stdp : 1
    db_dec/dt = -b_dec/tau_bdec : 1
    db_mon/dt = -b_mon/tau_bmon : 1
    
    dge/dt = -ge/taue : siemens
    dgi/dt = -gi/taui : siemens
    
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
    '''


    # Define reset rules

    if reset_potential:
        reset = '''
        H1 += a1
        H2 += a2
        b_dec += 1
        x += 1
        v = EL
        '''
    else:
        reset = '''
        H1 += a1
        H2 += a2
        b_dec += 1
        x += 1
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

    G_exc = NeuronGroup(N_exc, eqs, threshold='v > theta', reset=reset_exc, method='exponential_euler', refractory=refrac,
                        events={'burst': 'b_dec > 3'})
    G_inh = NeuronGroup(N_inh, eqs, threshold='v > theta', reset=reset, method='exponential_euler', refractory=refrac)

    G_exc.neuron_target_rate = target_rate

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

    G_exc.z = target_rate

    if thresholds is None:
        G_exc.basethr = omega
    else:
        G_exc.basethr = thresholds*mV

    G_inh.basethr = omega
    # ________________

    # Initiate background input
    # _________________________________
    P1 = PoissonInput(G_exc, 'ge', 1000, background_poisson*Hz, weight=poisson_amplitude*nS)
    P2 = PoissonInput(G_inh, 'ge', 1000, background_poisson*Hz, weight=poisson_amplitude*nS)
    # _________________________________

    # Activate patterns
    # ___________________________

    if stimuli is not None:
        stim_dt = 0.1
        rm = get_stim_matrix(stimuli, N_exc, simulation_time, dt=stim_dt) * 10
        ta = TimedArray(rm.T*kHz, dt=stim_dt*second)
        G_ext = PoissonGroup(N_exc, rates='ta(t,i)')
        Syn_ext = Synapses(G_ext, G_exc, model='w : 1', on_pre='ge += w*nS')
        Syn_ext.connect(i='j')
        Syn_ext.w = poisson_amplitude

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

    eta = learning_rate

    if plasticity == 'rate':
        pre_eqs_inh = '''
                 gi += w*nS
                 w = w + eta * (network_rate-target_rate) * (x_post + 1)'''

        post_eqs_inh = '''
                 w = w + eta * (network_rate-target_rate) * x_pre'''

    elif plasticity == 'hebb':
        pre_eqs_inh = '''
                 gi += w*nS
                 alpha = neuron_target_rate_post*Hz*tau_stdp*2
                 w = clip(w+(x_post-alpha)*eta, 0, 100)'''

        post_eqs_inh = '''
                  w = clip(w+x_pre*eta, 0, 100)'''

    elif plasticity == 'idip':
        pre_eqs_inh = '''
                 gi += w*nS
                 w = clip(w+(y_pre-17)*eta, 0, 100)'''
        post_eqs_inh = None
    elif plasticity == 'threshold':
        pre_eqs_inh = '''
                gi += w*nS
        '''
        post_eqs_inh = None
    else:
        raise ValueError(f"plasticity can be 'threshold', 'rate', 'hebb', or 'idip'. Received '{plasticity}' instead.")

    # ___________________________________

    # Define I<-E plasticity equations
    # ___________________________________

    pre_eqs_ie = '''
             ge += w*nS
             w = clip(w-(x_post)*eta, 0, 100)'''

    post_eqs_ie = '''
              w = clip(w+x_pre*eta, 0, 100)'''
    # ___________________________________

    # ___________________________________

    # Define E<-E plasticity equations
    # ___________________________________

    tauhstas = 450*second

    model_ee = '''
            dw/dt = -w/tauhstas : 1 (event-driven)'''

    pre_eqs_ee = '''
             ge += w*nS
             w = clip(w+(x_post)*eta, 0, 100)'''

    post_eqs_ee = '''
              w = clip(w+x_pre*eta, 0, 100)'''
    # ___________________________________


    # Initiate synapses
    # _________________________
    Sii = Synapses(G_inh, G_inh, model='w : 1', on_pre='gi += w*nS', method='exponential_euler')

    if plast_ee and learning_rate > 0:
        See = Synapses(G_exc, G_exc, model=model_ee, on_pre=pre_eqs_ee, on_post=post_eqs_ee, method='exponential_euler')
    else:
        See = Synapses(G_exc, G_exc, model='w : 1', on_pre='ge += w*nS', method='exponential_euler')

    if plast_ie and learning_rate > 0:
        Sie = Synapses(G_exc, G_inh, model='w : 1', on_pre=pre_eqs_ie, on_post=post_eqs_ie, method='exponential_euler')
    else:
        Sie = Synapses(G_exc, G_inh, model='w : 1', on_pre='ge += w*nS', method='exponential_euler')

    # only employ plasticity if learning is turned on
    model = '''
        w : 1
        alpha : 1
        '''
    if learning_rate > 0:
        Sei = Synapses(G_inh, G_exc, model=model, on_pre=pre_eqs_inh, on_post=post_eqs_inh,
                       method='exponential_euler')
    else:
        Sei = Synapses(G_inh, G_exc, model=model, on_pre='gi += w*nS', method='exponential_euler')

    delays_ee, delays_ie, delays_ii, delays_ei = delays

    # exc to exc
    ZEE = Z[:N_exc,:N_exc].T
    sources, targets = np.nonzero(ZEE)
    weights = ZEE[ZEE != 0]
    See.connect(i=sources, j=targets)
    See.w = weights*0.5
    See.delay = delays_ee*ms

    # exc to inh
    ZIE = Z[N_exc:,:N_exc].T
    sources, targets = np.nonzero(ZIE)
    weights = ZIE[ZIE != 0]
    Sie.connect(i=sources, j=targets)
    Sie.w = weights
    Sie.delay = delays_ie*ms

    # inh to exc
    ZEI = Z[:N_exc,N_exc:].T
    sources, targets = np.nonzero(ZEI)
    weights = ZEI[ZEI != 0]
    Sei.connect(i=sources, j=targets)
    Sei.w = np.abs(weights)
    Sei.delay = delays_ei*ms

    if np.isscalar(target_rate):
        if target_rate_std == 0:
            alpha_val = target_rate*Hz*tau_stdp*2
        else:
            E = target_rate
            Var = target_rate_std**2

            if target_distr == 'lognorm':
                sig = np.sqrt(np.log(Var/(E*E) + 1))
                mu = np.log(E) - sig**2 / 2

                target_rates = np.sort(np.exp(np.random.randn(N_exc)*sig + mu))

            elif target_distr == 'gamma':
                target_rates = np.sort(np.random.gamma(shape=E**2/(Var), scale=Var/E, size=N_exc))

            alpha_val = target_rates*Hz*tau_stdp*2
    else:
        target_rates = np.array(target_rate)
        alpha_val = target_rates*Hz*tau_stdp*2

    alpha_matrix = (ZEI != 0).astype(float) * alpha_val
    Sei.alpha = alpha_matrix[ZEI != 0]  # is this necessary?

    # inh to inh
    ZII = Z[N_exc:,N_exc:].T
    sources, targets = np.nonzero(ZII)
    weights = ZII[ZII != 0]
    Sii.connect(i=sources, j=targets)
    Sii.w = np.abs(weights)*3
    Sii.delay = delays_ii*ms

    # Run simulation
    # __________________________

    net = Network(collect())

    chunks = None
    folder = 'data/'

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

    if chunks is None:
        spike_monitors = [SpikeMonitor(G_exc), SpikeMonitor(G_inh)]
        net.add(spike_monitors)

        if state_variables is not None:
            print(state_variables)
            # define subset of neurons with state being recorded
            if type(state_subset) == float:
                n_state = int(state_subset * N_exc)
                subset_ix = np.random.permutation(N_exc)[:n_state]

            state_monitors = [StateMonitor(G_exc[:], state_variables, record=True), StateMonitor(G_inh[:], state_variables, record=True)]
            net.add(state_monitors)

        if report:
            report_status = 'stderr'
        else:
            report_status = None

        net.run(simulation_time*second, report=report_status)

        # __________________________

        results = {'spikes': {
            'exc': (np.array(spike_monitors[0].i), np.array(spike_monitors[0].t / second)),
            'inh': (np.array(spike_monitors[1].i), np.array(spike_monitors[1].t / second))
        }, 'weights': {}}

        results['weights']['ei'] = np.array(Sei.w)

        if plast_ie:
            results['weights']['ie'] = np.array(Sie.w)

        if plast_ee:
            results['weights']['ee'] = np.array(See.w)

        if target_rate_std != 0:
            results['target_rates'] = target_rates

        if meta_eta > 0:
            results['target_rates'] = np.array(G_exc.neuron_target_rate)

        if plasticity == 'threshold':
            results['thresholds'] = np.array(G_exc.basethr / mV)

        if state_variables is not None:
            results['state_exc'] = {}
            results['state_inh'] = {}

            # import pdb;pdb.set_trace()

            for variable in state_variables:
                for i, ei in enumerate(['exc','inh']):
                    variable_full_data = np.array(state_monitors[i].get_states([variable])[variable] / default_units[variable])
                    print(variable_full_data.shape, variable_full_data.mean())

                    if ei == 'exc':
                        n_neurons = N_exc
                    elif ei == 'inh':
                        n_neurons = N_inh

                    results[f'state_{ei}'][variable] = variable_full_data.reshape((-1,10,n_neurons)).mean(axis=1)
                    print(results[f'state_{ei}'][variable].shape, results[f'state_{ei}'][variable].mean())

        return results
    else:
        time_remaining = simulation_time
        ii = 0
        while time_remaining > 0:
            # the code below is repeated what is above, function made issues
            spike_monitors = [SpikeMonitor(G_exc), SpikeMonitor(G_inh)]
            net.add(spike_monitors)

            if state_variables is not None:
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

            net.run(time * second, report=report_status)

            # __________________________

            results = {'spikes': {
                'exc': (np.array(spike_monitors[0].i), np.array(spike_monitors[0].t / second)),
                'inh': (np.array(spike_monitors[1].i), np.array(spike_monitors[1].t / second))
            }, 'weights': {}}

            results['weights']['ei'] = np.array(Sei.w)

            if plast_ie:
                results['weights']['ie'] = np.array(Sie.w)

            if plast_ee:
                results['weights']['ee'] = np.array(See.w)

            if target_rate_std != 0:
                results['target_rates'] = target_rates

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
                    results['state'][variable] = variable_full_data.reshape((-1,10,8000)).mean(axis=1)

            time_remaining -= chunks

            with open(folder + f'chunk{ii}.pkl', 'wb') as f:
                pickle.dump(results, f)

            ii += 1

        return results

def update_matrix(Z, N_exc, delays, weights, plast_ie, plast_ee):
    Z_trained = np.copy(Z)

    delays_ee, delays_ie, delays_ii, delays_ei = delays

    # inh to exc
    warr = np.array(weights['ei'])

    cutout_ei = Z_trained[:N_exc,N_exc:].T
    indices = cutout_ei.nonzero()
    cutout_ei[indices[0],indices[1]] = warr

    mask = (warr != 0)

    delays_ei = delays_ei[mask]

    # exc to inh
    if plast_ie:
        warr = np.array(weights['ie'])

        cutout_ie = Z_trained[N_exc:,:N_exc].T
        indices = cutout_ie.nonzero()
        cutout_ie[indices[0],indices[1]] = warr

        mask = (warr != 0)

        delays_ie = delays_ie[mask]

    # exc to exc
    if plast_ee:
        warr = np.array(weights['ee'])

        cutout_ee = Z_trained[:N_exc,:N_exc].T
        indices = cutout_ee.nonzero()
        cutout_ee[indices[0],indices[1]] = warr

        mask = (warr != 0)

        delays_ee = delays_ee[mask]

    delays_trained = delays_ee, delays_ie, delays_ii, delays_ei

    return Z_trained, delays_trained

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

def load_stim_file(filename, patterns, randstim, fraction=10):
    stims = pd.read_csv(filename, header=None, index_col=False).values[:len(patterns)]

    if not randstim:
        tuples = []
        for x in stims:
            pt = np.argwhere(patterns[int(x[2])]).flatten()
            cut = int(len(pt) / fraction)
            tuples.append((x[0], x[1], pt[:cut]))
    else:
        tuples = []
        for x in stims:
            # patlen = len(patterns[int(x[2])])
            pt = np.argwhere(patterns[int(x[2])]).flatten()
            cut = int(len(pt) / 100)
            randix = np.random.permutation(8000)[:cut]
            tuples = [(x[0], x[1], randix)]

    return tuples


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
        meta_eta=args.meta_eta
    )

    run_n_save(simulation_params, args, matrix_file=args.input)