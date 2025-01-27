import argparse
import os
import yaml


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plasticity', type=str)
    parser.add_argument('--patterns', type=int)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--stim', action='store_true')
    parser.add_argument('--stim_short', action='store_true')
    parser.add_argument('--stim_long', action='store_true')
    parser.add_argument('--spont', action='store_true')
    parser.add_argument('--spont_nonadapt', action='store_true')
    parser.add_argument('--spont_state', action='store_true')
    parser.add_argument('--perturb', action='store_true')
    parser.add_argument('--state', action='store_true')
    parser.add_argument('--suffix', type=str, default='')
    parser.add_argument('--prefix', type=str, default='test')
    parser.add_argument('--folder', type=str, default='')
    parser.add_argument('--target_rate', type=float, default=3.)
    parser.add_argument('--fraction', type=float, default=10.)
    parser.add_argument('--eta', type=float, default=1e-3)
    parser.add_argument('--traintime', type=int, default=2000)
    parser.add_argument('--sponttime', type=int, default=2000)
    parser.add_argument('--train_matrix', type=str, default='naive')
    parser.add_argument('--rate_file', type=str)
    parser.add_argument('--tau_stdp', type=float, default=20)
    parser.add_argument('--meta_eta', type=float, default=0)
    args = parser.parse_args()

    with open('config/server_config.yaml') as f:
        config = yaml.safe_load(f)

    if args.suffix != '':
        suffix = '_' + args.suffix
    else:
        suffix = ''

    if args.train_matrix == 'naive':
        matrix = f'{args.prefix}{args.patterns}'
    else:
        matrix = f'training_{args.train_matrix}_{args.prefix}{args.patterns}_matrix'

    # if args.folder != '':
    #     folder = '/' + args.folder
    # else:
    #     folder = args.folder

    path = f"{config['data_path']}/{args.folder}"

    if args.plasticity == 'threshold':
        thresholds = ' --thr_file data/thresholds.pkl'
    else:
        thresholds = ''

    if args.rate_file is not None:
        rate_text = ' --rate_file ' + args.rate_file
    else:
        rate_text = ''

    if args.train:
        os.system(f"""
        python network.py \
            -f {path}/connectivity/{matrix}.pkl \
            -o {path}/data/training_{args.plasticity}_{args.prefix}{args.patterns}_results{suffix}.pkl \
            -t {args.traintime} \
            --matrix {path}/connectivity/training_{args.plasticity}_{args.prefix}{args.patterns}_matrix{suffix}.pkl \
            --eta {args.eta} \
            --target_rate {args.target_rate} \
            --alpha2 0.3 \
            --reset \
            --plasticity {args.plasticity} \
            --tau_stdp {args.tau_stdp} \
            --meta_eta {args.meta_eta} \
            --trstd 0 {rate_text}
        python analysis.py -r {path}/data/training_{args.plasticity}_{args.prefix}{args.patterns}_results{suffix}.pkl --img img/tmp.png""")

    if args.stim:
        os.system(f"""
        python network.py \
            -f {path}/connectivity/training_{args.plasticity}_{args.prefix}{args.patterns}_matrix{suffix}.pkl \
            -o {path}/data/trained_{args.plasticity}_{args.prefix}{args.patterns}_results_300stim{suffix}.pkl \
            -t 302 \
            --stimulus config/stimuli_all_01.csv \
            --eta 0 \
            --target_rate 3 \
            --alpha2 0.3 \
            --reset \
            --tau_stdp {args.tau_stdp} \
            --stimfrac 10 {thresholds}
        python analysis.py -r {path}/data/trained_{args.plasticity}_{args.prefix}{args.patterns}_results_300stim{suffix}.pkl --img img/tmp.png
        """)

    if args.stim_short:
        os.system(f"""
        python network.py \
            -f {path}/connectivity/training_{args.plasticity}_{args.prefix}{args.patterns}_matrix{suffix}.pkl \
            -o {path}/data/trained_{args.plasticity}_{args.prefix}{args.patterns}_results_stim{suffix}_short.pkl \
            -t 102 \
            --stimulus config/stimuli_short.csv \
            --eta 0 \
            --target_rate 3 \
            --alpha2 0.3 \
            --reset \
            --tau_stdp {args.tau_stdp} \
            --stimfrac {args.fraction} {thresholds}
        python analysis.py -r {path}/data/trained_{args.plasticity}_{args.prefix}{args.patterns}_results_stim{suffix}_short.pkl --dt 0.01 --img img/tmp.png
        """)

    if args.stim_long:
        os.system(f"""
        python network.py \
            -f {path}/connectivity/training_{args.plasticity}_{args.prefix}{args.patterns}_matrix{suffix}.pkl \
            -o {path}/data/trained_{args.plasticity}_{args.prefix}{args.patterns}_results_stim{suffix}_long.pkl \
            -t 202 \
            --stimulus config/stimuli_long.csv \
            --eta 0 \
            --target_rate 3 \
            --a1_off \
            --alpha2 0. \
            --reset \
            --tau_stdp {args.tau_stdp} \
            --stimfrac 1 {thresholds}
        python analysis.py -r {path}/data/trained_{args.plasticity}_{args.prefix}{args.patterns}_results_stim{suffix}_long.pkl --dt 0.1 --img img/tmp.png
        """)

    if args.spont_nonadapt:
        os.system(f"""
        python network.py \
            -f {path}/connectivity/training_{args.plasticity}_{args.prefix}{args.patterns}_matrix{suffix}.pkl \
            -o {path}/data/trained_{args.plasticity}_{args.prefix}{args.patterns}_results_spont{suffix}_nonadapt.pkl \
            -t {args.sponttime} \
            --eta 0 \
            --target_rate 3 \
            --a1_off \
            --alpha2 0. \
            --tau_stdp {args.tau_stdp} \
            --reset
        python analysis.py -r {path}/data/trained_{args.plasticity}_{args.prefix}{args.patterns}_results_spont{suffix}_nonadapt.pkl --dt 0.1 --img img/tmp.png
        """)

    if args.spont:
        os.system(f"""
        python network.py \
            -f {path}/connectivity/training_{args.plasticity}_{args.prefix}{args.patterns}_matrix{suffix}.pkl \
            -o {path}/data/trained_{args.plasticity}_{args.prefix}{args.patterns}_results_spont{suffix}.pkl \
            -t {args.sponttime} \
            --eta 0 \
            --target_rate 3 \
            --alpha2 0.3 \
            --tau_stdp {args.tau_stdp} \
            --reset {thresholds}
        python analysis.py -r {path}/data/trained_{args.plasticity}_{args.prefix}{args.patterns}_results_spont{suffix}.pkl --img img/tmp.png
        """)

    if args.state:
        os.system(f"""
        python network.py \
            -f {path}/connectivity/training_{args.plasticity}_{args.prefix}{args.patterns}_matrix{suffix}.pkl \
            -o {path}/data/trained_{args.plasticity}_{args.prefix}{args.patterns}_results_state{suffix}.pkl \
            -t 11 \
            --eta 0 \
            --target_rate 3 \
            --alpha2 0.3 \
            --reset \
            --stimfrac {args.fraction} \
            --tau_stdp {args.tau_stdp} \
            --record gi ge {thresholds}
        python analysis.py -r {path}/data/trained_{args.plasticity}_{args.prefix}{args.patterns}_results_state{suffix}.pkl --img img/tmp.png
        """)

    if args.spont_state:
        os.system(f"""
        python network.py \
            -f {path}/connectivity/training_{args.plasticity}_{args.prefix}{args.patterns}_matrix{suffix}.pkl \
            -o {path}/data/trained_{args.plasticity}_{args.prefix}{args.patterns}_results_spont_state{suffix}.pkl \
            -t 5 \
            --eta 0 \
            --target_rate 3 \
            --alpha2 0.3 \
            --reset \
            --tau_stdp {args.tau_stdp} \
            --record v ge gi {thresholds}
        python analysis.py -r {path}/data/trained_{args.plasticity}_{args.prefix}{args.patterns}_results_spont_state{suffix}.pkl --dt 0.1 --img img/tmp.png
        """)

    if args.perturb:
        os.system(f"""
        python single_neuron.py \
            -f {path}/connectivity/training_{args.plasticity}_{args.prefix}{args.patterns}_matrix{suffix}.pkl \
            -o {path}/data/trained_{args.plasticity}_{args.prefix}{args.patterns}_results_perturbed{suffix}.pkl \
            -t {args.sponttime} \
            --eta 0 \
            --target_rate 3 \
            --alpha2 0.3 \
            --vardata_e config/var_data_{args.plasticity}_{args.prefix}{args.patterns}{suffix}_e.csv \
            --vardata_i config/var_data_{args.plasticity}_{args.prefix}{args.patterns}{suffix}_i.csv \
            --tau_stdp {args.tau_stdp} \
            --reset {thresholds}
        python analysis.py -r {path}/data/trained_{args.plasticity}_{args.prefix}{args.patterns}_results_perturbed{suffix}.pkl -t 4001 --img img/tmp.png
        """)
        # os.system(f"""
        # python analysis.py -f -r {path}/data/trained_{args.plasticity}_nonburst{args.patterns}_results_perturbed{suffix}.pkl -t 2001 --img img/tmp.png
        # """)