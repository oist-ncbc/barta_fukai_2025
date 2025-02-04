python run_stim.py --plasticity hebb --patterns 1500 --perturb --folder sp1 --prefix lognorm --exc --count 1000 &
python run_stim.py --plasticity hebb --patterns 1500 --perturb --folder sp1 --prefix lognorm --inh --count 1000 &
python run_stim.py --plasticity hebb --patterns 1000 --perturb --folder sp1 --prefix lognorm --exc --count 1000 &
python run_stim.py --plasticity hebb --patterns 1000 --perturb --folder sp1 --prefix lognorm --inh --count 1000 &
python run_stim.py --plasticity rate --patterns 1000 --perturb --folder sp1 --prefix lognorm --exc --count 1000 &
python run_stim.py --plasticity rate --patterns 1000 --perturb --folder sp1 --prefix lognorm --inh --count 1000 &
python run_stim.py --plasticity threshold --patterns 1000 --perturb --folder sp1 --prefix lognorm --exc --count 1000 &
python run_stim.py --plasticity threshold --patterns 1000 --perturb --folder sp1 --prefix lognorm --inh --count 1000 &
python run_stim.py --plasticity hebb --patterns 1000 --perturb --folder sp1 --prefix lognorm --exc --count 1000 --suffix rand_var10 &
python run_stim.py --plasticity hebb --patterns 1000 --perturb --folder sp1 --prefix lognorm --inh --count 1000 --suffix rand_var10 &
