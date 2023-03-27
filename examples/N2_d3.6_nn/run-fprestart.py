import scipy.signal
import hafqmc.train
import hafqmc.config

cfg = hafqmc.config.example()

# cfg.molecule = dict(atom="N 0 0 0; N 0 0 3;", basis="ccpvdz", unit='B')
cfg.restart.hamiltonian = "hamiltonian.pkl"
cfg.restart.params = "oldstates.pkl"

cfg.ansatz.wfn_spinmix = True
cfg.ansatz.propagators[0].max_nhs = 100
cfg.ansatz.propagators[0].aux_network = {
    "hidden_sizes": [-1, -1, -1],
    "actv_fun": "gelu",
    "mod_density": False,
    "zero_init": True,
}
cfg.ansatz.propagators[0].init_tsteps = [0.01] * 3
cfg.ansatz.propagators[0].sqrt_tsvpar = True
cfg.ansatz.propagators[0].init_random = 0.1
cfg.ansatz.propagators[0].hermite_ops = False
cfg.ansatz.propagators[0].mf_subtract = True
cfg.ansatz.propagators[0].spin_mixing = True

cfg.ansatz.propagators.append(hafqmc.config.default_prop())
cfg.ansatz.propagators[1].aux_network = None
cfg.ansatz.propagators[1].init_tsteps = [0.1] * 20
cfg.ansatz.propagators[1].parametrize = False

cfg.trial = {
    "propagators": [],
    "wfn_param": False,
    "wfn_random": 0.,
    "wfn_complex": False,
}

cfg.optim.optimizer = "adabelief"
cfg.optim.grad_clip = 1.
cfg.optim.iteration = 10000
cfg.optim.lr.start = 0.

cfg.sample.batch = 1000
cfg.sample.sampler = "gaussian" #{"name": "langevin", "tau": 0.03, "steps": 10}
#cfg.sample.sampler = {"name": "hmc", "dt": 0.1, "length": 1.}
cfg.sample.burn_in = 100

cfg.loss.sign_factor = 1
cfg.loss.sign_target = 0.7
# cfg.loss.std_factor = 0.1
# cfg.loss.std_target = 100.

cfg.seed = 0
cfg.log.level = "info"

if __name__ == "__main__": 
    hafqmc.train.train(cfg)
