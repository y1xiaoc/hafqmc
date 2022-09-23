from ml_collections import ConfigDict, config_dict


def default_prop(with_net=False) -> ConfigDict:
    net_dict = {
        "hidden_sizes": [-1, -1, -1],
        "actv_fun": "gelu",
        "zero_init": True,
        "mod_density": False,
        }
    return ConfigDict({
        "max_nhs": None,
        "init_tsteps": [0.01]*3,
        "ortho_intvl": 0,
        "expm_option": ["scan", 6, 1],
        "parametrize": "all",
        "timevarying": "hmf",
        "aux_network": net_dict if with_net else None,
        "init_random": 0.,
        "sqrt_tsvpar": True,
        "use_complex": False,
        "hermite_ops": False,
        "mf_subtract": False,
        "dyn_mfshift": False,
        "priori_mask": None,
        "spin_mixing": False,
    }, 
    type_safe=False, convert_dict=True)


def ccsd_prop() -> ConfigDict:
    return ConfigDict({
        "type": "ccsd",
        "with_mask": True,
        "ortho_intvl": 0,
        "expm_option": ["scan", 1, 1],
        "parametrize": False,
        "timevarying": False,
        "init_random": 0.,
        "use_complex": False,
        "mf_subtract": False,
        "dyn_mfshift": False,
    }, 
    type_safe=False, convert_dict=True)


def ueg_prop() -> ConfigDict:
    return ConfigDict({
        "init_tsteps": [0.01]*3,
        "ortho_intvl": 0,
        "expm_option": ["scan", 6, 1],
        "parametrize": True,
        "timevarying": True,
        "init_random": 0.,
        "sqrt_tsvpar": True,
        "use_complex": False,
    },
    type_safe=False, convert_dict=True)


def default() -> ConfigDict:
    return ConfigDict({
        "restart": {
            "hamiltonian": None,
            "params": None,
            "states": None,
        },
        "seed": None,
        "molecule": {},
        "hamiltonian": {
            "chol_cut": 1e-6,
            "orth_ao": None,
            "full_eri": False,
            "with_cc": False,
        },
        "ansatz":{
            "propagators":[default_prop()],
            "wfn_param": True,
            "wfn_random": 0.,
            "wfn_complex": False,
            "wfn_spinmix": False,
        },
        "trial": None,
        "sample": {
            "size": 10_000,
            "sampler": {
                "name": "metropolis",
                "sigma": 0.05,
                "steps": 5,
            },
            "batch": 1_000,
            "burn_in": 1_000,
            "prop_steps": None,
        },
        "loss": {
            "sign_factor": 1.,
            "sign_target": 0.5,
            "sign_power": 2.,
            "std_factor": 0.,
            "std_target": 100.,
            "std_power": 2.,
        },
        "optim": {
            "batch": None,
            "iteration": 10_000,
            "optimizer": "adam", # {"name": "adam"},
            "grad_clip": None,
            "lr": {
                "start": 1e-4,
                "delay": 1e3,
                "decay": 1.0,
            },
            "baseline": None, #{"decay": 0.99},
        },
        "log": {
            "stat_freq": 1,
            "stat_path": "tbdata/",
            "ckpt_freq": 100,
            "ckpt_path": "checkpoint.pkl",
            "hpar_path": "hparams.yml",
            "hamil_path": "hamiltonian.pkl",
            "level": "WARNING",
        }
    }, 
    type_safe=False, convert_dict=True)


def example() -> ConfigDict:
    cfg = default()
    # use one propagator with 5 steps and 100 aux fields on each step
    cfg.ansatz.propagators[0].max_nhs = 100
    cfg.ansatz.propagators[0].init_tsteps = [0.1] * 5
    cfg.ansatz.propagators[0].expm_option = ["scan", 2, 1]
    cfg.ansatz.propagators[0].aux_network = None
    cfg.ansatz.propagators[0].init_random = 0.1
    cfg.ansatz.propagators[0].sqrt_tsvpar = True
    cfg.ansatz.propagators[0].mf_subtract = True
    # use adabelief and a long training
    cfg.optim.optimizer = "adabelief"
    cfg.optim.grad_clip = 1.
    cfg.optim.iteration = 40_000
    cfg.optim.lr.start = 3e-4
    cfg.optim.lr.delay = 5e3
    cfg.optim.lr.decay = 1.
    cfg.optim.baseline = {"decay": 0.99}
    # use mala sampler
    cfg.sample.sampler = {"name": "hmc", "dt": 0.1, "length": 1.}
    cfg.sample.burn_in = 100
    # add sign penalty to prevent it go below 0.7
    cfg.loss.sign_factor = 1
    cfg.loss.sign_target = 0.7
    # these are the suggested settings
    return cfg


def make_test(cfg) -> ConfigDict:
    cfg.trial = {
        "propagators": [],
        "wfn_param": False,
        "wfn_random": 0.,
        "wfn_complex": False,
    }
    cfg.optim.lr.start = 0.
    return cfg
    