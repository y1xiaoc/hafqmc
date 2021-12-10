from ml_collections import ConfigDict, config_dict


def default_prop() -> ConfigDict:
    return ConfigDict({
        "max_nhs": None,
        "init_tsteps": [0.01]*10,
        "ortho_intvl": 0,
        "extra_field": 0,
        "expm_option": ["scan", 6, 1],
        "parametrize": "all",
        "timevarying": "hmf",
        "aux_network": {
            "hidden_sizes": [-1, -1, -1],
            "actv_fun": "gelu",
            "zero_init": True,
        },
        "init_random": 0.,
        "use_complex": False,
        "sqrt_tsvpar": False,
        "hermite_ops": False,
        "mf_subtract": False,
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
        },
        "ansatz":{
            "propagators":[default_prop()],
            "wfn_param": True,
            "wfn_random": 0.,
            "wfn_complex": False,
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
        },
        "loss": {
            # "step_weights": None,
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

    cfg.ansatz.propagators[0].max_nhs = 100
    cfg.ansatz.propagators[0].init_tsteps = [0.1] * 5
    cfg.ansatz.propagators[0].expm_option = ["scan", 2, 1]
    cfg.ansatz.propagators[0].aux_network = None
    cfg.ansatz.propagators[0].init_random = 0.1
    cfg.ansatz.propagators[0].sqrt_tsvpar = True
    cfg.ansatz.propagators[0].mf_subtract = True

    cfg.optim.optimizer = "adabelief"
    cfg.optim.grad_clip = 1.
    cfg.optim.iteration = 40_000
    cfg.optim.lr.start = 3e-4
    cfg.optim.lr.delay = 5e3
    cfg.optim.lr.decay = 1.

    cfg.sample.sampler = {"name": "langevin", "tau": 0.03, "steps": 10}
    cfg.sample.burn_in = 100

    cfg.loss.sign_factor = 1
    cfg.loss.sign_target = 0.7

    return cfg

