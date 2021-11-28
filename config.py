from ml_collections import ConfigDict, config_dict


def default() -> ConfigDict:
    return ConfigDict({
        "restart": {
            "hamiltonian": None,
            "params": None,
            "states": None,
        },
        "seed": config_dict.placeholder(int),
        "molecule": {},
        "hamiltonian": {
            "chol_cut": 1e-6,
            "orth_ao": None,
            "full_eri": False,
        },
        "propagator":{
            "max_nhs": None,
            "init_tsteps": [0.01]*10,
            "ortho_intvl": 0,
            "extra_field": 0,
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