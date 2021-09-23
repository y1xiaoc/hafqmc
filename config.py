from ml_collections import ConfigDict, config_dict


def default() -> ConfigDict:
    return ConfigDict({
        "seed": config_dict.placeholder(int, required=True),
        "molecule": config_dict.placeholder(dict, required=True),
        "hamiltonian": {
            "chol_cut": 1e-6,
            "orth_ao": None,
            "full_eri": False,
        },
        "propagator":{
            "max_nhs": None,
            "init_tsteps": [0.01]*10,
            "extra_field": 0,
            "parametrize": True,
            "timevarying": "hmf",
            "aux_network": {
                "hidden_sizes": [-1, -1, -1],
                "actv_fun": "gelu",
                "zero_init": True,
            },
            "use_complex" : False,
        },
        "sample": {
            "size": 10_000,
            "sampler": "gaussian", # {"name": "gaussian"},
            "batch": 1_000,
            "burn_in": 0,
        },
        "loss": {
            "sign_factor": 1.,
            "sign_target": 0.5,
            "sign_power": 2.,
        },
        "optim": {
            "iteration": 10_000,
            "optimizer": "adam", # {"name": "adam"},
            "grad_clip": 1.,
            "lr": {
                "start" : 1e-4,
                "delay" : 1e4,
                "decay" : 1.0
            },
        },
        "log": {
            "stat_freq": 1,
            "stat_path": "tbdata/",
            "ckpt_freq": 100,
            "ckpt_path": "checkpoint.pkl",
            "hpar_path": "hparams.yml",
            "level": "WARNING"
        }
    })