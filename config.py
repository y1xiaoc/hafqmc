from ml_collections import ConfigDict, config_dict


def default() -> ConfigDict:
    cfg = ConfigDict({
        "molecule": config_dict.placeholder(dict, required=True),
        "hamiltonian": {
            "chol_cut": 1e-6,
            "orth_ao": None,
            "full_eri": False,
        },
        "propagator":{
            "init_tsteps": [0.01]*10,
            "extra_field": 0,
            "parametrize": True,
            "timevarying": False,
            "aux_network": {
                "hidden_sizes": [-1, -1, -1],
                "actv_fn": "gelu",
                "zero_init": True,
            },
            "use_complex" : False,
        },
        "sample": {
            "size": 10_000,
            "sampler": "gaussian", # {"name": "gaussian"},
            "batch": 100,
            "burn_in": 0,
        },
        "loss": {
            "sign_factor": 1.,
            "sign_power": 2.,
        },
        "optim": {
            "iteration": 100000,
            "optimizer": "adam", # {"name": "adam"},
            "lr": {
                "start" : 1e-4,
                "delay" : 1e4,
                "decay" : 1.0
            },
        },
    })