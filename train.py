import time
import logging
import jax 
import optax
from jax import numpy as jnp
from optax._src import alias as optax_alias
from ml_collections import ConfigDict
from tensorboardX import SummaryWriter

from .molecule import build_mf
from .hamiltonian import Hamiltonian
from .ansatz import Ansatz, BraKet
from .estimator import make_eval_total
from .sampler import make_sampler, make_multistep
from .utils import ensure_mapping, save_pickle, load_pickle, Printer, cfg_to_yaml


def lower_penalty(s, factor=1., target=1., power=2.):
    return factor * jnp.maximum(target - s, 0) ** power

def upper_penalty(s, factor=1., target=1., power=2.):
    return factor * jnp.maximum(s - target, 0) ** power


def make_optimizer(name, lr_schedule, grad_clip=None, **kwargs):
    opt_fn = getattr(optax_alias, name)
    opt = opt_fn(lr_schedule, *kwargs)
    if grad_clip is not None:
        opt = optax.chain(optax.clip(grad_clip), opt)
    return opt


def make_lr_schedule(start=1e-4, decay=1., delay=1e4):
    if decay is None:
        return start
    return lambda t: start * jnp.power((1.0 / (1.0 + (t/delay))), decay)


def make_loss(expect_fn, step_weights=None,
              sign_factor=0., sign_target=1., sign_power=2.,
              std_factor=0., std_target=1., std_power=2):

    if step_weights is None:
        step_weights = 1.
    step_weights = jnp.asarray(step_weights).reshape(-1)

    def loss(params, data):
        e_tot, aux = expect_fn(params, data)
        e_tot = jnp.reshape(e_tot, -1)[-step_weights.size:]
        loss = (e_tot * step_weights[-e_tot.size:]).sum()
        if e_tot.size > 1:
            aux = jax.tree_map(lambda x: x[-1], aux)
            for ii in range(e_tot.size - 1):
                aux[f"e_mid{ii}"] = e_tot[ii]
        if sign_factor > 0:
            exp_s = aux["exp_s"]
            loss += lower_penalty(exp_s, sign_factor, sign_target, sign_power)
        if std_factor > 0:
            std_es = aux["std_es"]
            loss += upper_penalty(std_es, std_factor, std_target, std_power)
        return loss, aux
         
    return loss


def make_training_step(loss_and_grad, mc_sampler, optimizer):

    def step(key, params, mc_state, opt_state):
        mc_state, data = mc_sampler.sample(key, params, mc_state)
        (loss, aux), grads = loss_and_grad(params, data)
        grads = jax.tree_map(jnp.conj, grads) # for complex parameters
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        mc_state = mc_sampler.refresh(mc_state, params)
        return (params, mc_state, opt_state), (loss, aux)

    return step


def make_evaluation_step(expect_fn, mc_sampler):
    
    def step(key, params, mc_state, extras=None):
        mc_state, data = mc_sampler.sample(key, params, mc_state)
        e_tot, aux = expect_fn(params, data)
        return (params, mc_state, extras), (e_tot, aux)
    
    return step
        

def train(cfg: ConfigDict):
    # handle logging
    numeric_level = getattr(logging, cfg.log.level.upper())
    logging.basicConfig(level=numeric_level,
        format='# [%(asctime)s] %(levelname)s: %(message)s')
    writer = SummaryWriter(cfg.log.stat_path)
    print_fields = {"step": "", "loss": ".4f", "e_tot": ".4f", 
                    "exp_es": ".4f", "exp_s": ".4f"}
    if cfg.loss.std_factor >= 0:
        print_fields.update({"std_es": ".4f", "std_s": ".4f"})
    print_fields["lr"] = ".1e"
    printer = Printer(print_fields, time_format=".4f")
    if cfg.log.hpar_path:
        with open(cfg.log.hpar_path, "w") as hpfile:
            print(cfg_to_yaml(cfg), file=hpfile)

    # get the constants
    total_iter = cfg.optim.iteration
    sample_size = cfg.sample.size
    batch_size = cfg.sample.batch
    if sample_size % batch_size != 0:
        logging.warning("Sample size not divisible by batch size, rounding up")
    batch_multi = -(-sample_size // batch_size)
    sample_size = batch_size * batch_multi
    eval_size = cfg.optim.batch if cfg.optim.batch is not None else batch_size
    eval_mstep = jnp.size(cfg.loss.step_weights) - (cfg.loss.step_weights is None)
    if sample_size % eval_size != 0:
        logging.warning("Eval batch size not dividing sample size, using sample batch size")
        eval_size = batch_size

    # set up the hamiltonian
    if cfg.restart.hamiltonian is None:
        logging.info("Building molecule and doing HF calculation to get Hamiltonian")
        mf = build_mf(**cfg.molecule)
        print(f"# HF energy from pyscf calculation: {mf.e_tot}")
        if not mf.converged:
            logging.warning("HF calculation does not converge!")
        hamiltonian, init_wfn = Hamiltonian.from_pyscf_with_wfn(mf, **cfg.hamiltonian)
        save_pickle(cfg.log.hamil_path, 
            (hamiltonian.h1e, hamiltonian.ceri, hamiltonian.enuc, init_wfn))
    else:
        logging.info("Loading Hamiltonian from saved file")
        h1e, ceri, enuc, init_wfn = load_pickle(cfg.restart.hamiltonian)
        hamiltonian = Hamiltonian(h1e, ceri, enuc, cfg.hamiltonian.full_eri)
        print(f"# HF energy from loaded: {hamiltonian.local_energy(init_wfn, init_wfn)}")

    # set up all other classes and functions
    logging.info("Setting up the training loop")
    ansatz = Ansatz.create(hamiltonian, init_wfn, **cfg.ansatz)
    trial = (Ansatz.create(hamiltonian, init_wfn, **cfg.trial) 
             if cfg.trial is not None else None)
    braket = BraKet(ansatz, trial)
    sampler_1s = make_sampler(braket, 
        **ensure_mapping(cfg.sample.sampler, default_key="name"))
    mc_sampler = make_multistep(sampler_1s, batch_multi, concat=True)
    lr_schedule = make_lr_schedule(**cfg.optim.lr)
    optimizer = make_optimizer(lr_schedule=lr_schedule, grad_clip=cfg.optim.grad_clip,
        **ensure_mapping(cfg.optim.optimizer, default_key="name"))
    expect_fn = make_eval_total(hamiltonian, braket, 
        multi_steps=eval_mstep, default_batch=eval_size, calc_stds=True)
    loss_fn = make_loss(expect_fn, **cfg.loss)
    loss_and_grad = jax.value_and_grad(loss_fn, has_aux=True)

    # the core training iteration, to be pmaped
    if cfg.optim.lr.start > 0:
        train_step = make_training_step(loss_and_grad, mc_sampler, optimizer)
    else:
        train_step = make_evaluation_step(expect_fn, mc_sampler)
    train_step = jax.jit(train_step)
    
    # set up all states
    if cfg.restart.states is None:
        logging.info("Initializing parameters and states")
        key = jax.random.PRNGKey(cfg.seed)
        key, pakey, mckey = jax.random.split(key, 3)
        fshape = braket.fields_shape()
        if cfg.restart.params is None:
            params = jax.jit(braket.init)(pakey, jax.tree_map(jnp.zeros, fshape))
        else:
            logging.info("Loading parameters from saved file")
            params = load_pickle(cfg.restart.params)
            if isinstance(params, tuple): 
                params = params[1]
        mc_state = mc_sampler.init(mckey, params, batch_size)
        opt_state = optimizer.init(params)
        if cfg.sample.burn_in > 0:
            logging.info(f"Burning in the sampler for {cfg.sample.burn_in} steps")
            for ii in range(cfg.sample.burn_in):
                key, subkey = jax.random.split(key)
                mc_state, _ = jax.jit(sampler_1s.sample)(subkey, params, mc_state)
    else:
        logging.info("Loading parameters and states from saved file")
        key, params, mc_state, opt_state = load_pickle(cfg.restart.states)

    # the actual training iteration
    logging.info("Start training")
    printer.print_header(prefix="# ")
    for ii in range(total_iter + 1):
        printer.reset_timer()
        key, subkey = jax.random.split(key)
        (params, mc_state, opt_state), (loss, aux) = train_step(subkey, params, mc_state, opt_state)
        # logging anc checkpointing
        if ii % cfg.log.stat_freq == 0:
            _lr = lr_schedule(opt_state[-1][0].count) if callable(lr_schedule) else lr_schedule
            printer.print_fields({"step": ii, "loss": loss, **aux, "lr": _lr})
            writer.add_scalars("stat", {"loss": loss, **aux, "lr": _lr}, global_step=ii)
        if ii % cfg.log.ckpt_freq == 0:
            save_pickle(cfg.log.ckpt_path, (key, params, mc_state, opt_state))
    writer.close()
    
    return params, mc_state, opt_state