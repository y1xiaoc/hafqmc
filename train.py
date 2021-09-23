import time
import logging
import jax 
import optax
from jax import lax
from jax import numpy as jnp
from optax._src import alias as optax_alias
from ml_collections import ConfigDict
from tensorboardX import SummaryWriter

from .molecule import build_mf
from .hamiltonian import Hamiltonian
from .propagator import Propagator
from .estimator import make_eval_total
from .sampler import get_shape, make_sampler, make_multistep
from .utils import ensure_mapping, save_pickle


def sign_penalty(s, factor=1., target=1., power=2.):
    return factor * jnp.maximum(target - s, 0) ** power


def make_optimizer(name, lr_schedule, grad_clip=None, **kwargs):
    opt_fn = getattr(optax_alias, name)
    opt = opt_fn(lr_schedule, *kwargs)
    if grad_clip is not None:
        opt = optax.chain(opt, optax.clip(grad_clip))
    return opt


def make_lr_schedule(start=1e-4, delay=1e4, decay=1.):
    return lambda t: start * jnp.power((1.0 / (1.0 + (t/delay))), decay)


def make_loss(expect_fn, sign_factor=1., sign_target=1., sign_power=2.):

    def loss(params, data):
        e_tot, aux = expect_fn(params, data)
        exp_s = aux["exp_s"]
        sp = sign_penalty(exp_s, sign_factor, sign_target, sign_power)
        return e_tot + sp, aux
         
    return loss


def make_training_step(loss_and_grad, mc_sampler, optimizer):

    def step(key, params, mc_state, opt_state):
        mc_state, data = mc_sampler.sample(key, params, mc_state)
        (loss, aux), grads = loss_and_grad(params, data)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return (params, mc_state, opt_state), (loss, aux)

    return step
        

def train(cfg: ConfigDict):
    # handle logging
    numeric_level = getattr(logging, cfg.log.level.upper())
    logging.basicConfig(level=numeric_level)
    writer = SummaryWriter(cfg.log.stat_path)
    if cfg.log.hpar_path:
        with open(cfg.log.hpar_path, "w") as hpfile:
            print(cfg, file=hpfile)

    # get the constants
    key = jax.random.PRNGKey(cfg.seed)
    total_iter = cfg.optim.iteration
    sample_size = cfg.sample.size
    batch_size = cfg.sample.batch
    if sample_size % batch_size != 0:
        logging.warning("sample size is not divisible by batch size, round up")
    batch_multi = -(-sample_size // batch_size)

    # do the scf calculation as init guess
    mf = build_mf(**cfg.molecule)
    print(f"# HF energy from pyscf calculation: {mf.e_tot}")
    if not mf.converged:
        logging.warning("HF calculation does not converge!")

    # set up all classes and functions
    hamiltonian, init_wfn = Hamiltonian.from_pyscf_with_wfn(mf, **cfg.hamiltonian)
    propagator = Propagator.create(hamiltonian, init_wfn, **cfg.propagator)
    sampler_1s = make_sampler(propagator, 
        **ensure_mapping(cfg.sample.sampler, default_key="name"))
    mc_sampler = make_multistep(sampler_1s, batch_multi, concat=False)
    lr_schedule = make_lr_schedule(**cfg.optim.lr)
    optimizer = make_optimizer(lr_schedule=lr_schedule, grad_clip=cfg.optim.grad_clip,
        **ensure_mapping(cfg.optim.optimizer, default_key="name"))
    expect_fn = make_eval_total(hamiltonian, propagator, default_batch=batch_size)
    loss_fn = make_loss(expect_fn, **cfg.loss)
    loss_and_grad = jax.value_and_grad(loss_fn, has_aux=True)

    # the core training iteration, to be pmaped
    train_step = make_training_step(loss_and_grad, mc_sampler, optimizer)
    train_step = jax.jit(train_step)
    
    # set up all states
    key, pakey, mckey = jax.random.split(key, 3)
    field_shape = get_shape(propagator)
    params = propagator.init(pakey, jnp.zeros(field_shape[-2:]))
    mc_state = mc_sampler.init(mckey, params, batch_size, cfg.sample.burn_in)
    opt_state = optimizer.init(params)

    # the actual training iteration
    for ii in range(total_iter):
        tic = time.time()
        key, subkey = jax.random.split(key)
        (params, mc_state, opt_state), (loss, aux) = train_step(subkey, params, mc_state, opt_state)

        # logging anc checkpointing
        if ii == 0:
            print("# step\tloss\te_tot\texp_es\texp_s\ttime")
        if ii % cfg.log.stat_freq == 0:
            print(f"{ii}\t{loss:.4f}\t{aux['e_tot']:.4f}\t"
                  f"{aux['exp_es']:.4f}\t{aux['exp_s']:.4f}\t{time.time()-tic:.4f}")
            writer.add_scalars("stat", {"loss": loss, **aux}, global_step=ii)
        if ii % cfg.log.ckpt_freq == 0:
            save_pickle(cfg.log.ckpt_path, (key, params, mc_state, opt_state))
    
    return params, mc_state, opt_state