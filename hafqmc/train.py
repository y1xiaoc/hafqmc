import time
import logging
import jax 
import optax
import numpy as onp
from jax import numpy as jnp
from optax._src import alias as optax_alias
from ml_collections import ConfigDict
from typing import NamedTuple

from .molecule import build_mf
from .hamiltonian import Hamiltonian, HamiltonianPW
from .ansatz import Ansatz, BraKet
from .estimator import make_eval_total
from .sampler import make_sampler, make_multistep, make_batched, SamplerUnion
from .utils import save_pickle, load_pickle, Printer, cfg_to_yaml, backup_if_exist
from .utils import ensure_mapping, make_moving_avg, PyTree, tree_map


def lower_penalty(s, factor=1., target=1., power=2.):
    return factor * jnp.maximum(target - s, 0) ** power

def upper_penalty(s, factor=1., target=1., power=2.):
    return factor * jnp.maximum(s - target, 0) ** power


def make_optimizer(name, lr_schedule, grad_clip=None, **kwargs):
    opt_fn = getattr(optax_alias, name)
    opt = opt_fn(lr_schedule, **kwargs)
    if grad_clip is not None:
        opt = optax.chain(optax.clip(grad_clip), opt)
    return opt


def make_lr_schedule(start=1e-4, decay=1., delay=1e4):
    if decay is None:
        return start
    return lambda t: start * jnp.power((1.0 / (1.0 + (t/delay))), decay)


def make_loss(expect_fn, 
              sign_factor=0., sign_target=1., sign_power=2.,
              std_factor=0., std_target=1., std_power=2):

    def loss(params, data, *extra, **kwargs):
        e_tot, aux = expect_fn(params, data, *extra, **kwargs)
        loss = e_tot
        if sign_factor > 0:
            exp_s = aux["exp_s"]
            loss += lower_penalty(exp_s, sign_factor, sign_target, sign_power)
        if std_factor > 0:
            std_es = aux["std_es"]
            loss += upper_penalty(std_es, std_factor, std_target, std_power)
        return loss, aux
         
    return loss


class TrainingState(NamedTuple):
    step: int
    params: PyTree
    mc_state: PyTree
    opt_state: PyTree
    est_state: PyTree = None


def make_training_step(loss_and_grad, mc_sampler, optimizer, accumulator=None):
    is_union = isinstance(mc_sampler, SamplerUnion)

    def step(key, train_state, sample_flag=None):
        ii, params, mc_state, opt_state, ebar = train_state
        sampler = mc_sampler.switch(sample_flag) if is_union else mc_sampler
        mc_state = sampler.refresh(mc_state, params)
        mc_state, data = sampler.sample(key, params, mc_state)
        (loss, aux), grads = loss_and_grad(params, data, ebar)
        grads = tree_map(jnp.conj, grads) # for complex parameters
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        if accumulator is not None: ebar = accumulator(ebar, aux["e_tot"], ii)
        new_state = TrainingState(ii+1, params, mc_state, opt_state, ebar)
        return new_state, (loss, aux)

    return step


def make_evaluation_step(expect_fn, mc_sampler):
    is_union = isinstance(mc_sampler, SamplerUnion)
    
    def step(key, train_state, sample_flag=None):
        ii, params, mc_state, *other = train_state
        sampler = mc_sampler.switch(sample_flag) if is_union else mc_sampler
        mc_state, data = sampler.sample(key, params, mc_state)
        e_tot, aux = expect_fn(params, data)
        new_state = TrainingState(ii+1, params, mc_state, *other)
        return new_state, (e_tot, aux)
    
    return step
        

def train(cfg: ConfigDict):
    # handle logging
    logging.basicConfig(force=True, format='# [%(asctime)s] %(levelname)s: %(message)s')
    logger = logging.getLogger("train")
    log_level = getattr(logging, cfg.log.level.upper())
    logger.setLevel(log_level)
    backup_if_exist(cfg.log.stat_path, prefix="bck")
    writer = open(cfg.log.stat_path, "w", buffering=1)
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
    sample_batch = cfg.sample.batch
    if sample_size % sample_batch != 0:
        logger.warning("Sample size not divisible by batch size, rounding up")
    sample_step = -(-sample_size // sample_batch)
    sample_size = sample_batch * sample_step
    sample_prop = cfg.sample.prop_steps
    eval_batch = cfg.optim.batch if cfg.optim.batch is not None else sample_batch
    if sample_size % eval_batch != 0:
        logger.warning("Eval batch size not dividing sample size, using sample batch size")
        eval_batch = sample_batch

    # set up the hamiltonian
    if cfg.restart.hamiltonian is None:
        if "ueg" not in cfg:
            logger.info("Building molecule and doing HF calculation to get Hamiltonian")
            mf = build_mf(**cfg.molecule)
            print(f"# HF energy from pyscf calculation: {mf.e_tot}")
            if not mf.converged:
                logger.warning("HF calculation does not converge!")
            hamiltonian = Hamiltonian.from_pyscf(mf, **cfg.hamiltonian)
        else:
            logger.info("Using uniform electron gas Hamiltonian")
            hamiltonian = HamiltonianPW.from_ueg(**cfg.ueg)
            print(f"# HF energy for UEG hamiltonian: {hamiltonian.local_energy()}")
        save_pickle(cfg.log.hamil_path, hamiltonian.to_tuple())
    else:
        logger.info("Loading Hamiltonian from saved file")
        hamil_data = load_pickle(cfg.restart.hamiltonian)
        HamCls = Hamiltonian if len(hamil_data) <= 5 else HamiltonianPW
        hamiltonian = HamCls(*hamil_data)
        print(f"# HF energy from loaded: {hamiltonian.local_energy()}")

    # set up all other classes and functions
    logger.info("Setting up the training loop")
    ansatz = Ansatz.create(hamiltonian, **cfg.ansatz)
    trial = (None if cfg.trial is None 
             else ansatz if isinstance(cfg.trial, str) and 
                    cfg.trial.lower() in ("same", "share", "ansatz")
             else Ansatz.create(hamiltonian, **cfg.trial))
    braket = BraKet(ansatz, trial)
    if (sample_prop is None or isinstance(sample_prop, int)
      or (isinstance(sample_prop, (tuple, list)) and len(sample_prop) == 2)):
        sampler_1s_1c = make_sampler(braket, max_prop=sample_prop,
            **ensure_mapping(cfg.sample.sampler, default_key="name"))
    else:
        sampler_1s_1c = SamplerUnion({
            mp: make_sampler(braket, max_prop=mp,
                **ensure_mapping(cfg.sample.sampler, default_key="name"))
            for mp in sample_prop})
    sampler_1s_nc = make_batched(sampler_1s_1c, sample_batch, concat=False)
    mc_sampler = make_multistep(sampler_1s_nc, sample_step, concat=True)
    lr_schedule = make_lr_schedule(**cfg.optim.lr)
    optimizer = make_optimizer(lr_schedule=lr_schedule, grad_clip=cfg.optim.grad_clip,
        **ensure_mapping(cfg.optim.optimizer, default_key="name"))
    expect_fn = make_eval_total(hamiltonian, braket, 
        default_batch=eval_batch, calc_stds=True)
    loss_fn = make_loss(expect_fn, **cfg.loss)
    loss_and_grad = jax.value_and_grad(loss_fn, has_aux=True)
    moving_avg_fn = (make_moving_avg(**cfg.optim.baseline)
        if cfg.optim.baseline is not None else None)

    # the core training iteration, to be pmaped
    if cfg.optim.lr.start > 0:
        train_step = make_training_step(loss_and_grad, mc_sampler, optimizer, moving_avg_fn)
    else:
        train_step = make_evaluation_step(expect_fn, mc_sampler)
    train_step = jax.jit(train_step, static_argnames="sample_flag")
    
    # set up all states
    if cfg.restart.states is None:
        logger.info("Initializing parameters and states")
        key = jax.random.PRNGKey(cfg.seed)
        key, pakey, mckey = jax.random.split(key, 3)
        fshape = braket.fields_shape()
        if cfg.restart.params is None:
            params = jax.jit(braket.init)(pakey, tree_map(jnp.zeros, fshape))
        else:
            logger.info("Loading parameters from saved file")
            params = load_pickle(cfg.restart.params)
            if isinstance(params, tuple): params = params[1]
            if isinstance(params, tuple): params = params[1]
        mc_state = mc_sampler.init(mckey, params)
        opt_state = optimizer.init(params)
        if cfg.sample.burn_in > 0:
            logger.info(f"Burning in the sampler for {cfg.sample.burn_in} steps")
            key, subkey = jax.random.split(key)
            mc_state = sampler_1s_nc.burn_in(subkey, params, mc_state, cfg.sample.burn_in)
        ebar = hamiltonian.local_energy() if cfg.optim.baseline is not None else None
        train_state = TrainingState(0, params, mc_state, opt_state, ebar)
    else:
        logger.info("Loading parameters and states from saved file")
        key, *rest = load_pickle(cfg.restart.states)
        rest = rest[0] if len(rest) == 1 else (0, *rest)
        if len(rest) < 5 and cfg.optim.baseline is not None:
            rest = (*rest, hamiltonian.local_energy())
        train_state = TrainingState(*rest)

    # the actual training iteration
    logger.info("Start training")
    printer.print_header(prefix="# ")
    for ii in range(total_iter + 1):
        printer.reset_timer()
        # choose sampler
        sflag = None
        if not (sample_prop is None or isinstance(sample_prop, int)):
            key, flagkey = jax.random.split(key)
            sflag = sample_prop[jax.random.choice(flagkey, len(sample_prop))]
        # core training step
        key, subkey = jax.random.split(key)
        train_state, (loss, aux) = train_step(subkey, train_state, sample_flag=sflag)
        # logging anc checkpointing
        if ii % cfg.log.stat_freq == 0:
            if sflag is not None: aux["nprop"] = sflag
            _lr = (lr_schedule(train_state.opt_state[-1][0].count) 
                if callable(lr_schedule) else lr_schedule)
            stat_dict = {"step": ii, "loss": loss, **aux, "lr": _lr}
            printer.print_fields(stat_dict)
            onp.savetxt(writer, onp.array(list(stat_dict.values())).reshape(1,-1),
                        header=" ".join(stat_dict.keys()) if ii == 0 else "")
        if ii % cfg.log.ckpt_freq == 0:
            save_pickle(cfg.log.ckpt_path, (key, tuple(train_state)))
    writer.close()
    
    return train_state