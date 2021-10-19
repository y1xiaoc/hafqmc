import jax
from jax import lax
from jax import numpy as jnp
from typing import NamedTuple, Callable, Tuple
from functools import partial

from .propagator import Propagator
from .utils import PyTree, Array


def log_dens_gaussian(x, mu=0., sigma=1.):
    """unnormalized log density of Gaussian distribution"""
    return -0.5 * ((x - mu) / sigma) ** 2


def get_shape(prop: Propagator):
    """"calculate needed field shape from propagator: [2 x nts x nsite]"""
    nts = len(prop.init_tsteps)
    nfield = prop.init_vhs.shape[0] + prop.extra_field
    return (2, nts, nfield)


KeyArray = Array
Params = PyTree
State = PyTree
Data = PyTree
class MCSampler(NamedTuple):
    sample: Callable[[KeyArray, Params, State], Tuple[State, Data]]
    init: Callable[[KeyArray, Params, int], State]
    refresh: Callable[[State, Params], State]
    def __call__(self, key: KeyArray, params: Params, state: State):
        """Call the sample function. See `self.sample` for details."""
        return self.sample(key, params, state)


def make_multistep_fn(sample_fn, nstep, concat=False):

    def multi_sample(key, params, state):
        inner = lambda s,k: sample_fn(k, params, s)
        keys = jax.random.split(key, nstep)
        new_state, data = lax.scan(inner, state, keys)
        if concat:
            data = jax.tree_map(jnp.concatenate, data)
        return new_state, data

    return multi_sample


def make_multistep(sampler, nstep, concat=False):
    sample_fn, init_fn, refresh_fn = sampler
    multisample_fn = make_multistep_fn(sample_fn, nstep, concat)
    return MCSampler(multisample_fn, init_fn, refresh_fn)


def make_sampler(prop: Propagator, name: str, **kwargs):
    name = name.lower()
    if name == "gaussian":
        maker = make_gaussian
    elif name in ("metropolis", "mcmc", "mh"):
        maker = make_metropolis
    else:
        raise NotImplementedError(f"unsupported sampler type: {name}")
    return maker(prop, **kwargs)


def make_gaussian(prop: Propagator, mu=0., sigma=1., truncate=None):
    sample_shape = get_shape(prop)

    def sample(key, params, state):
        nbatch = state.shape[0]
        shape = (nbatch, *sample_shape)
        if truncate is not None:
            trc = jnp.abs(truncate)
            rawgs = jax.random.truncated_normal(key, -trc, trc, shape)
        else:
            rawgs = jax.random.normal(key, shape)
        new_fields = rawgs * sigma + mu
        new_logdens = log_dens_gaussian(new_fields, mu, sigma).sum((-1,-2,-3))
        return state, (new_fields, new_logdens)
    
    def init(key, params, batch_size):
        return sample(key, params, jnp.zeros((batch_size, *sample_shape)))[0]

    def refresh(state, params):
        return state

    return MCSampler(sample, init, refresh)


def make_metropolis(prop: Propagator, beta=1., sigma=0.05, steps=5):
    sample_shape = get_shape(prop)
    raw_logdens_fn = lambda p, x: beta * prop.sign_logov(p, x)[1]
    logdens_fn = jax.vmap(raw_logdens_fn, in_axes=(None, 0))

    def step(key, params, state):
        x1, ld1 = state
        gkey, ukey = jax.random.split(key)
        x2 = x1 + sigma * jax.random.normal(gkey, shape=x1.shape)
        ld2 = logdens_fn(params, x2)
        ratio = ld2 - ld1
        rnd = jnp.log(jax.random.uniform(ukey, shape=ratio.shape))
        cond = ratio > rnd
        x_new = jnp.where(cond[...,None,None,None], x2, x1)
        ld_new = jnp.where(cond, ld2, ld1)
        acc_rate = cond.mean()
        return (x_new, ld_new), acc_rate

    def sample(key, params, state):
        multi_step = make_multistep_fn(step, steps, concat=False)
        new_state, acc_rate = multi_step(key, params, state)
        new_fields, new_logdens = new_state
        return new_state, (new_fields, new_logdens)

    def init(key, params, batch_size):
        sigma, mu = 1., 0.
        fields = jax.random.normal(key, (batch_size, *sample_shape)) * sigma + mu
        logdens = logdens_fn(params, fields)
        return (fields, logdens)

    def refresh(state, params):
        fields, ld_old = state
        ld_new = logdens_fn(params, fields)
        return (fields, ld_new)

    return MCSampler(sample, init, refresh)