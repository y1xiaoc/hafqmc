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
    init: Callable[..., State]
    def __call__(self, *args, **kwargs):
        """Call the sample function. See `self.sample` for details."""
        return self.sample(*args, **kwargs)


def make_multistep_fn(sample_fn, nstep, concat=False):
    @jax.jit
    def multi_sample(key, params, state):
        inner = lambda s,k: sample_fn(k, params, s)
        keys = jax.random.split(key, nstep)
        new_state, data = lax.scan(inner, state, keys)
        if concat:
            data = jax.tree_map(jnp.concatenate, data)
        return new_state, data
    return multi_sample

def make_multistep(sampler, nstep, concat=False):
    sample_fn, init_fn = sampler
    multisample_fn = make_multistep_fn(sample_fn, nstep, concat)
    return MCSampler(multisample_fn, init_fn)


def make_sampler(prop: Propagator, name: str, **kwargs):
    name = name.lower()
    if name == "gaussian":
        maker = make_gaussian
    else:
        raise NotImplementedError(f"unsupported sampler type: {name}")
    return maker(prop, **kwargs)


def make_gaussian(prop: Propagator, mu=0., sigma=1.):
    sample_shape = get_shape(prop)

    @jax.jit
    def sample(key, params, state):
        nbatch = state.shape[0]
        new_fields = jax.random.normal(key, (nbatch, *sample_shape)) * sigma + mu
        new_logdens = log_dens_gaussian(new_fields, mu, sigma).sum((-1,-2,-3))
        return state, (new_fields, new_logdens)
    
    def init(key, params, batch_size, burn_in=0, **kwargs):
        return sample(key, params, jnp.zeros((batch_size, *sample_shape)))[0]

    return MCSampler(sample, init)