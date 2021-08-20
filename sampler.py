import jax
from jax import lax
from jax import numpy as jnp
from functools import partial
from .propagator import Propagator


def log_dens_gaussian(x, mu=0., sigma=1.):
    """unnormalized log density of Gaussian distribution"""
    return -0.5 * ((x - mu) / sigma) ** 2


def get_shape(prop: Propagator):
    """"calculate needed field shape from propagator: [2 x nts x nsite]"""
    nts = len(prop.init_tsteps)
    nsite = prop.init_vhs.shape[0]
    return (2, nts, nsite)


def make_multistep(sample_fn, nstep, concat=False):

    def sample_multi(params, key, fields, aux_data=None):
        inner = lambda c,i: sample_fn(params, *c)
        new_carry, data = lax.scan(inner, (key, fields, aux_data), None, nstep)
        if concat:
            data = jax.tree_map(jnp.concatenate, data)
        return new_carry, data
    
    return sample_multi


def make_gaussian(prop: Propagator, mu=0., sigma=1.):
    sample_shape = get_shape(prop)

    @jax.jit
    def sample(params, key, fields, aux_data=None):
        key, subkey = jax.random.split(key)
        nbatch = fields.shape[0]
        new_fields = jax.random.normal(subkey, (nbatch, *sample_shape)) * sigma + mu
        new_logdens = log_dens_gaussian(new_fields, mu, sigma).sum((-1,-2,-3))
        return (key, new_fields, aux_data), (new_fields, new_logdens)
    
    def init(params, key, batch_size, **kwargs):
        return sample(params, key, jnp.zeros((batch_size, *sample_shape)), None)[0]

    return sample, init