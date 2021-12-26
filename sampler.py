import jax
from jax import lax
from jax import numpy as jnp
from functools import partial
from typing import NamedTuple, Callable, Tuple

from .ansatz import BraKet
from .utils import PyTree, Array


def logd_gaussian(x, mu=0., sigma=1.):
    """unnormalized log density of Gaussian distribution"""
    return -0.5 * jnp.abs((x - mu) / sigma) ** 2


def ravel_shape(target_shape):
    from jax.flatten_util import ravel_pytree
    tmp = jax.tree_map(jnp.zeros, target_shape)
    flat, unravel_fn = ravel_pytree(tmp)
    return flat.size, unravel_fn


def tree_where(condition, x, y):
    return jax.tree_map(partial(jnp.where, condition), x, y)


def mh_select(key, ratio, state1, state2):
    rnd = jnp.log(jax.random.uniform(key, shape=ratio.shape))
    cond = ratio > rnd
    new_state = jax.vmap(tree_where)(cond, state2, state1)
    return new_state, cond


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


def make_sampler(braket: BraKet, name: str, **kwargs):
    maker = choose_sampler_maker(name)
    logdens_fn = lambda p, x: braket.apply(p, x, method=braket.sign_logov)[1]
    fields_shape = braket.fields_shape()
    return maker(logdens_fn, fields_shape, **kwargs)
    

def choose_sampler_maker(name: str) -> Callable[..., MCSampler]:
    name = name.lower()
    if name in ("direct", "gaussian"):
        return make_gaussian
    if name in ("metropolis", "mcmc", "mh"):
        return make_metropolis
    if name in ("langevin", "mala"):
        return make_langevin
    if name in ("hamiltonian", "hybrid", "hmc"):
        return make_hamiltonian
    raise NotImplementedError(f"unsupported sampler type: {name}")


def make_gaussian(logdens_fn, fields_shape, mu=0., sigma=1., truncate=None):
    fsize, unravel = ravel_shape(fields_shape)
    batch_unravel = jax.vmap(unravel, in_axes=0)

    def sample(key, params, state):
        nbatch = state.shape[0]
        shape = (nbatch, fsize)
        if truncate is not None:
            trc = jnp.abs(truncate)
            rawgs = jax.random.truncated_normal(key, -trc, trc, shape)
        else:
            rawgs = jax.random.normal(key, shape)
        new_fields = rawgs * sigma + mu
        new_logdens = logd_gaussian(new_fields, mu, sigma).sum(-1)
        return state, (batch_unravel(new_fields), new_logdens)
    
    def init(key, params, batch_size):
        return jnp.zeros((batch_size, 0))

    def refresh(state, params):
        return state

    return MCSampler(sample, init, refresh)


def make_metropolis(logdens_fn, fields_shape, beta=1., sigma=0.05, steps=5):
    fsize, unravel = ravel_shape(fields_shape)
    ravel_logd = lambda p, x: beta * logdens_fn(p, unravel(x))
    batch_unravel = jax.vmap(unravel, in_axes=0)
    batch_logd = jax.vmap(ravel_logd, in_axes=(None, 0))

    def step(key, params, state):
        x1, ld1 = state
        gkey, ukey = jax.random.split(key)
        x2 = x1 + sigma * jax.random.normal(gkey, shape=x1.shape)
        ld2 = batch_logd(params, x2)
        ratio = ld2 - ld1
        return mh_select(ukey, ratio, state, (x2, ld2))

    def sample(key, params, state):
        multi_step = make_multistep_fn(step, steps, concat=False)
        new_state, accepted = multi_step(key, params, state)
        new_fields, new_logdens = new_state
        return new_state, (batch_unravel(new_fields), new_logdens)

    def init(key, params, batch_size):
        sigma, mu = 1., 0.
        fields = jax.random.normal(key, (batch_size, fsize)) * sigma + mu
        logdens = batch_logd(params, fields)
        return (fields, logdens)

    def refresh(state, params):
        fields, ld_old = state
        ld_new = batch_logd(params, fields)
        return (fields, ld_new)

    return MCSampler(sample, init, refresh)


def make_langevin(logdens_fn, fields_shape, beta=1., tau=0.01, steps=5):
    fsize, unravel = ravel_shape(fields_shape)
    ravel_logd = lambda p, x: beta * logdens_fn(p, unravel(x))
    batch_unravel = jax.vmap(unravel, in_axes=0)
    logd_and_grad = jax.vmap(jax.value_and_grad(ravel_logd, 1), in_axes=(None, 0))

    # log transition probability q(x2|x1)
    def log_q(x2, x1, g1): 
        d = x2 - x1 - tau * g1
        norm = (d * d.conj()).real.sum(-1)
        return -1/(4*tau) * norm

    def step(key, params, state):
        x1, g1, ld1 = state
        gkey, ukey = jax.random.split(key)
        x2 = x1 + tau*g1 + jnp.sqrt(2*tau)*jax.random.normal(gkey, shape=x1.shape)
        ld2, g2 = logd_and_grad(params, x2)
        g2 = g2.conj() # handle complex grads, no influence for real case
        ratio = ld2 + log_q(x1, x2, g2) - ld1 - log_q(x2, x1, g1)
        return mh_select(ukey, ratio, state, (x2, g2, ld2))

    def sample(key, params, state):
        multi_step = make_multistep_fn(step, steps, concat=False)
        new_state, accepted = multi_step(key, params, state)
        new_fields, new_grads, new_logdens = new_state
        return new_state, (batch_unravel(new_fields), new_logdens)

    def init(key, params, batch_size):
        sigma, mu = 1., 0.
        fields = jax.random.normal(key, (batch_size, fsize)) * sigma + mu
        logdens, grads = logd_and_grad(params, fields)
        return (fields, grads.conj(), logdens)

    def refresh(state, params):
        fields, grads_old, ld_old = state
        ld_new, grads_new = logd_and_grad(params, fields)
        return (fields, grads_new.conj(), ld_new)

    return MCSampler(sample, init, refresh)


def make_hamiltonian(logdens_fn, fields_shape, beta=1., dt=0.1, length=1.):
    fsize, unravel = ravel_shape(fields_shape)
    ravel_logd = lambda p, x: beta * logdens_fn(p, unravel(x))
    batch_unravel = jax.vmap(unravel, in_axes=0)
    batch_logd = jax.vmap(ravel_logd, in_axes=(None, 0))
    logd_and_grad = jax.vmap(jax.value_and_grad(ravel_logd, 1), in_axes=(None, 0))

    def sample(key, params, state):
        gkey, ukey = jax.random.split(key)
        q1, f1, ld1 = state
        p1 = jax.random.normal(gkey, shape=q1.shape)
        grad_fn = jax.vmap(jax.grad(lambda x: -ravel_logd(params, x)))
        q2, p2, f2 = leapfrog_carry(q1, p1, f1, grad_fn, dt, round(length / dt))
        ld2 = batch_logd(params, q2)
        ratio = (logd_gaussian(-p2).sum(-1)+ld2) - (logd_gaussian(p1).sum(-1)+ld1)
        (qn, fn, ldn), accepted = mh_select(ukey, ratio, state, (q2, f2, ld2))
        return (qn, fn, ldn), (batch_unravel(qn), ldn)

    def init(key, params, batch_size):
        sigma, mu = 1., 0.
        fields = jax.random.normal(key, (batch_size, fsize)) * sigma + mu
        logdens, grads = logd_and_grad(params, fields)
        return (fields, grads.conj(), logdens)

    def refresh(state, params):
        fields, grads_old, ld_old = state
        ld_new, grads_new = logd_and_grad(params, fields)
        return (fields, grads_new.conj(), ld_new)

    return MCSampler(sample, init, refresh)


def leapfrog(q, p, grad_fn, dt, steps):
    # simple Euler integration step
    def int_step(carry, _):
        q, p = carry
        q += dt * p
        p -= dt * grad_fn(q)
        return (q, p), None
    # leapfrog by shifting half step
    p -= 0.5 * dt * grad_fn(q) # first half p
    (q, p), _ = lax.scan(int_step, (q, p), None, length=steps-1)
    q += dt * p # final whole step update of q
    p -= 0.5 * dt * grad_fn(q) # final half p
    return q, p


def leapfrog_carry(q, p, f, grad_fn, dt, steps):
    # simple Euler integration step
    def int_step(carry, _):
        q, p = carry
        q += dt * p
        p -= dt * grad_fn(q)
        return (q, p), None
    # leapfrog by shifting half step
    p += 0.5 * dt * f # first half p
    (q, p), _ = lax.scan(int_step, (q, p), None, length=steps-1)
    q += dt * p # final whole step update of q
    f = -grad_fn(q) # force at final point
    p += 0.5 * dt * f # final half p
    return q, p, f