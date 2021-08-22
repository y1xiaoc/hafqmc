import jax 
import optax
from jax import lax
from jax import numpy as jnp

from .hamiltonian import Hamiltonian
from .propagator import Propagator
from .estimator import make_eval_total
from .sampler import make_sampler, make_multistep


def sign_penalty(s, factor=1., power=2.):
    return factor * (1-s) ** power


def make_loss(expect_fn, sign_factor=1., sign_power=2.):

    def loss(params, data):
        e_tot, aux = expect_fn(params, data)
        exp_s = aux["exp_s"]
        sp = sign_penalty(exp_s, sign_factor, sign_power)
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
    