import jax
from jax import lax
from jax import numpy as jnp
from flax import linen as nn
from typing import Sequence, Union, Callable
from functools import partial
import dataclasses


_t_real = jnp.float64
_t_cplx = jnp.complex128


def wrap_if_pmap(p_func):

    def p_func_if_pmap(obj, axis_name):
        try:
            jax.core.axis_frame(axis_name)
            return p_func(obj, axis_name)
        except NameError:
            return obj

    return p_func_if_pmap


pmax_if_pmap = wrap_if_pmap(lax.pmax)
pmin_if_pmap = wrap_if_pmap(lax.pmin)
psum_if_pmap = wrap_if_pmap(lax.psum)
pmean_if_pmap = wrap_if_pmap(lax.pmean)


@dataclasses.dataclass(frozen=True)
class PAxis:
    name  : str
    vmap  : Callable = dataclasses.field(init=False)
    pmap  : Callable = dataclasses.field(init=False)
    pmax  : Callable = dataclasses.field(init=False)
    pmin  : Callable = dataclasses.field(init=False)
    psum  : Callable = dataclasses.field(init=False)
    pmean : Callable = dataclasses.field(init=False)

    def __post_init__(self):
        for nm, fn in (("vmap", jax.vmap), ("pmap", jax.pmap),
                       ("pmax", pmax_if_pmap), ("pmin", pmin_if_pmap),
                       ("psum", psum_if_pmap), ("pmean", pmean_if_pmap)):
            object.__setattr__(self, nm, partial(fn, axis_name=self.name))


PMAP_AXIS_NAME = "_pmap_axis"
paxis = PAxis(PMAP_AXIS_NAME)


_EXPMA_S = 2
_EXPMA_M = 10
def expm_apply(A, B):
    n = A.shape[-1]
    mu = jnp.trace(A, axis1=-1, axis2=-2) / n
    eta = jnp.expand_dims(jnp.exp(mu), -1)
    A = A - mu * jnp.identity(n, dtype=A.dtype)
    F = B
    for _ in range(_EXPMA_S):
        for n in range(1, _EXPMA_M + 1):
            B = A @ B / (_EXPMA_S * n)
            F = F + B
        B = F
    return eta * F


def fix_init(key, value, dtype=None):
    return jnp.asarray(value, dtype=dtype)


def make_hermite(A):
    return 0.5 * (A.conj().T + A)


def pack_spin(wfn):
    if not (isinstance(wfn, (tuple, list)) or wfn.ndim >= 3):
        return wfn, wfn.shape[-1]
    w_up, w_dn = wfn
    n_up, n_dn = w_up.shape[-1], w_dn.shape[-1]
    w_packed = jnp.concatenate((w_up, w_dn), -1)
    return w_packed, (n_up, n_dn)

def unpack_spin(wfn, nelec):
    if isinstance(nelec, int):
        return wfn
    n_up, n_dn = nelec
    w_up = wfn[:, :n_up]
    w_dn = wfn[:, n_up : n_up+n_dn]
    return (w_up, w_dn)


def parse_activation(name, **kwargs):
    if not isinstance(name, str):
        return name
    raw_fn = getattr(nn, name)
    return partial(raw_fn, **kwargs)

def parse_bool(keys, inputs):
    res_dict = {}
    if isinstance(inputs, bool):
        for key in keys:
            res_dict[key] = inputs
    else:
        for key in keys:
            res_dict[key] = key in inputs
    return res_dict


class Serial(nn.Module):
    layers : Sequence[nn.Module]
    skip_cxn : bool = True
    actv_fun : Union[str, Callable] = "gelu"

    @nn.compact
    def __call__(self, x):
        actv = parse_activation(self.actv_fun)
        for i, lyr in enumerate(self.layers):
            tmp = lyr(x)
            if i != len(self.layers) - 1:
                tmp = actv(tmp)
            if self.skip_cxn and x.shape[-1] == tmp.shape[-1]:
                x = x + tmp
            else:
                x = tmp
        return x
