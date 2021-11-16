import jax
from jax import lax
from jax import numpy as jnp
from jax import scipy as jsp
from flax import linen as nn
from typing import Dict, Sequence, Union, Callable, Any, Optional
from functools import partial
import dataclasses
import pickle
import time


_t_real = jnp.float64
_t_cplx = jnp.complex128


Array = Any
PyTree = Any


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
    def __post_init__(self):
        for nm, fn in (("vmap", jax.vmap), ("pmap", jax.pmap),
                       ("pmax", pmax_if_pmap), ("pmin", pmin_if_pmap),
                       ("psum", psum_if_pmap), ("pmean", pmean_if_pmap)):
            object.__setattr__(self, nm, partial(fn, axis_name=self.name))
        for nm in ("max", "min", "sum", "mean"):
            jnp_fn = getattr(jnp, nm)
            pax_fn = getattr(self, f"p{nm}")
            all_fn = lambda x: pax_fn(jnp_fn(x))
            object.__setattr__(self, f"all_{nm}", all_fn)

PMAP_AXIS_NAME = "_pmap_axis"
paxis = PAxis(PMAP_AXIS_NAME)


_EXPMA_S = 1
_EXPMA_M = 6

def expm_apply_loop(A, B):
    # n = A.shape[-1]
    # mu = jnp.trace(A, axis1=-1, axis2=-2) / n
    # eta = jnp.expand_dims(jnp.exp(mu), -1)
    # A = A - mu * jnp.identity(n, dtype=A.dtype)
    F = B
    for _ in range(_EXPMA_S):
        for n in range(1, _EXPMA_M + 1):
            B = A @ B / (_EXPMA_S * n)
            F = F + B
        B = F
    return F # * eta

def expm_apply_scan(A, B):
    # n = A.shape[-1]
    # mu = jnp.trace(A, axis1=-1, axis2=-2) / n
    # eta = jnp.expand_dims(jnp.exp(mu), -1)
    # A = A - mu * jnp.identity(n, dtype=A.dtype)
    ns = jnp.arange(1., _EXPMA_M+1, dtype=A.dtype)
    def _loop_m(B_and_F, n):
        B, F = B_and_F
        B = A @ B / (_EXPMA_S * n)
        return (B, F + B), None
    def _loop_s(B, _):
        (_, B), _ = lax.scan(_loop_m, (B, B), ns)
        return B, None
    B, _ = lax.scan(_loop_s, B, None, _EXPMA_S)
    return B # * eta

def expm_apply_exact(A, B):
    exp_A = jsp.linalg.expm(A)
    return exp_A @ B

expm_apply = expm_apply_scan


def cmult(x1, x2):
    return ((x1.real * x2.real - x1.imag * x2.imag) 
        + 1j * (x1.imag * x2.real + x1.real * x2.imag))


_INIT_RAND_MULTIPLICATIVE = False
def fix_init(key, value, dtype=None, random=0.):
    value = jnp.asarray(value, dtype=dtype)
    if random <= 0.:
        return value
    else:
        perturb = jax.random.truncated_normal(
            key, -2, 2, value.shape, _t_real) * random
        if _INIT_RAND_MULTIPLICATIVE:
            return value * (1 + perturb)
        else:
            return value + perturb


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
    if isinstance(inputs, str) and inputs.lower() in ("all", "true"):
        inputs = True
    if isinstance(inputs, str) and inputs.lower() in ("none", "false"):
        inputs = False
    if isinstance(inputs, bool):
        for key in keys:
            res_dict[key] = inputs
    else:
        for key in keys:
            res_dict[key] = key in inputs
    return res_dict

def ensure_mapping(obj, default_key="name"):
    try:
        return dict(**obj)
    except TypeError:
        return {default_key: obj}


def save_pickle(filename, data):
    with open(filename, 'wb') as file:
        pickle.dump(data, file)

def load_pickle(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)


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
            if self.skip_cxn:
                if x.shape[-1] >= tmp.shape[-1]:
                    x = x[...,:tmp.shape[-1]] + tmp
                else:
                    x = tmp.at[...,:x.shape[-1]].add(x)
            else:
                x = tmp
        return x


class Printer:

    def __init__(self, 
                 field_format: Dict[str, Optional[str]], 
                 time_format: Optional[str]=None,
                 **print_kwargs):
        all_format = {**field_format, "time": time_format}
        all_format = {k: v for k, v in all_format.items() if v is not None}
        self.fields = all_format
        self.header = "\t".join(self.fields.keys())
        self.format = "\t".join(f"{{{k}:{v}}}" for k, v in self.fields.items())
        self.kwargs = print_kwargs
        self.tick = time.perf_counter()

    def print_header(self, prefix: str = ""):
        print(prefix+self.header, **self.kwargs)

    def print_fields(self, field_dict: Dict[str, Any], prefix: str = ""):
        output = self.format.format(**field_dict, time=time.perf_counter()-self.tick)
        print(prefix+output, **self.kwargs)

    def reset_timer(self):
        self.tick = time.perf_counter()