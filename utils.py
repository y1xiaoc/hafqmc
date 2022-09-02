import jax
from jax import lax
from jax import numpy as jnp
from jax import scipy as jsp
from jax.tree_util import tree_map
from flax import linen as nn
from ml_collections import ConfigDict
from typing import Dict, Sequence, Union, Callable, Any, Optional
from functools import partial, reduce
import dataclasses
import pickle
import time


_t_real = jnp.float64
_t_cplx = jnp.complex128


Array = jnp.ndarray
PyTree = Any


def compose(*funcs):
    def c2(f, g):
        return lambda *a, **kw: f(g(*a, **kw))
    return reduce(c2, funcs)


def just_grad(x):
    return x - lax.stop_gradient(x)


def _T(x): 
    return jnp.swapaxes(x, -1, -2)

def _H(x): 
    return jnp.conj(_T(x))


def symmetrize(x): 
    return (x + _H(x)) / 2


def cmult(x1, x2):
    return ((x1.real * x2.real - x1.imag * x2.imag) 
        + 1j * (x1.imag * x2.real + x1.real * x2.imag))


@partial(jax.custom_jvp, nondiff_argnums=(1,))
def chol_qr(x, shift=None):
    *_, m, n = x.shape
    a = _H(x) @ x
    if shift is None:
        shift = 1.2e-15 * (m*n + n*(n+1)) * a.trace(0,-1,-2).max()
    r = jsp.linalg.cholesky(a + shift * jnp.eye(n, dtype=x.dtype), lower=False)
    q = lax.linalg.triangular_solve(r, x, left_side=False, lower=False)
    return q, r

@chol_qr.defjvp
def _chol_qr_jvp(shift, primals, tangents):
    x, = primals
    dx, = tangents
    *_, m, n = x.shape
    if m < n:
        raise NotImplementedError("Unimplemented case of QR decomposition derivative")
    q, r = chol_qr(x, shift=shift)
    dx_rinv = lax.linalg.triangular_solve(r, dx)
    qt_dx_rinv = jnp.matmul(_H(q), dx_rinv)
    qt_dx_rinv_lower = jnp.tril(qt_dx_rinv, -1)
    do = qt_dx_rinv_lower - _H(qt_dx_rinv_lower)  # This is skew-symmetric
    # The following correction is necessary for complex inputs
    I = lax.expand_dims(jnp.eye(n, dtype=do.dtype), range(qt_dx_rinv.ndim - 2))
    do = do + I * (qt_dx_rinv - jnp.real(qt_dx_rinv))
    dq = jnp.matmul(q, do - qt_dx_rinv) + dx_rinv
    dr = jnp.matmul(qt_dx_rinv - do, r)
    return (q, r), (dq, dr)


ExpmFnType = Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]

def make_expm_apply(method="scan", m=6, s=1) -> ExpmFnType:
    # the native python loop, slow compiling
    def expm_apply_loop(A, B):
        # n = A.shape[-1]
        # mu = jnp.trace(A, axis1=-1, axis2=-2) / n
        # eta = jnp.expand_dims(jnp.exp(mu), -1)
        # A = A - mu * jnp.identity(n, dtype=A.dtype)
        F = B
        for _ in range(s):
            for n in range(1, m + 1):
                B = A @ B / (s * n)
                F = F + B
            B = F
        return F # * eta
    # the jax scan version, faster compiling
    def expm_apply_scan(A, B):
        # n = A.shape[-1]
        # mu = jnp.trace(A, axis1=-1, axis2=-2) / n
        # eta = jnp.expand_dims(jnp.exp(mu), -1)
        # A = A - mu * jnp.identity(n, dtype=A.dtype)
        ns = jnp.arange(1., m+1., dtype=A.dtype)
        def _loop_m(B_and_F, n):
            B, F = B_and_F
            B = A @ B / (s * n)
            return (B, F + B), None
        def _loop_s(B, _):
            (_, B), _ = lax.scan(_loop_m, (B, B), ns)
            return B, None
        B, _ = lax.scan(_loop_s, B, None, s)
        return B # * eta
    # the exact verison, slow execution
    def expm_apply_exact(A, B):
        exp_A = jsp.linalg.expm(A)
        return exp_A @ B
    # choose the function from the method name
    if method == "loop":
        return expm_apply_loop
    if method == "scan":
        return expm_apply_scan
    if method == "exact":
        return expm_apply_exact
    raise ValueError(f"unknown expm_apply method type: {method}")

def warp_spin_expm(fun_expm: ExpmFnType) -> ExpmFnType:
    def new_expm(A, B):
        if A.shape[-1] == B.shape[-2]:
            return fun_expm(A, B)
        elif A.shape[-1]*2 == B.shape[-2]:
            nao = A.shape[-1]
            nelec = B.shape[-1]
            fB = B.reshape(2, nao, nelec).swapaxes(0,1).reshape(nao, 2*nelec)
            nfB = fun_expm(A, fB)
            nB = nfB.reshape(nao, 2, nelec).swapaxes(0,1).reshape(2*nao, nelec)
            return nB
        else:
            return fun_expm(A, B)
    return new_expm

DEFAULT_EXPM = warp_spin_expm(make_expm_apply("scan", 6, 1))


def make_moving_avg(decay=0.99, early_growth=True):
    def moving_avg(acc, new, i):
        if early_growth:
            iteration_decay = jnp.minimum(decay, (1.0 + i) / (10.0 + i))
        else:
            iteration_decay = decay
        updated_acc = iteration_decay * acc
        updated_acc += (1 - iteration_decay) * new
        return jax.lax.stop_gradient(updated_acc)
    return moving_avg


def ravel_shape(target_shape):
    from jax.flatten_util import ravel_pytree
    tmp = tree_map(jnp.zeros, target_shape)
    flat, unravel_fn = ravel_pytree(tmp)
    return flat.size, unravel_fn


def tree_where(condition, x, y):
    return tree_map(partial(jnp.where, condition), x, y)


def fix_init(key, value, dtype=None, random=0., rnd_additive=False):
    value = jnp.asarray(value, dtype=dtype)
    if random <= 0.:
        return value
    else:
        perturb = jax.random.truncated_normal(
            key, -2, 2, value.shape, _t_real) * random
        if rnd_additive:
            return value + perturb
        else:
            return value * (1 + perturb)


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


def block_spin(a, b, perturb=0.):
    p1 = jnp.eye(a.shape[-2], b.shape[-1]) * perturb
    p2 = jnp.eye(b.shape[-2], a.shape[-1]) * perturb
    return jnp.block([[a, p1],[p2, b]])


def parse_activation(name, **kwargs):
    if not isinstance(name, str):
        return name
    raw_fn = getattr(nn, name)
    return partial(raw_fn, **kwargs)


def parse_bool(keys, inputs):
    if isinstance(keys, str):
        return parse_bool((keys,), inputs)[keys]
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


def cfg_to_dict(cfg):
    if not isinstance(cfg, ConfigDict):
        return cfg
    return tree_map(cfg_to_dict, cfg.to_dict())

def cfg_to_yaml(cfg):
    import yaml
    from yaml import representer
    representer.Representer.add_representer(
        dict,
        lambda self, data: self.represent_mapping(
            'tag:yaml.org,2002:map', data, False))
    return yaml.dump(cfg_to_dict(cfg), default_flow_style=None)

def dict_to_cfg(cdict, **kwargs):
    if not isinstance(cdict, (dict, ConfigDict)):
        return cdict
    tree_type = (tuple, list)
    cfg = ConfigDict(cdict, **kwargs)
    for k, v in cfg.items():
        if isinstance(v, ConfigDict):
            cfg[k] = dict_to_cfg(v, **kwargs)
        if type(v) in tree_type:
            cfg[k] = type(v)(dict_to_cfg(vi, **kwargs) for vi in v)
    return cfg
    

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
            all_fn = compose(pax_fn, jnp_fn)
            object.__setattr__(self, f"all_{nm}", all_fn)

PMAP_AXIS_NAME = "_pmap_axis"
paxis = PAxis(PMAP_AXIS_NAME)