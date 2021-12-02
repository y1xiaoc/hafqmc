from flax.linen.module import init
import jax
from jax import lax
from jax import numpy as jnp
from flax import linen as nn
from jax.numpy import ndarray
import dataclasses
from typing import Sequence, Union, Tuple, Optional

from .utils import _t_real, _t_cplx
from .utils import parse_bool, ensure_mapping
from .utils import fix_init
from .utils import pack_spin, unpack_spin
from .utils import expm_apply
from .propagator import Propagator


class Ansatz(nn.Module):
    init_wfn: Union[ndarray, Tuple[ndarray, ndarray]]
    wfn_optim: bool = False
    wfn_random: float = 0.
    wfn_complex: bool = False
    propagators: Sequence[Propagator] = dataclasses.field(default_factory=list)

    def fields_shape(self):
        return tuple(p.fields_shape() for p in self.propagators)

    def setup(self):
        wfn = self.init_wfn 
        if not (isinstance(wfn, (tuple, list)) or wfn.ndim >= 3):
            wfn = (wfn, wfn[:,:0])
        _dtype = _t_cplx if self.wfn_complex else _t_real
        self.wfn = ((self.param("wfn_a", fix_init, wfn[0], _dtype, self.wfn_random),
                     self.param("wfn_b", fix_init, wfn[1], _dtype, self.wfn_random))
                    if self.wfn_optim else wfn)
 
    def __call__(self, fields):
        if isinstance(fields, ndarray):
            fields = (fields,)
        assert (jax.tree_map(jnp.shape, fields) 
                == jax.tree_map(tuple, self.fields_shape()))
        wfn = self.wfn
        log_weight = 0.
        for prop, flds in zip(self.propagators, fields):
            wfn, logw = prop(wfn, flds)
            log_weight += logw
        return wfn, log_weight


class BraKet(nn.Module):
    ansatz : Ansatz
    trial : Optional[Ansatz] = None

    def fields_shape(self):
        if self.trial is None:
            return jax.tree_map(lambda s: jnp.array((2, *s)), 
                    self.ansatz.fields_shape())
        else:
            return (self.trial.fields_shape(), 
                    self.ansatz.fields_shape())

    def __call__(self, fields):
        return None