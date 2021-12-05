import jax
from jax import numpy as jnp
from flax import linen as nn
from jax.numpy import ndarray
import dataclasses
from functools import partial
from typing import Sequence, Union, Tuple, Optional

from .utils import _t_real, _t_cplx
from .utils import fix_init
from .propagator import Propagator
from .hamiltonian import calc_slov


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
 
    def __call__(self, fields, keep_last=0):
        if isinstance(fields, ndarray):
            fields = (fields,)
        assert (jax.tree_map(jnp.shape, fields) 
                == jax.tree_map(tuple, self.fields_shape()))
        wfn = self.wfn
        log_weight = 0.
        if keep_last > 0:
            results = []
        for ii, (prop, flds) in enumerate(zip(self.propagators, fields)):
            if keep_last > 0 and ii > len(self.propagators) - keep_last:
                results.append((wfn, log_weight))
            wfn, logw = prop(wfn, flds)
            log_weight += logw
        if keep_last > 0:
            results.append((wfn, log_weight))
            wfn, log_weight = jax.tree_multimap(lambda *xs: jnp.stack(xs, 0), *results)
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

    def __call__(self, fields, **kwargs):
        if self.trial is None:
            out = jax.vmap(partial(self.ansatz, **kwargs))(fields)
            bra_out = jax.tree_map(lambda x: x[0], out)
            ket_out = jax.tree_map(lambda x: x[1], out)
        else:
            bra_out = self.trial(fields[0], **kwargs)
            ket_out = self.ansatz(fields[1], **kwargs)
        return bra_out, ket_out

    def sign_logov(self, fields):
        (bra, bra_lw), (ket, ket_lw) = self(fields, keep_last=0)
        sign, logov = calc_slov(bra, ket)
        return sign, logov + bra_lw + ket_lw
