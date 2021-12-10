import jax
import numpy as onp
from jax import numpy as jnp
from flax import linen as nn
from jax.numpy import ndarray
import dataclasses
from functools import partial
from typing import Sequence, Union, Tuple, Optional

from .utils import _t_real, _t_cplx
from .utils import fix_init, parse_bool
from .propagator import Propagator
from .hamiltonian import calc_slov


class Ansatz(nn.Module):
    init_wfn: Union[ndarray, Tuple[ndarray, ndarray]]
    wfn_param: bool = False
    wfn_random: float = 0.
    wfn_complex: bool = False
    propagators: Sequence[Optional[Propagator]] = dataclasses.field(default_factory=list)

    @nn.nowrap
    @classmethod
    def create(cls, hamiltonian, init_wfn, propagators=None, **kwargs):
        if propagators is None:
            ansatz_props = [Propagator.create(hamiltonian, init_wfn, **kwargs)]
            ansatz_kwargs = dict(
                wfn_param = parse_bool("wfn", kwargs['parametrize']),
                wfn_random = kwargs['init_random'],
                wfn_complex = kwargs['use_complex'])
        else:
            if not isinstance(propagators, (list, tuple)):
                propagators = [propagators]
            ansatz_kwargs = kwargs
            ansatz_props = []
            for popt in propagators:
                prop = (Propagator.create(hamiltonian, init_wfn, **popt) 
                        if popt is not None else None)
                ansatz_props.append(prop)
        return cls(init_wfn, propagators=ansatz_props, **ansatz_kwargs)

    @nn.nowrap
    def fields_shape(self):
        shapes = []
        prev_p = None
        for p in self.propagators:
            if p is None:
                p = prev_p
            shapes.append(p.fields_shape())
            prev_p = p
        return tuple(shapes)

    def setup(self):
        wfn = self.init_wfn 
        if not (isinstance(wfn, (tuple, list)) or wfn.ndim >= 3):
            wfn = (wfn, wfn[:,:0])
        _dtype = _t_cplx if self.wfn_complex else _t_real
        self.wfn = ((self.param("wfn_a", fix_init, wfn[0], _dtype, self.wfn_random),
                     self.param("wfn_b", fix_init, wfn[1], _dtype, self.wfn_random))
                    if self.wfn_param else wfn)
 
    def __call__(self, fields, *, keep_last=0):
        if isinstance(fields, ndarray):
            fields = (fields,)
        assert (jax.tree_map(jnp.shape, fields) 
                == jax.tree_map(tuple, self.fields_shape()))
        wfn = self.wfn
        log_weight = 0.
        if keep_last > 0:
            results = []
        prev_prop = None
        for ii, (prop, flds) in enumerate(zip(self.propagators, fields)):
            if keep_last > 0 and ii > len(self.propagators) - keep_last:
                results.append((wfn, log_weight))
            if prop is None:
                prop = prev_prop
            wfn, logw = prop(wfn, flds)
            log_weight += logw
            prev_prop = prop
        if keep_last > 0:
            results.append((wfn, log_weight))
            wfn, log_weight = jax.tree_map(lambda *xs: jnp.stack(xs, 0), *results)
        return wfn, log_weight


class BraKet(nn.Module):
    ansatz : Ansatz
    trial : Optional[Ansatz] = None

    @nn.nowrap
    def fields_shape(self):
        if self.trial is None:
            return jax.tree_map(lambda s: onp.array((2, *s)), 
                    self.ansatz.fields_shape())
        else:
            return (self.trial.fields_shape(), 
                    self.ansatz.fields_shape())

    def __call__(self, fields, *, keep_last=0):
        if self.trial is None:
            out = jax.vmap(partial(self.ansatz, keep_last=keep_last))(fields)
            bra_out = jax.tree_map(lambda x: x[0], out)
            ket_out = jax.tree_map(lambda x: x[1], out)
        else:
            keep_last = min(keep_last, 
                len(self.trial.propagators)+1, len(self.ansatz.propagators)+1)
            bra_out = self.trial(fields[0], keep_last=keep_last)
            ket_out = self.ansatz(fields[1], keep_last=keep_last)
        return bra_out, ket_out

    def sign_logov(self, fields):
        (bra, bra_lw), (ket, ket_lw) = self(fields, keep_last=0)
        sign, logov = calc_slov(bra, ket)
        return sign, logov + bra_lw + ket_lw

