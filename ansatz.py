import jax
import numpy as onp
from jax import numpy as jnp
from flax import linen as nn
from jax.numpy import ndarray
import dataclasses
from typing import Sequence, Union, Tuple, Optional

from .utils import _t_real, _t_cplx
from .utils import fix_init, parse_bool
from .propagator import Propagator
from .hamiltonian import calc_slov, _has_spin, _make_ghf


class Ansatz(nn.Module):
    init_wfn: Union[ndarray, Tuple[ndarray, ndarray]]
    wfn_param: bool = False
    wfn_random: float = 0.
    wfn_complex: bool = False
    propagators: Sequence[Propagator] = dataclasses.field(default_factory=list)

    @nn.nowrap
    @classmethod
    def create(cls, hamiltonian, propagators=None, **kwargs):
        if propagators is None:
            ansatz_props = [Propagator.create(hamiltonian, **kwargs)]
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
                prop = (Propagator.create(hamiltonian, **popt) 
                        if popt is not None else ansatz_props[-1])
                ansatz_props.append(prop)
        return cls(hamiltonian.wfn0, propagators=ansatz_props, **ansatz_kwargs)

    @nn.nowrap
    def fields_shape(self, max_prop=None):
        nprop = len(self.propagators) if max_prop is None else max_prop
        return tuple(p.fields_shape() for p in self.propagators[:nprop])

    def setup(self):
        wfn = self.init_wfn
        nao = wfn[0].shape[0] if _has_spin(wfn) else wfn.shape[0]
        if any(p.init_hmf.shape[-1] != nao for p in self.propagators):
            wfn = _make_ghf(wfn)
        _dtype = _t_cplx if self.wfn_complex else _t_real
        if not self.wfn_param:
            self.wfn = wfn
        elif _has_spin(wfn):
            self.wfn = (self.param("wfn_a", fix_init, wfn[0], _dtype, self.wfn_random),
                        self.param("wfn_b", fix_init, wfn[1], _dtype, self.wfn_random))
        else:
            self.wfn = self.param("wfn_a", fix_init, wfn, _dtype, self.wfn_random)
 
    def __call__(self, fields):
        if isinstance(fields, ndarray):
            fields = (fields,)
        assert (jax.tree_map(jnp.shape, fields) 
                == jax.tree_map(tuple, self.fields_shape(len(fields))))
        wfn = self.wfn
        log_weight = 0.
        for prop, flds in zip(self.propagators, fields):
            wfn, logw = prop(wfn, flds)
            log_weight += logw
        return wfn, log_weight


class BraKet(nn.Module):
    ansatz : Ansatz
    trial : Optional[Ansatz] = None

    @nn.nowrap
    def fields_shape(self, max_prop=None):
        lmp, rmp = (max_prop if isinstance(max_prop, (tuple, list))
                    else (max_prop, max_prop))
        if self.trial is None:
            return jax.tree_map(lambda s: onp.array((2, *s)), 
                    self.ansatz.fields_shape(rmp))
        else:
            return (self.trial.fields_shape(lmp), 
                    self.ansatz.fields_shape(rmp))

    def __call__(self, fields):
        if self.trial is None:
            out = jax.vmap(self.ansatz)(fields)
            bra_out = jax.tree_map(lambda x: x[0], out)
            ket_out = jax.tree_map(lambda x: x[1], out)
        else:
            bra_out = self.trial(fields[0])
            ket_out = self.ansatz(fields[1])
        return bra_out, ket_out

    def sign_logov(self, fields):
        (bra, bra_lw), (ket, ket_lw) = self(fields)
        sign, logov = calc_slov(bra, ket)
        return sign, logov + bra_lw + ket_lw

