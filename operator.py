import jax
from jax import numpy as jnp
from flax import linen as nn
from typing import Optional, Sequence
from functools import partial

from .utils import _t_real, _t_cplx
from .utils import fix_init, make_hermite, Serial, cmult


class OneBody(nn.Module):
    init_hmf : jnp.ndarray
    parametrize : bool = False
    init_random : float = 0.
    hermite_out : bool = False
    dtype: Optional[jnp.dtype] = None
    
    def setup(self):
        if self.parametrize:
            self.hmf = self.param("hmf", fix_init, 
                self.init_hmf, self.dtype, self.init_random)
        else:
            self.hmf = self.init_hmf

    def __call__(self, step):
        hmf = make_hermite(self.hmf) if self.hermite_out else self.hmf
        hmf = cmult(step, hmf)
        return hmf


class AuxField(nn.Module):
    init_vhs : jnp.ndarray
    trial_rdm : Optional[jnp.ndarray] = None
    parametrize : bool = False
    init_random : float = 0.
    hermite_out : bool = False
    dtype: Optional[jnp.dtype] = None

    def setup(self):
        if self.parametrize:
            self.vhs = self.param("vhs", fix_init, 
                self.init_vhs, self.dtype, self.init_random)
        else:
            self.vhs = self.init_vhs
        self.nhs = self.init_vhs.shape[0]

    def __call__(self, step, fields):
        vhs = make_hermite(self.vhs) if self.hermite_out else self.vhs
        log_weight = - 0.5 * (fields.conj() @ fields)
        if self.trial_rdm is not None:
            vhs, vbar = meanfield_subtract(vhs, self.trial_rdm)
            fields += step * vbar
        vhs_sum = jnp.tensordot(fields, vhs, axes=1)
        vhs_sum = cmult(step, vhs_sum)
        return vhs_sum, log_weight


class AuxFieldNet(AuxField):
    hidden_sizes : Optional[Sequence[int]] = None
    actv_fun : str = "gelu"
    zero_init : bool = True

    def setup(self):
        super().setup()
        nhs = self.nhs
        last_init = (partial(nn.zeros, dtype=self.dtype) if self.zero_init 
                     else nn.initializers.lecun_normal(dtype=_t_real))
        self.last_dense = nn.Dense(nhs+1, dtype=self.dtype, kernel_init=last_init, 
                                   bias_init=partial(nn.zeros, dtype=self.dtype))
        if self.hidden_sizes:
            inner_init = nn.initializers.orthogonal(
                scale=1., column_axis=-1, dtype=_t_real)
            self.network = Serial(
                [nn.Dense(
                    ls if ls and ls > 0 else nhs, 
                    dtype = _t_real,
                    kernel_init = inner_init,
                    bias_init = partial(nn.zeros, dtype=_t_real)) 
                 for ls in self.hidden_sizes],
                skip_cxn = True,
                actv_fun = self.actv_fun)
        else:
            self.network = None
        
    def __call__(self, step, fields):
        vhs = make_hermite(self.vhs) if self.hermite_out else self.vhs
        log_weight = - 0.5 * (fields.conj() @ fields)
        tmp = fields
        if self.network is not None:
            tmp = self.network(tmp)
        tmp = self.last_dense(tmp)
        log_weight -= tmp[-1]
        nfields = fields[:self.nhs] + tmp[:-1]
        if self.trial_rdm is not None:
            vhs, vbar = meanfield_subtract(vhs, self.trial_rdm)
            nfields += step * vbar
        vhs_sum = jnp.tensordot(nfields, vhs, axes=1)
        vhs_sum = cmult(step, vhs_sum)
        return vhs_sum, log_weight


def meanfield_subtract(vhs, rdm):
    if rdm.ndim == 3:
        rdm = rdm.sum(0)
    nelec = rdm.trace()
    vbar = jnp.einsum("kpq,pq->k", vhs, rdm)
    vhs = vhs - vbar.reshape(-1,1,1) * jnp.eye(vhs.shape[-1]) / nelec
    return vhs, vbar