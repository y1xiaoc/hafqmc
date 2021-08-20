import jax
from jax import numpy as jnp
from flax import linen as nn
from typing import Optional, Sequence
from functools import partial

from .utils import _t_real, _t_cplx
from .utils import fix_init, make_hermite, Serial


class OneBody(nn.Module):
    init_hmf : jnp.ndarray
    parametrize : bool = False
    dtype: Optional[jnp.dtype] = None
    
    def setup(self):
        if self.parametrize:
            self.hmf = self.param("hmf", fix_init, self.init_hmf, self.dtype)
        else:
            self.hmf = self.init_hmf

    def __call__(self):
        return make_hermite(self.hmf)


class AuxField(nn.Module):
    init_vhs : jnp.ndarray
    parametrize : bool = False
    dtype: Optional[jnp.dtype] = None

    def setup(self):
        if self.parametrize:
            self.vhs = self.param("vhs", fix_init, self.init_vhs, self.dtype)
        else:
            self.vhs = self.init_vhs
        self.nhs = self.init_vhs.shape[0]

    def __call__(self, fields):
        vhs_sum = jnp.tensordot(fields, self.vhs, axes=1)
        log_weight = - 0.5 * (fields.conj() @ fields)
        return make_hermite(vhs_sum), log_weight


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
            self.network = Serial(
                [nn.Dense(
                    ls if ls and ls > 0 else nhs, 
                    dtype = _t_real,
                    kernel_init = nn.initializers.lecun_normal(dtype=_t_real),
                    bias_init = partial(nn.zeros, dtype=_t_real)) 
                 for ls in self.hidden_sizes],
                skip_cxn = True,
                actv_fun = self.actv_fun)
        else:
            self.network = None
        
    def __call__(self, fields):
        tmp = fields
        if self.network is not None:
            tmp = self.network(tmp)
        tmp = self.last_dense(tmp)
        fields = fields[:self.nhs] + tmp[:-1]
        vhs_sum = jnp.tensordot(fields, self.vhs, axes=1)
        log_weight = - 0.5 * (fields.conj() @ fields) - tmp[-1]
        return make_hermite(vhs_sum), log_weight
