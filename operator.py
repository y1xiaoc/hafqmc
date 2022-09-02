import jax
from jax import lax
from jax import numpy as jnp
from flax import linen as nn
from typing import Optional, Sequence, Union
from functools import partial

from .utils import _t_real, _t_cplx
from .utils import fix_init, symmetrize, Serial, cmult, make_expm_apply
from .hamiltonian import _align_rdm, calc_rdm


class OneBody(nn.Module):
    init_hmf : jnp.ndarray
    parametrize : bool = False
    init_random : float = 0.
    hermite_out : bool = False
    dtype: Optional[jnp.dtype] = None
    expm_option : Union[str, tuple] = ()

    def setup(self):
        if self.parametrize:
            self.hmf = self.param("hmf", fix_init, 
                self.init_hmf, self.dtype, self.init_random)
        else:
            self.hmf = self.init_hmf

    def __call__(self, step):
        hmf = symmetrize(self.hmf) if self.hermite_out else self.hmf
        hmf = cmult(step, hmf)
        return hmf
    
    @property
    def expm_apply(self):
        _expm_op = self.expm_option
        _expm_op = (_expm_op,) if isinstance(_expm_op, str) else _expm_op
        return _warp_spin(make_expm_apply(*_expm_op))


class AuxField(nn.Module):
    init_vhs : jnp.ndarray
    trial_wfn : Optional[jnp.ndarray] = None
    parametrize : bool = False
    init_random : float = 0.
    hermite_out : bool = False
    dtype: Optional[jnp.dtype] = None
    expm_option : Union[str, tuple] = ()

    def setup(self):
        if self.parametrize:
            self.vhs = self.param("vhs", fix_init, 
                self.init_vhs, self.dtype, self.init_random)
        else:
            self.vhs = self.init_vhs
        self.nhs = self.init_vhs.shape[0]
        self.trial_rdm = (calc_rdm(self.trial_wfn, self.trial_wfn) 
            if self.trial_wfn is not None else None)

    def __call__(self, step, fields, trdm=None):
        vhs = symmetrize(self.vhs) if self.hermite_out else self.vhs
        log_weight = - 0.5 * (fields ** 2).sum()
        if self.trial_rdm is not None:
            vhs, vbar0 = meanfield_subtract(vhs, self.trial_rdm)
            fields += step * vbar0
        if trdm is not None:
            _, vbar = meanfield_subtract(vhs, lax.stop_gradient(trdm), 0.1)
            fshift = step * vbar
            log_weight += - fields @ fshift - 0.5 * (fshift ** 2).sum()
            fields += fshift
        vhs_sum = jnp.tensordot(fields, vhs, axes=1)
        vhs_sum = cmult(step, vhs_sum)
        return vhs_sum, log_weight
    
    @property
    def expm_apply(self):
        _expm_op = self.expm_option
        _expm_op = (_expm_op,) if isinstance(_expm_op, str) else _expm_op
        return _warp_spin(make_expm_apply(*_expm_op))


class AuxFieldNet(AuxField):
    hidden_sizes : Optional[Sequence[int]] = None
    actv_fun : str = "gelu"
    zero_init : bool = True
    mod_density: bool = False

    def setup(self):
        super().setup()
        nhs = self.nhs
        last_init =nn.zeros if self.zero_init else nn.initializers.lecun_normal()
        outdim = nhs+1 if self.mod_density else nhs
        self.last_dense = nn.Dense(outdim, param_dtype=self.dtype, 
                                   kernel_init=last_init, bias_init=nn.zeros)
        if self.hidden_sizes:
            inner_init = nn.initializers.orthogonal(scale=1., column_axis=-1)
            self.network = Serial(
                [nn.Dense(
                    ls if ls and ls > 0 else nhs, 
                    param_dtype = _t_real,
                    kernel_init = inner_init,
                    bias_init = nn.zeros) 
                 for ls in self.hidden_sizes],
                skip_cxn = True,
                actv_fun = self.actv_fun)
        else:
            self.network = None
        
    def __call__(self, step, fields, trdm=None):
        vhs = symmetrize(self.vhs) if self.hermite_out else self.vhs
        log_weight = - 0.5 * (fields ** 2).sum()
        tmp = fields
        if self.network is not None:
            tmp = self.network(tmp)
        tmp = self.last_dense(tmp)
        nfields = fields[:self.nhs] + tmp[:self.nhs]
        if self.mod_density:
            log_weight -= tmp[-1]
        if self.trial_rdm is not None:
            vhs, vbar0 = meanfield_subtract(vhs, self.trial_rdm)
            nfields += step * vbar0
        if trdm is not None:
            _, vbar = meanfield_subtract(vhs, lax.stop_gradient(trdm), 0.1)
            fshift = step * vbar
            log_weight += - nfields @ fshift - 0.5 * (fshift ** 2).sum()
            nfields += fshift
        vhs_sum = jnp.tensordot(nfields, vhs, axes=1)
        vhs_sum = cmult(step, vhs_sum)
        return vhs_sum, log_weight


def meanfield_subtract(vhs, rdm, cutoff=None):
    if rdm.ndim == 3:
        rdm = rdm.sum(0)
    nao = vhs.shape[-1]
    if rdm.shape[-1] == nao * 2:
        rdm = rdm[:nao, :nao] + rdm[nao:, nao:]
    nelec = lax.stop_gradient(rdm).trace().real
    vbar = jnp.einsum("kpq,pq->k", vhs, rdm)
    if cutoff is not None:
        cutoff *= vbar.shape[-1]
        vbar = vbar / (jnp.maximum(jnp.linalg.norm(vbar), cutoff) / cutoff)
    vhs = vhs - vbar.reshape(-1,1,1) * jnp.eye(vhs.shape[-1]) / nelec
    return vhs, vbar


def _warp_spin(fun_expm):
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