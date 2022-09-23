import jax
from jax import lax
from jax import numpy as jnp
from flax import linen as nn
from typing import Optional, Sequence, Union
from functools import partial

from .utils import _t_real, _t_cplx
from .utils import fix_init, symmetrize, Serial, cmult
from .utils import warp_spin_expm, make_expm_apply
from .hamiltonian import _align_rdm, calc_rdm


class OneBody(nn.Module):
    init_hmf: jnp.ndarray
    parametrize: bool = False
    init_random: float = 0.
    hermite_out: bool = False
    dtype: Optional[jnp.dtype] = None
    expm_option: Union[str, tuple] = ()

    @property
    def nbasis(self):
        return self.init_hmf.shape[-1]

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
        return warp_spin_expm(make_expm_apply(*_expm_op))


class AuxField(nn.Module):
    init_vhs: jnp.ndarray
    trial_wfn: Optional[jnp.ndarray] = None
    parametrize: bool = False
    init_random: float = 0.
    hermite_out: bool = False
    dtype: Optional[jnp.dtype] = None
    expm_option: Union[str, tuple] = ()

    @property
    def nbasis(self):
        return self.init_vhs.shape[-1]

    @property
    def nfield(self):
        return self.init_vhs.shape[0]

    def setup(self):
        if self.parametrize:
            self.vhs = self.param("vhs", fix_init, 
                self.init_vhs, self.dtype, self.init_random)
        else:
            self.vhs = self.init_vhs
        self.nhs = self.init_vhs.shape[0]
        self.trial_rdm = (calc_rdm(self.trial_wfn, self.trial_wfn) 
            if self.trial_wfn is not None else None)

    def __call__(self, step, fields, curr_wfn=None):
        vhs = symmetrize(self.vhs) if self.hermite_out else self.vhs
        log_weight = - 0.5 * (fields ** 2).sum()
        if self.trial_rdm is not None:
            vhs, vbar0 = meanfield_subtract(vhs, self.trial_rdm)
            fields += step * vbar0
        # this dynamic shift is buggy, keep it here for reference
        if curr_wfn is not None and self.trial_wfn is not None:
            trdm = calc_rdm(self.trial_wfn, curr_wfn)
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
        return warp_spin_expm(make_expm_apply(*_expm_op))


class AuxFieldNet(AuxField):
    hidden_sizes: Optional[Sequence[int]] = None
    actv_fun: str = "gelu"
    zero_init: bool = True
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
        
    def __call__(self, step, fields, curr_wfn=None):
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
        # this dynamic shift is buggy
        if curr_wfn is not None and self.trial_wfn is not None:
            trdm = calc_rdm(self.trial_wfn, curr_wfn)
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


# below are classes and functions for pw basis

class OneBodyPW(nn.Module):
    init_ke: jnp.array
    kmask: Optional[jnp.array] = None
    parametrize: bool = False
    k_symmetric: bool = False
    init_random: float = 0.
    dtype: Optional[jnp.dtype] = None
    expm_option: Union[str, tuple] = ()

    @property
    def nbasis(self):
        return self.init_ke.shape[-1]

    def setup(self):
        if self.parametrize:
            if self.k_symmetric:
                raw_ke, self.ke_invidx = jnp.unique(self.init_ke, return_inverse=True)
            else:
                raw_ke = self.init_ke
            self.ke = self.param("ke", fix_init, 
                raw_ke, self.dtype, self.init_random)
        else:
            self.ke = self.init_ke
    
    def __call__(self, step):
        ke = self.ke
        if self.parametrize and self.k_symmetric:
            ke = ke[self.ke_invidx]
        ke = cmult(step, ke)
        return ke

    @property
    def expm_apply(self):
        matmul_fn = lambda A, B: jnp.einsum('k,ki->ki', A, B)
        _expm_op = self.expm_option
        _expm_op = (_expm_op,) if isinstance(_expm_op, str) else _expm_op
        return make_expm_apply(*_expm_op, matmul_fn=matmul_fn)


class AuxFieldPW(nn.Module):
    init_vq: jnp.ndarray
    kmask: jnp.ndarray
    qmask: jnp.ndarray
    parametrize: bool = False
    q_symmetric: bool = False
    init_random: float = 0.
    dtype: Optional[jnp.dtype] = None
    expm_option: Union[str, tuple] = ()

    @property
    def nbasis(self):
        return int(self.kmask.sum().item())

    @property
    def nfield(self):
        return self.init_vq.shape[0] * 2

    def setup(self):
        if self.q_symmetric and self.parametrize:
            raw_vq, self.vq_invidx = jnp.unique(self.init_ke, return_inverse=True)
        else:
            raw_vq = self.init_vq
        raw_vhs = jnp.sqrt(1/2 * raw_vq)
        vhs = jnp.tile(raw_vhs, (4, 1)) # for A and B; plus and minus Q
        if self.parametrize:
            self.vhs = self.param("vhs", fix_init, 
                vhs, self.dtype, self.init_random)
        else:
            self.vhs = vhs
        self.nq = self.init_vq.shape[0]
        self.nhs = self.nq * 2
    
    def __call__(self, step, fields, curr_wfn=None):
        log_weight = - 0.5 * (fields ** 2).sum()
        vhs = self.vhs
        if self.q_symmetric and self.parametrize:
            vhs = vhs[:, self.vq_invidx]
        fields = fields.reshape(2, self.nq)
        vplus = jnp.array([1, 1j]) @ (fields * vhs[(0,2)])   # rho(Q) terms
        vminus = jnp.array([1, -1j]) @ (fields * vhs[(1,3)]) # rho(-Q) terms
        vsum = vplus + jnp.flip(vminus)
        vsum = cmult(step, vsum)
        # remove pure multiplication at Q = 0
        vsum.at[self.nq//2].set(0)
        return vsum, log_weight
    
    @property
    def expm_apply(self):
        from jax.scipy.signal import convolve
        # sum over all Q for one electron
        def conv1ele(vhs, wfn):
            vq_mesh = jnp.zeros_like(self.qmask, dtype=vhs.dtype).at[self.qmask].set(vhs)
            wk_mesh = jnp.zeros_like(self.kmask, dtype=wfn.dtype).at[self.kmask].set(wfn)
            nwk_mesh = convolve(vq_mesh, wk_mesh, 'valid')
            return nwk_mesh[self.kmask]
        # map it for all electrons (at the last axis)
        matmul_fn = jax.vmap(conv1ele, in_axes=(None, -1), out_axes=-1)
        _expm_op = self.expm_option
        _expm_op = (_expm_op,) if isinstance(_expm_op, str) else _expm_op
        return make_expm_apply(*_expm_op, matmul_fn=matmul_fn)
