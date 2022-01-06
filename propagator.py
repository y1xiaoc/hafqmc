import jax
import numpy as onp
from jax import lax
from jax import numpy as jnp
from flax import linen as nn
from jax.numpy import ndarray
from typing import Sequence, Union, Optional

from .utils import _t_real, _t_cplx
from .utils import parse_bool, ensure_mapping
from .utils import fix_init
from .utils import pack_spin, unpack_spin
from .utils import make_expm_apply
from .operator import OneBody, AuxField, AuxFieldNet
from .hamiltonian import calc_rdm


class Propagator(nn.Module):
    init_hmf : ndarray
    init_vhs : ndarray
    init_enuc : float
    init_tsteps : Sequence[float]
    ortho_intvl : int = 0
    extra_field : int = 0
    expm_option : Union[str, tuple] = ()
    parametrize : Union[bool, str, Sequence[str]] = True
    timevarying : Union[bool, str, Sequence[str]] = False
    aux_network : Union[None, Sequence[int], dict] = None
    init_random : float = 0.
    hermite_ops : bool = False
    sqrt_tsvpar : bool = False
    use_complex : bool = False
    mfshift_rdm : Optional[ndarray] = None

    @nn.nowrap
    @classmethod
    def create(cls, hamiltonian, trial_wfn, init_tsteps, *, 
               max_nhs=None, mf_subtract=False, **init_kwargs):
        init_hmf, init_vhs, init_enuc = hamiltonian.make_proj_op(trial_wfn)
        if max_nhs is not None:
            init_vhs = init_vhs[:max_nhs]
        mfrdm = calc_rdm(trial_wfn, trial_wfn) if mf_subtract else None
        return cls(init_hmf, init_vhs, init_enuc, 
            init_tsteps=init_tsteps, mfshift_rdm=mfrdm, **init_kwargs)

    @nn.nowrap
    def fields_shape(self):
        nts = len(self.init_tsteps)
        nfield = self.init_vhs.shape[0] + self.extra_field
        return onp.array((nts, nfield))

    def setup(self):
        # handle the expm_apply method
        _expm_op = self.expm_option
        _expm_op = (_expm_op,) if isinstance(_expm_op, str) else _expm_op
        self.expm_apply = make_expm_apply(*_expm_op)
        # decide whether to make quantities changeable / parametrized in complex
        _dtype = _t_cplx if self.use_complex else _t_real
        _pd = parse_bool(("hmf", "vhs", "enuc", "tsteps"), self.parametrize)
        _vd = parse_bool(("hmf", "vhs"), self.timevarying)
        # handle the time steps, for Hmf and Vhs separately
        _ts_v = jnp.asarray(self.init_tsteps).reshape(-1)
        _ts_h = jnp.convolve(_ts_v, jnp.array([0.5,0.5]), "full")
        if self.sqrt_tsvpar:
            _ts_v = jnp.sqrt(_ts_v if self.use_complex else jnp.abs(_ts_v))
        self.ts_v = (self.param("ts_v", fix_init, _ts_v, _dtype) 
                     if _pd["tsteps"] else _ts_v)
        self.ts_h = (self.param("ts_h", fix_init, _ts_h, _dtype) 
                     if _pd["tsteps"] else _ts_h)
        self.nts_h = self.ts_h.shape[0]
        self.nts_v = self.ts_v.shape[0]
        # core energy, should be useless
        self.enuc = (self.param("enuc", fix_init, self.init_enuc, _t_real) 
                     if _pd["enuc"] else self.init_enuc)
        # build Hmf operator
        _hop = OneBody(
            self.init_hmf, 
            parametrize=_pd["hmf"], 
            init_random=self.init_random,
            hermite_out=self.hermite_ops,
            dtype=_dtype)
        self.hmf_ops = [_hop.clone() if _vd["hmf"] else _hop 
                        for _ in range(self.nts_h)]
        # build Vhs operator
        if self.aux_network is None:
            AuxFieldCls = AuxField
            network_args = {}
        else:
            AuxFieldCls = AuxFieldNet
            network_args = ensure_mapping(self.aux_network, "hidden_sizes")
        _vop = AuxFieldCls(
            self.init_vhs,
            trial_rdm = self.mfshift_rdm,
            parametrize=_pd["vhs"],
            init_random=self.init_random,
            hermite_out=self.hermite_ops,
            dtype=_dtype,
            **network_args)
        self.vhs_ops = [_vop.clone() if _vd["vhs"] else _vop 
                        for _ in range(self.nts_v)]

    def __call__(self, wfn, fields):
        wfn, nelec = pack_spin(wfn)
        log_weight = self.enuc # + 0.5 * self.nts_v * self.nsite
        # get prop times
        _ts_h = -self.ts_h # the negation of t goes to here
        _ts_v = 1j * (self.ts_v if self.sqrt_tsvpar else 
            jnp.sqrt(self.ts_v if self.use_complex else jnp.abs(self.ts_v)))
        # step functions in iterative prop
        def app_h(wfn, ii):
            hmf = self.hmf_ops[ii](_ts_h[ii])
            return self.expm_apply(hmf, wfn), 0.
        def app_v(wfn, ii):
            vhs, lw = self.vhs_ops[ii](_ts_v[ii], fields[ii])
            return self.expm_apply(vhs, wfn), lw
        def nmlz(wfn, ii):
            if self.ortho_intvl == 0:
                return normalize(wfn)
            if (ii+1) % self.ortho_intvl == 0:
                return orthonormalize(wfn, nelec)
            return wfn, 0.
        # iteratively apply step functions
        wfn = wfn+0j
        for its in range(self.nts_v):
            wfn, ldh = app_h(wfn, its)
            wfn, ldv = app_v(wfn, its)
            wfn, ldn = nmlz(wfn, its)
            log_weight += ldh + ldv + ldn
        wfn, ldh = app_h(wfn, -1)
        log_weight += ldh
        # split different spin part
        wfn = unpack_spin(wfn, nelec)
        # return both the wave function matrix and the log of scalar part
        return wfn, log_weight.real


def orthonormalize_ns(wfn):
    owfn, rmat = jnp.linalg.qr(wfn)
    rdiag = rmat.diagonal(0,-1,-2)
    rabs = jnp.abs(rdiag)
    owfn *= rdiag / rabs
    logd = jnp.sum(jnp.log(rabs), axis=-1)
    return owfn, logd


def orthonormalize(wfn, nelec=None):
    if isinstance(wfn, tuple):
        wa, wb = wfn
        owa, lda = orthonormalize_ns(wa)
        owb, ldb = orthonormalize_ns(wb)
        return (owa, owb), (lda + ldb)
    elif nelec is not None:
        owfn, logd = orthonormalize(unpack_spin(wfn, nelec))
        return pack_spin(owfn)[0], logd
    else:
        return orthonormalize_ns(wfn)


def normalize(wfn):
    norm = jnp.linalg.norm(wfn, 2, -2, keepdims=True)
    nwfn = wfn / norm
    logd = jnp.log(norm).sum()
    return nwfn, logd
