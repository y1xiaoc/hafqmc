import jax
import numpy as onp
from jax import lax
from jax import numpy as jnp
from flax import linen as nn
from jax.numpy import ndarray
from typing import Sequence, Union, Optional, Tuple

from .utils import _t_real, _t_cplx
from .utils import parse_bool, ensure_mapping
from .utils import fix_init
from .utils import pack_spin, unpack_spin, block_spin
from .utils import make_expm_apply
from .utils import chol_qr
from .operator import OneBody, AuxField, AuxFieldNet
from .hamiltonian import calc_rdm, _make_ghf, _has_spin


class Propagator(nn.Module):
    init_hmf : ndarray
    init_vhs : ndarray
    init_enuc : float
    init_tsteps : Sequence[float]
    ortho_intvl : int = 0
    expm_option : Union[str, tuple] = ()
    parametrize : Union[bool, str, Sequence[str]] = True
    timevarying : Union[bool, str, Sequence[str]] = False
    use_complex : Union[bool, str, Sequence[str]] = False
    aux_network : Union[None, Sequence[int], dict] = None
    init_random : float = 0.
    hermite_ops : bool = False
    sqrt_tsvpar : bool = False
    mfshift_wfn : Optional[Tuple[ndarray, ndarray]] = None
    dyn_mfshift : bool = False
    priori_mask : Optional[ndarray] = None

    @nn.nowrap
    @classmethod
    def create(cls, hamiltonian, type="normal", **kwargs):
        if type.lower() in ("cc", "ccsd"):
            return cls.create_ccsd(hamiltonian, **kwargs)
        else:
            return cls.create_normal(hamiltonian, **kwargs)

    @nn.nowrap
    @classmethod
    def create_normal(cls, hamiltonian, init_tsteps, *, 
                      max_nhs=None, mf_subtract=False, spin_mixing=False, 
                      **init_kwargs):
        twfn = hamiltonian.wfn0
        init_hmf, init_vhs, init_enuc = hamiltonian.make_proj_op(twfn)
        if max_nhs is not None:
            init_vhs = init_vhs[:max_nhs]
        if spin_mixing:
            ptb = (spin_mixing 
                if isinstance(spin_mixing, (float, complex)) else 0.01)
            init_hmf = block_spin(init_hmf, init_hmf, ptb)
            init_vhs = jax.vmap(block_spin, (0,0,None))(init_vhs, init_vhs, ptb)
            twfn = _make_ghf(twfn)
        mfwfn = twfn if mf_subtract else None
        return cls(init_hmf, init_vhs, init_enuc, 
            init_tsteps=init_tsteps, mfshift_wfn=mfwfn, **init_kwargs)
    
    @nn.nowrap
    @classmethod
    def create_ccsd(cls, hamiltonian, *, 
                    with_mask=True, use_complex=False,
                    expm_option=(), mf_subtract=False, **init_kwargs):
        init_hmf, init_vhs, mask = hamiltonian.make_ccsd_op()
        if with_mask:
            expm_option = ("loop", 1, 1)
        else:
            mask = None
        _cd = parse_bool(("hmf", "tsteps"), use_complex)
        use_complex = "vhs" + ",".join(k for k in _cd if _cd[k])
        mfwfn = hamiltonian.wfn0 if mf_subtract else None
        return cls(init_hmf, init_vhs, 0, init_tsteps=[-1.], 
            expm_option=expm_option, use_complex=use_complex,
            hermite_ops=False, aux_network=None, sqrt_tsvpar=False,
            priori_mask=mask, mfshift_wfn=mfwfn, **init_kwargs)
            
    @nn.nowrap
    def fields_shape(self):
        nts = len(self.init_tsteps)
        nfield = self.init_vhs.shape[0]
        return onp.array((nts, nfield))

    def setup(self):
        # handle the expm_apply method
        _expm_op = self.expm_option
        _expm_op = (_expm_op,) if isinstance(_expm_op, str) else _expm_op
        self.expm_apply = _warp_spin(make_expm_apply(*_expm_op))
        # decide whether to make quantities changeable / parametrized in complex
        _ifcplx = lambda t: _t_cplx if t else _t_real
        _td = {k: _ifcplx(v) for k, v in 
               parse_bool(("hmf", "vhs", "tsteps"), self.use_complex).items()}
        _pd = parse_bool(("hmf", "vhs", "enuc", "tsteps"), self.parametrize)
        _vd = parse_bool(("hmf", "vhs"), self.timevarying)
        # handle the time steps, for Hmf and Vhs separately
        _ts_v = jnp.asarray(self.init_tsteps).reshape(-1)
        _ts_h = jnp.convolve(_ts_v, jnp.array([0.5,0.5]), "full")
        if self.sqrt_tsvpar:
            _ts_v = jnp.sqrt(_ts_v if self.use_complex else jnp.abs(_ts_v))
        self.ts_v = (self.param("ts_v", fix_init, _ts_v, _td["tsteps"]) 
                     if _pd["tsteps"] else _ts_v)
        self.ts_h = (self.param("ts_h", fix_init, _ts_h, _td["tsteps"]) 
                     if _pd["tsteps"] else _ts_h)
        self.nts_h = self.ts_h.shape[0]
        self.nts_v = self.ts_v.shape[0]
        # core energy, should be useless
        self.enuc = (self.param("enuc", fix_init, self.init_enuc, _t_real) 
                     if _pd["enuc"] else self.init_enuc)
        # operator prioir masks
        if self.priori_mask is None:
            self.hmask = self.vmask = 1
        elif len(self.priori_mask) == 2:
            self.hmask, self.vmask = self.priori_mask
        else:
            self.hmask = self.vmask = self.priori_mask
        # build Hmf operator
        _hop = OneBody(
            self.init_hmf, 
            parametrize=_pd["hmf"], 
            init_random=self.init_random,
            hermite_out=self.hermite_ops,
            dtype=_td["hmf"])
        self.hmf_ops = [_hop.clone() if _vd["hmf"] else _hop 
                        for _ in range(self.nts_h)]
        # build Vhs operator
        if self.aux_network is None:
            AuxFieldCls = AuxField
            network_args = {}
        else:
            AuxFieldCls = AuxFieldNet
            network_args = ensure_mapping(self.aux_network, "hidden_sizes")
        _trdm = (calc_rdm(self.mfshift_wfn, self.mfshift_wfn) 
                if self.mfshift_wfn is not None else None)
        _vop = AuxFieldCls(
            self.init_vhs,
            trial_rdm=_trdm,
            parametrize=_pd["vhs"],
            init_random=self.init_random,
            hermite_out=self.hermite_ops,
            dtype=_td["vhs"],
            **network_args)
        self.vhs_ops = [_vop.clone() if _vd["vhs"] else _vop 
                        for _ in range(self.nts_v)]

    def __call__(self, wfn, fields):
        if _has_spin(wfn) and wfn[0].shape[0] < self.init_hmf.shape[-1]:
            wfn = _make_ghf(wfn)
        wfn, nelec = pack_spin(wfn)
        log_weight = self.enuc # + 0.5 * self.nts_v * self.nsite
        # get prop times
        _ts_h = -self.ts_h # the negation of t goes to here
        _ts_v = 1j * self.ts_v if self.sqrt_tsvpar else jnp.sqrt(-self.ts_v+0j)
        # step functions in iterative prop
        def app_h(wfn, ii):
            hmf = self.hmf_ops[ii](_ts_h[ii])
            return self.expm_apply(hmf * self.hmask, wfn), 0.
        def app_v(wfn, ii):
            trdm = (calc_rdm(self.mfshift_wfn, unpack_spin(wfn, nelec))
                if self.dyn_mfshift and self.mfshift_wfn is not None else None)
            vhs, lw = self.vhs_ops[ii](_ts_v[ii], fields[ii], trdm=trdm)
            return self.expm_apply(vhs * self.vmask, wfn), lw
        def nmlz(wfn, ii):
            if self.ortho_intvl == 0:
                return normalize(wfn)
            if self.ortho_intvl > 0 and (ii+1) % self.ortho_intvl == 0:
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
        # resolve the sign in the log
        wfn *= jnp.exp(log_weight.imag * 1j / onp.sum(nelec))
        # split different spin part
        wfn = unpack_spin(wfn, nelec)
        # return both the wave function matrix and the log of scalar part
        return wfn, log_weight.real


def orthonormalize_ns(wfn):
    owfn, rmat = chol_qr(wfn)
    rdiag = rmat.diagonal(0,-1,-2)
    # chol_qr gaurantees rdiag is real and positive
    # rabs = jnp.abs(rdiag)
    # owfn *= rdiag / rabs
    logd = jnp.sum(jnp.log(rdiag.real), axis=-1)
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