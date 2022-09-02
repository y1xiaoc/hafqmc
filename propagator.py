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
    hmf_op : nn.Module
    vhs_op : nn.Module
    init_tsteps : Sequence[float]
    ortho_intvl : int = 0
    timevarying : Union[bool, str, Sequence[str]] = False
    para_tsteps : bool = False
    cplx_tsteps : bool = False
    sqrt_tsvpar : bool = False
    dyn_mfshift : bool = False
    priori_mask : Optional[ndarray] = None
    # TODO: below are parameters to be moved to create function
    mfshift_wfn : Optional[Tuple[ndarray, ndarray]] = None
    parametrize : Union[bool, str, Sequence[str]] = True
    use_complex : Union[bool, str, Sequence[str]] = False
    aux_network : Union[None, Sequence[int], dict] = None
    init_random : float = 0.
    hermite_ops : bool = False

    @nn.nowrap
    @classmethod
    def create(cls, hamiltonian, type="normal", **kwargs):
        if type.lower() in ("cc", "ccsd"):
            return cls.create_ccsd(hamiltonian, **kwargs)
        else:
            return cls.create_normal(hamiltonian, **kwargs)

    @nn.nowrap
    @classmethod
    def create_normal(cls, 
            hamiltonian, 
            init_tsteps, *, 
            max_nhs : Optional[int] = None,
            expm_option : Union[str, tuple] = (),
            parametrize : Union[bool, str, Sequence[str]] = True,
            use_complex : Union[bool, str, Sequence[str]] = False,
            aux_network : Union[None, Sequence[int], dict] = None,
            init_random : float = 0.,
            hermite_ops : bool = False,
            mf_subtract : bool = False, 
            spin_mixing : Union[bool, float, complex] = False, 
            **init_kwargs):
        # prepare data
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
        # handle parameter options
        _pd = parse_bool(("hmf", "vhs", "tsteps"), parametrize)
        _ifcplx = lambda t: _t_cplx if t else _t_real
        _cd = parse_bool(("hmf", "vhs", "tsteps"), use_complex)
        # make one body operator
        hmf_op = OneBody(
            init_hmf, 
            parametrize=_pd["hmf"], 
            init_random=init_random,
            hermite_out=hermite_ops,
            dtype=_ifcplx(_cd["hmf"]),
            expm_option=expm_option)
        # make two body operator
        if aux_network is None:
            AuxFieldCls = AuxField
            network_args = {}
        else:
            AuxFieldCls = AuxFieldNet
            network_args = ensure_mapping(aux_network, "hidden_sizes")
        vhs_op = AuxFieldCls(
            init_vhs,
            trial_wfn=mfwfn,
            parametrize=_pd["vhs"],
            init_random=init_random,
            hermite_out=hermite_ops,
            dtype=_ifcplx(_cd["vhs"]),
            expm_option=expm_option,
            **network_args)
        # build propagator
        return cls(hmf_op, vhs_op, 
            init_tsteps=init_tsteps, 
            para_tsteps=_pd["tsteps"], 
            cplx_tsteps=_cd["tsteps"], 
            **init_kwargs)
    
    @nn.nowrap
    @classmethod
    def create_ccsd(cls, 
            hamiltonian, *, 
            with_mask : bool =True, 
            expm_option : Union[str, tuple] = (),
            parametrize : Union[bool, str, Sequence[str]] = True,
            use_complex : Union[bool, str, Sequence[str]] = False,
            init_random : float = 0.,
            mf_subtract : bool = False, 
            **init_kwargs):
        # prepare data
        init_hmf, init_vhs, mask = hamiltonian.make_ccsd_op()
        if with_mask:
            expm_option = ("loop", 1, 1)
        else:
            mask = None
        mfwfn = hamiltonian.wfn0 if mf_subtract else None
        # handle parameter options
        _pd = parse_bool(("hmf", "vhs", "tsteps"), parametrize)
        _ifcplx = lambda t: _t_cplx if t else _t_real
        _cd = parse_bool(("hmf", "tsteps"), use_complex)
        # make one body operator
        hmf_op = OneBody(
            init_hmf, 
            parametrize=_pd["hmf"], 
            init_random=init_random,
            hermite_out=False,
            dtype=_ifcplx(_cd["hmf"]),
            expm_option=expm_option)
        # make two body operator
        vhs_op = AuxField(
            init_vhs,
            trial_wfn=mfwfn,
            parametrize=_pd["vhs"],
            init_random=init_random,
            hermite_out=False,
            dtype=_t_cplx,
            expm_option=expm_option)
        return cls(hmf_op, vhs_op, 
            init_tsteps=[-1.], 
            para_tsteps=_pd["tsteps"], 
            cplx_tsteps=_cd["tsteps"], 
            sqrt_tsvpar=False,
            priori_mask=mask, 
            **init_kwargs)
            
    @nn.nowrap
    def fields_shape(self):
        nts = len(self.init_tsteps)
        nfield = self.vhs_op.init_vhs.shape[0]
        return onp.array((nts, nfield))

    def setup(self):
        # handle the time steps, for Hmf and Vhs separately
        _t_tsteps = _t_cplx if self.cplx_tsteps else _t_real
        _ts_v = jnp.asarray(self.init_tsteps).reshape(-1)
        _ts_h = jnp.convolve(_ts_v, jnp.array([0.5,0.5]), "full")
        if self.sqrt_tsvpar:
            _ts_v = jnp.sqrt(_ts_v if self.use_complex else jnp.abs(_ts_v))
        self.ts_v = (self.param("ts_v", fix_init, _ts_v, _t_tsteps) 
                     if self.para_tsteps else _ts_v)
        self.ts_h = (self.param("ts_h", fix_init, _ts_h, _t_tsteps) 
                     if self.para_tsteps else _ts_h)
        self.nts_h = self.ts_h.shape[0]
        self.nts_v = self.ts_v.shape[0]
        # operator prioir masks
        if self.priori_mask is None:
            self.hmask = self.vmask = 1
        elif len(self.priori_mask) == 2:
            self.hmask, self.vmask = self.priori_mask
        else:
            self.hmask = self.vmask = self.priori_mask
        # handle the option for time varying operators
        _vd = parse_bool(("hmf", "vhs"), self.timevarying)
        # build Hmf operators
        _hop = self.hmf_op.clone()
        self.hmf_ops = [_hop.clone() if _vd["hmf"] else _hop 
                        for _ in range(self.nts_h)]
        # build Vhs operators
        _vop = self.vhs_op.clone()
        self.vhs_ops = [_vop.clone() if _vd["vhs"] else _vop 
                        for _ in range(self.nts_v)]

    def __call__(self, wfn, fields):
        if _has_spin(wfn) and wfn[0].shape[0] < self.hmf_op.init_hmf.shape[-1]:
            wfn = _make_ghf(wfn)
        wfn, nelec = pack_spin(wfn)
        log_weight = 0. # + 0.5 * self.nts_v * self.nsite
        # get prop times
        _ts_h = -self.ts_h # the negation of t goes to here
        _ts_v = 1j * self.ts_v if self.sqrt_tsvpar else jnp.sqrt(-self.ts_v+0j)
        # step functions in iterative prop
        def app_h(wfn, ii):
            hop = self.hmf_ops[ii]
            hmf = hop(_ts_h[ii])
            return hop.expm_apply(hmf * self.hmask, wfn), 0.
        def app_v(wfn, ii):
            vop = self.vhs_ops[ii]
            twfn = vop.trial_wfn
            trdm = (calc_rdm(twfn, unpack_spin(wfn, nelec))
                if self.dyn_mfshift and twfn is not None else None)
            vhs, lw = vop(_ts_v[ii], fields[ii], trdm=trdm)
            return vop.expm_apply(vhs * self.vmask, wfn), lw
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
