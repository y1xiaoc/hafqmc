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
        _vd = parse_bool(("hmf", "vhs", "vnet"), self.timevarying)

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

        # build Hmf operator, can be vmapped to eval all the timesteps at the same time
        if _pd["hmf"] and not _vd["hmf"]:
            _hmf = self.param("hmf", fix_init, self.init_hmf, _dtype, self.init_random)
        else:
            _hmf = self.init_hmf
        OneBodyCls = nn.vmap(
            OneBody, 
            in_axes=0,
            out_axes=0,
            axis_size=self.nts_h,
            variable_axes={'params': 0},
            split_rngs={'params': True})
        self.hmf_ops = OneBodyCls(
            _hmf, 
            parametrize=_pd["hmf"] and _vd["hmf"], 
            init_random=self.init_random,
            hermite_out=self.hermite_ops,
            dtype=_dtype)

        # build Vhs operator, vmapped to eval all timesteps with fields shape [nts_v, nsite]
        if _pd["vhs"] and not _vd["vhs"]:
            _vhs = self.param("vhs", fix_init, self.init_vhs, _dtype, self.init_random)
        else:
            _vhs = self.init_vhs
        if self.aux_network is None:
            AuxFieldCls = AuxField
            network_args = {}
        else:
            AuxFieldCls = AuxFieldNet
            network_args = ensure_mapping(self.aux_network, "hidden_sizes")
        AuxFieldCls = nn.vmap(
            AuxFieldCls,
            in_axes=0,
            out_axes=0,
            axis_size=self.nts_v,
            variable_axes={'params': 0 if (_vd["vhs"] or _vd["vnet"]) else None},
            split_rngs={'params': _vd["vhs"] or _vd["vnet"]})
        self.vhs_ops = AuxFieldCls(
            _vhs,
            trial_rdm = self.mfshift_rdm,
            parametrize=_pd["vhs"] and _vd["vhs"],
            init_random=self.init_random,
            hermite_out=self.hermite_ops,
            dtype=_dtype,
            **network_args)

    def __call__(self, wfn, fields):
        wfn, nelec = pack_spin(wfn)
        # get ops with time
        hmf_steps = self.hmf_ops(-self.ts_h)
        _ts_v = 1j * (self.ts_v if self.sqrt_tsvpar else 
            jnp.sqrt(self.ts_v if self.use_complex else jnp.abs(self.ts_v)))
        vhs_steps, all_lw = self.vhs_ops(_ts_v, fields)
        # may add a constant shift to the log weight
        log_weight = all_lw.sum() + self.enuc # + 0.5 * self.nts_v * self.nsite
        # iteratively apply the projection step
        ####### begin naive for loop version #######
        # wfn = wfn+0j
        # for its in range(self.nts_v):
        #     wfn = expm_apply(hmf_steps[its], wfn)
        #     wfn = expm_apply(vhs_steps[its], wfn)
        # wfn = expm_apply(hmf_steps[-1], wfn)
        ######## end naive for loop version ########
        def app_ops(wfn, i_ops):
            ii, *ops = i_ops
            for op in ops:
                wfn = self.expm_apply(op, wfn)
            if self.ortho_intvl < 0:
                return wfn, 0.
            if self.ortho_intvl == 0:
                return normalize(wfn)
            return lax.cond(
                (ii+1) % self.ortho_intvl == 0,
                lambda w: orthonormalize(w, nelec),
                lambda w: (w, 0.),
                wfn) # orthonormalize for every these steps
        wfn, logd = lax.scan(app_ops, wfn+0j, 
            (jnp.arange(self.nts_v), hmf_steps[:-1], vhs_steps))
        wfn = self.expm_apply(hmf_steps[-1], wfn)
        # recover the normalizing factor during qr
        log_weight += logd.sum()
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