import jax
from jax import lax
from jax import numpy as jnp
from flax import linen as nn
from typing import Sequence, Union, Tuple
import collections

from .utils import _t_real, _t_cplx
from .utils import parse_bool, ensure_mapping
from .utils import fix_init
from .utils import pack_spin, unpack_spin
from .utils import expm_apply, cmult
from .operator import OneBody, AuxField, AuxFieldNet
from .hamiltonian import calc_slov


class Propagator(nn.Module):
    init_hmf : jnp.ndarray
    init_vhs : jnp.ndarray
    init_enuc : float
    init_wfn : Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]
    init_tsteps : Sequence[float]
    ortho_intvl : int = 0
    extra_field : int = 0
    parametrize : Union[bool, str, Sequence[str]] = True
    timevarying : Union[bool, str, Sequence[str]] = False
    aux_network : Union[None, Sequence[int], dict] = None
    init_random : float = 0.
    hermite_ops : bool = True
    use_complex : bool = False
    sqrt_tsvpar : bool = False

    @classmethod
    def create(cls, hamiltonian, init_wfn, init_tsteps, *, 
               max_nhs=None, **init_kwargs):
        init_hmf, init_vhs, init_enuc = hamiltonian.make_proj_op(init_wfn)
        if max_nhs is not None:
            init_vhs = init_vhs[:max_nhs]
        return cls(init_hmf, init_vhs, init_enuc, init_wfn, init_tsteps, **init_kwargs)

    def setup(self):
        # decide whether to make quantities changeable / parametrized in complex
        _dtype = _t_cplx if self.use_complex else _t_real
        _pd = parse_bool(("hmf", "vhs", "wfn", "enuc", "tsteps"), self.parametrize)
        _vd = parse_bool(("hmf", "vhs", "vnet"), self.timevarying)

        # handle the time steps, for Hmf and Vhs separately
        _ts_v = jnp.asarray(self.init_tsteps).reshape(-1)
        _ts_h = jnp.convolve(_ts_v, jnp.array([0.5,0.5]), "full")
        if self.sqrt_tsvpar:
            _ts_v = jnp.sqrt(jnp.abs(_ts_v))
        self.ts_v = (self.param("ts_v", fix_init, _ts_v, _dtype) 
                     if _pd["tsteps"] else _ts_v)
        self.ts_h = (self.param("ts_h", fix_init, _ts_h, _dtype) 
                     if _pd["tsteps"] else _ts_h)
        self.nts_h = self.ts_h.shape[0]
        self.nts_v = self.ts_v.shape[0]

        # concat the spin of wavefunctions to make it an array
        _wfn_packed, self.nelec = pack_spin(self.init_wfn)  
        self.wfn_packed = (self.param("wfn", fix_init, _wfn_packed, _dtype, self.init_random) 
                           if _pd["wfn"] else _wfn_packed)
        self.nbasis = self.wfn_packed.shape[0]
        # core energy, should be useless
        self.enuc = (self.param("enuc", fix_init, self.init_enuc, _t_real) 
                     if _pd["enuc"] else self.init_enuc)

        # build Hmf operator, can be vmapped to eval all the timesteps at the same time
        if _pd["hmf"] and not _vd["hmf"]:
            _hmf = self.param("hmf", fix_init, self.init_hmf, _dtype, self.init_random)
        else:
            _hmf = self.init_hmf
        if _pd["hmf"] and _vd["hmf"]:
            OneBodyCls = nn.vmap(
                OneBody, 
                in_axes=0,
                out_axes=0,
                axis_size=self.nts_h,
                variable_axes={'params': 0},
                split_rngs={'params': True})
        else:
            OneBodyCls = OneBody
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
        self.nsite = _vhs.shape[0]
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
            parametrize=_pd["vhs"] and _vd["vhs"],
            init_random=self.init_random,
            hermite_out=self.hermite_ops,
            dtype=_dtype,
            **network_args)

    def __call__(self, fields):
        # get ops without time
        all_hmf = self.hmf_ops()
        all_vhs, all_lw = self.vhs_ops(fields)
        # may add a constant shift to the log weight
        log_weight = all_lw.sum() + self.enuc # + 0.5 * self.nts_v * self.nsite
        # scale by the time step in advance, cmult is just complex number multiply
        hmf_steps = cmult(-jnp.abs(self.ts_h)[..., None, None], all_hmf)
        _ts_v = self.ts_v if self.sqrt_tsvpar else jnp.sqrt(jnp.abs(self.ts_v))
        vhs_steps = cmult(1j * _ts_v[..., None, None], all_vhs)
        # iteratively apply the projection step
        ####### begin naive for loop version #######
        # wfn = self.wfn_packed+0j
        # for its in range(self.nts_v):
        #     wfn = expm_apply(hmf_steps[its], wfn)
        #     wfn = expm_apply(vhs_steps[its], wfn)
        # wfn = expm_apply(hmf_steps[-1], wfn)
        ######## end naive for loop version ########
        def app_ops(wfn, i_ops):
            ii, *ops = i_ops
            for op in ops:
                wfn = expm_apply(op, wfn)
            if self.ortho_intvl <= 0:
                return wfn, 0.
            return lax.cond(
                (ii+1) % self.ortho_intvl == 0,
                lambda w: orthonormalize(w, self.nelec),
                lambda w: (w, 0.),
                wfn) # orthonormalize for every these steps
        wfn, logd = lax.scan(app_ops, self.wfn_packed+0j, 
            (jnp.arange(self.nts_v), hmf_steps[:-1], vhs_steps))
        wfn = expm_apply(hmf_steps[-1], wfn)
        # recover the normalizing factor during qr
        log_weight += logd.sum()
        # split different spin part
        wfn = unpack_spin(wfn, self.nelec)
        # return both the wave function matrix and the log of scalar part
        return wfn, log_weight
        
    def sign_logov(self, params, fields):
        # this method only works with fields of both bra and ket
        assert fields.ndim == 3 and fields.shape[0] == 2
        vapply = jax.vmap(self.apply, in_axes=(None, 0))
        res = vapply(params, fields)
        bra, bra_lw = jax.tree_map(lambda x: x[0], res)
        ket, ket_lw = jax.tree_map(lambda x: x[1], res)
        sign, logov = calc_slov(bra, ket)
        return sign, logov + bra_lw + ket_lw


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
