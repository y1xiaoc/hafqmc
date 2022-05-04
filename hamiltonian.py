import jax
import numpy as onp
from jax import numpy as jnp
from jax import scipy as jsp

from .molecule import integrals_from_scf, initwfn_from_scf
from .molecule import initwfn_from_ghf, get_orth_ao, solve_ghf


def _has_spin(wfn):
    return not (isinstance(wfn, (jnp.ndarray, onp.ndarray)) 
                and wfn.ndim == 2)

def _make_ghf(wfn):
    assert _has_spin(wfn)
    wa, wb = wfn
    return jsp.linalg.block_diag(wa, wb)

def _align_wfn(V, U):
    uhf_v, uhf_u = _has_spin(V), _has_spin(U)
    if uhf_v and not uhf_u: #U is ghf
        V = _make_ghf(V)
    if not uhf_v and uhf_u: #V is ghf
        U = _make_ghf(U)
    return V, U

def _align_rdm(rdm, nao):
    # return sum diag block and all components
    if rdm.ndim == 2 and rdm.shape[-1] == nao:
        # rhf case, assume rdm is from single wfn
        return rdm*2, jnp.stack([rdm, rdm])
    if rdm.ndim == 3 and rdm.shape[0] in (2,4) and rdm.shape[-1] == nao:
        # uhf case, no non-diag block (or aligned ghf case)
        return rdm[0]+rdm[-1], rdm
    if rdm.ndim == 2 and rdm.shape[-1] == nao * 2:
        # ghf case, return four blocks in uu,ud,du,dd order
        lrdm = rdm.reshape(2,nao,2,nao).swapaxes(1,2)
        return lrdm[0,0]+lrdm[1,1], lrdm
    raise ValueError("unknown rdm type")
   
    
def calc_ovlp_ns(V, U):
    r"""
    Overlap of two (non-orthogonal) Slater determinants V and U, no spin index
    """
    return jnp.linalg.det( V.conj().T @ U )

def calc_ovlp(V, U):
    r"""
    Overlap of two (non-orthogonal) Slater determinants V and U with spin components

    Args:
        V, U (array or tuple of array):
            matrix representation of the bra(V) and ket(U) in calculate the RDM, 
            with row index (-2) for basis and column index (-1) for electrons. 

    Returns:
        ovlp (float):
            overlap of the two Slater determinants
    """
    V, U = _align_wfn(V, U)
    if not _has_spin(V) and not _has_spin(U):
        return calc_ovlp_ns(V, U)
    Va, Vb = V
    Ua, Ub = U
    return calc_ovlp_ns(Va, Ua) * calc_ovlp_ns(Vb, Ub)


def calc_slov_ns(V, U):
    r"""
    Sign and log of overlap of two SD V and U, no spin index
    """
    return jnp.linalg.slogdet( V.conj().T @ U )

def calc_slov(V, U):
    r"""
    Sign and log of overlap of two SD V and U with spin components

    Args:
        V, U (array or tuple of array):
            matrix representation of the bra(V) and ket(U) in calculate the RDM, 
            with row index (-2) for basis and column index (-1) for electrons. 

    Returns:
        sign (float):
            sign of the overlap of two SD
        logdet (float):
            log of the absolute value of the overlap
    """
    V, U = _align_wfn(V, U)
    if not _has_spin(V) and not _has_spin(U):
        return calc_slov_ns(V, U)
    Va, Vb = V
    Ua, Ub = U
    sa, la = calc_slov_ns(Va, Ua)
    sb, lb = calc_slov_ns(Vb, Ub)
    return sa * sb, la + lb


def calc_rdm_ns(V, U):
    r"""
    One-particle reduced density matrix. No spin index.
    """
    V_h = V.conj().T
    inv_O = jnp.linalg.inv(V_h @ U)
    rdm = U @ inv_O @ V_h
    return rdm.T

def calc_rdm(V, U):
    r"""
    One-particle reduced density matrix, with both spin components in a tuple
    
    Calculate the one particle RDM from two (non-orthogonal) 
    Slater determinants (V for bra and U for ket) for each spin components.
    
    .. math::
        \langle \phi_V | c_i^{\dagger} c_j | \phi_U \rangle =
        [U (V^{\dagger}U)^{-1} V^{\dagger}]_{ji}
        
    :math:`U` stands for the matrix representation of Slater determinant :math:`|\psi_U\rangle`.
    
    Args:
        V, U (array or tuple of array):
            matrix representation of the bra(V) and ket(U) in calculate the RDM, 
            with row index (-2) for basis and column index (-1) for electrons. 
        
    Returns:
        rdm (array):
            spin up and spin down one-particle reduced density matrix in computing basis
    """
    V, U = _align_wfn(V, U)
    if not _has_spin(V) and not _has_spin(U):
        return calc_rdm_ns(V, U)
    Va, Vb = V
    Ua, Ub = U
    return jnp.stack((calc_rdm_ns(Va, Ua), calc_rdm_ns(Vb, Ub)), 0)


def calc_e1b(h1e, rdm):
    # jnp.einsum("ij,ij", h1e, rdm)
    gd, gl = _align_rdm(rdm, h1e.shape[-1])
    return (h1e * gd).sum()


def calc_e2b(eri, rdm):
    gs, ga = _align_rdm(rdm, eri.shape[-1])
    if eri.ndim == 4:
        return calc_ej_dense(eri, gs) - calc_ek_dense(eri, ga)
    elif eri.ndim == 3:
        return calc_ej_chol(eri, gs) - calc_ek_chol(eri, ga)
    else:
        raise RuntimeError(f"invalid shape of ERI: {eri.shape}")

def calc_ej_dense(eri, srdm):
    return 0.5 * jnp.einsum("prqs,pr,qs", eri, srdm, srdm)

def calc_ej_chol(ceri, srdm):
    chol_j = jnp.einsum("kpr,pr->k", ceri, srdm)
    return 0.5 * jnp.einsum("k,k", chol_j, chol_j)

def calc_ek_dense(eri, rdm):
    assert rdm.ndim in (3, 4)
    subs = "prqs,lps,lqr" if rdm.ndim == 3 else "prqs,abps,baqr"
    return 0.5 * jnp.einsum(subs, eri, rdm, rdm)

def calc_ek_chol(ceri, rdm):
    assert rdm.ndim in (3, 4)
    chol_k = jnp.einsum("kpr,...ps->k...rs", ceri, rdm)
    subs = "klrs,klsr" if rdm.ndim == 3 else "kabrs,kbasr"
    return 0.5 * jnp.einsum(subs, chol_k, chol_k)


def calc_v0(eri):
    if eri.ndim == 4:
        return calc_v0_dense(eri)
    elif eri.ndim == 3:
        return calc_v0_chol(eri)
    else:
        raise RuntimeError(f"invalid shape of ERI: {eri.shape}")

def calc_v0_dense(eri):
    return jnp.einsum("prrs->ps", eri)

def calc_v0_chol(ceri):
    return jnp.einsum("kpr,krs->ps", ceri, ceri)


def calc_theta_ns(V, U):
    V_h = V.conj().T
    inv_O = jnp.linalg.inv(V_h @ U)
    return U @ inv_O

def calc_theta(V, U):
    V, U = _align_wfn(V, U)
    if not _has_spin(V) and not _has_spin(U):
        return calc_theta_ns(V, U)
    Va, Vb = V
    Ua, Ub = U
    return (calc_theta_ns(Va, Ua), calc_theta_ns(Vb, Ub))


def calc_rdm_opt(V, theta):
    V, theta = _align_wfn(V, theta)
    if not _has_spin(V) and not _has_spin(theta):
        return theta @ V.conj().T
    Va, Vb = V
    tha, thb = theta
    return jnp.stack((tha @ Va.conj().T, thb @ Vb.conj().T), 0)


def calc_e2b_opt(ceri, bra, theta):
    bra, theta = _align_wfn(bra, theta)
    if _has_spin(bra) and _has_spin(theta):
        ej, ek = calc_ejk_opt_u(ceri, bra, theta)
    else:
        if bra.shape[0] == ceri.shape[-1]:
            ej, ek = calc_ejk_opt_r(ceri, bra, theta)
        else:
            ej, ek = calc_ejk_opt_g(ceri, bra, theta)
    return ej - ek

def calc_ejk_opt_r(ceri, bra, theta):
    f = jnp.einsum("kpq,pi,qj->kij", ceri, bra.conj(), theta)
    ej = 2 * jnp.sum(f.trace(0, -1, -2) ** 2)
    ek = jnp.einsum("kij,kji", f, f)
    return ej, ek

def calc_ejk_opt_u(ceri, bra, theta):
    ej = ek = 0.
    fup = jnp.einsum("kpq,pi,qj->kij", ceri, bra[0].conj(), theta[0])
    cup = fup.trace(0, -1, -2)
    ek += 0.5 * jnp.einsum("kij,kji", fup, fup)
    del fup
    fdn = jnp.einsum("kpq,pi,qj->kij", ceri, bra[1].conj(), theta[1])
    cdn = fdn.trace(0, -1, -2)
    ek += 0.5 * jnp.einsum("kij,kji", fdn, fdn)
    ej = 0.5 * jnp.sum((cup + cdn)**2)
    return ej, ek

def calc_ejk_opt_g(ceri, bra, theta):
    nao = ceri.shape[-1]
    nele = bra.shape[-1]
    bra = bra.reshape(2, nao, nele)
    theta = theta.reshape(2, nao, nele)
    f = jnp.einsum("kpq,api,aqj->kij", ceri, bra.conj(), theta)
    ej = 0.5 * jnp.sum(f.trace(0, -1, -2) ** 2)
    ek = 0.5 * jnp.einsum("kij,kji", f, f)
    return ej, ek


@jax.tree_util.register_pytree_node_class
class Hamiltonian:

    def __init__(self, h1e, ceri, enuc, wfn0, aux=None, *, full_eri=False):
        self.h1e = jnp.asarray(h1e)
        self.ceri = jnp.asarray(ceri)
        self._eri = jnp.einsum("kpr,kqs->prqs", ceri, ceri) if full_eri else None
        self.enuc = enuc
        self.wfn0 = wfn0
        self.aux = aux if aux is not None else {}

    def calc_e1b(self, rdm):
        return calc_e1b(self.h1e, rdm)
    
    def calc_e2b(self, rdm):
        eri = self.ceri if self._eri is None else self._eri
        return calc_e2b(eri, rdm)
    
    def calc_e2b_opt(self, bra, theta):
        return calc_e2b_opt(self.ceri, bra, theta)

    calc_ovlp = staticmethod(calc_ovlp)
    calc_slov = staticmethod(calc_slov)
    calc_rdm  = staticmethod(calc_rdm)

    def local_energy(self, bra=None, ket=None, optimize=True):
        """the normalized energy from two slater determinants"""
        bra = bra if bra is not None else self.wfn0
        ket = ket if ket is not None else self.wfn0
        le_fn = (self.local_energy_opt 
            if optimize and self._eri is None else self.local_energy_raw)
        return le_fn(bra, ket)

    def local_energy_raw(self, bra, ket):
        rdm = calc_rdm(bra, ket)
        return self.enuc + self.calc_e1b(rdm) + self.calc_e2b(rdm)
    
    def local_energy_opt(self, bra, ket):
        bra, ket = _align_wfn(bra, ket)
        theta = calc_theta(bra, ket)
        rdm = calc_rdm_opt(bra, theta)
        return self.enuc + self.calc_e1b(rdm) + self.calc_e2b_opt(bra, theta)

    def make_proj_op(self, trial):
        """generate the modified hmf, vhs and enuc for projection"""
        eri = self.ceri if self._eri is None else self._eri
        hmf_raw = self.h1e - 0.5 * calc_v0(eri)
        vhs_raw = self.ceri # vhs is real here, will time 1j in propagator
        if trial is None:
            return hmf_raw, vhs_raw, self.enuc
        rdm_t = calc_rdm(trial, trial)
        if rdm_t.ndim == 3:
            rdm_t = rdm_t.sum(0)
        vbar = jnp.einsum("kpq,pq->k", vhs_raw, rdm_t)
        enuc = self.enuc - 0.5 * (vbar**2).sum()
        hmf = hmf_raw + jnp.einsum('kpq,k->pq', vhs_raw, vbar)
        vhs = vhs_raw - vbar.reshape(-1,1,1) * jnp.eye(vhs_raw.shape[-1]) / rdm_t.trace()
        return hmf, vhs, enuc

    def make_ccsd_op(self):
        """generate hmf, vhs and corresponding masks from ccsd amps"""
        if "cc_t1" not in self.aux:
            raise ValueError("cc data not found in hamiltonian")
        t1, t2 = self.aux["cc_t1"], self.aux["cc_t2"]
        t1 = t1[0] if isinstance(t1, tuple) else t1
        t2 = t2[0] if isinstance(t2, tuple) else t2
        nocc, nvir = t1.shape
        noxv = nocc * nvir
        nbas = nocc + nvir
        hmf = jnp.zeros((nbas, nbas)).at[nocc:, :nocc].set(t1.swapaxes(0,1))
        evs, vecs = jnp.linalg.eigh(t2.swapaxes(1,2).reshape(noxv, noxv))
        tmpv = (jnp.sqrt(evs + 0j) * vecs).T.reshape(noxv, nocc, nvir)
        vhs = jnp.zeros((noxv, nbas, nbas), tmpv.dtype)
        vhs = vhs.at[:, nocc:, :nocc].set(tmpv.swapaxes(-1, -2))
        mask = jnp.zeros((nbas, nbas), bool).at[nocc:, :nocc].set(True)
        return hmf, vhs, mask
    
    def to_tuple(self):
        return (self.h1e, self.ceri, self.enuc, self.wfn0, self.aux)

    @classmethod
    def from_pyscf(cls, mol_or_mf,
                   chol_cut=1e-6, orth_ao=None, full_eri=False, 
                   with_cc=False, with_ghf=False):
        if not hasattr(mol_or_mf, "mo_coeff"):
            mf = mol_or_mf.HF().run()
        else:
            mf = mol_or_mf
        orth_mat = get_orth_ao(mf, orth_ao)
        aux = {"orth_mat": orth_mat}
        if with_cc:
            assert orth_ao is None, "only support MO basis for CCSD amplitudes"
            mcc = with_cc if hasattr(with_cc, "t1") else mf.CCSD().run()
            aux.update(cc_t1=mcc.t1, cc_t2=mcc.t2)
        if with_ghf:
            mghf = with_ghf if hasattr(with_ghf, "mo_coeff") else solve_ghf(mf.mol)
            aux.update(ghf_wfn=initwfn_from_ghf(mghf, mf, orth_mat))
        ints = integrals_from_scf(mf, 
            use_mcd=True, chol_cut=chol_cut, orth_ao=orth_mat)
        wfn0 = initwfn_from_scf(mf, orth_mat)
        return cls(*ints, wfn0, aux, full_eri=full_eri)
    
    def tree_flatten(self):
        fields = ("h1e", "ceri", "enuc", "_eri", "wfn0", "aux")
        children = tuple(getattr(self, f) for f in fields)
        return (children, fields)
    
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = object.__new__(cls)
        for name, data in zip(aux_data, children):
            setattr(obj, name, data)
        return obj
