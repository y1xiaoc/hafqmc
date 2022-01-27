import jax
from jax import numpy as jnp

from .molecule import integrals_from_scf, initwfn_from_scf

    
def calc_ovlp_ns(V, U):
    r"""
    Overlap of two (non-orthogonal) Slater determinants V and U, no spin index
    """
    return jnp.linalg.det( V.conj().T @ U )

def calc_ovlp(V, U):
    r"""
    Overlap of two (non-orthogonal) Slater determinants V and U with spin components

    Args:
        V, U (tuple of array):
            pair of array for spin up (first) and spin down (second) matrix
            representation of the bra(V) and ket(U) in calculate the RDM, 
            with row index (-2) for basis and column index (-1) for electrons. 

    Returns:
        ovlp (float):
            overlap of the two Slater determinants
    """
    if (isinstance(V, jnp.ndarray) and isinstance(U, jnp.ndarray) and V.ndim == U.ndim == 2):
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
        V, U (array):
            matrix representation of the bra(V) and ket(U) in calculate the RDM, 
            with row index (-2) for basis and column index (-1) for electrons. 

    Returns:
        sign (float):
            sign of the overlap of two SD
        logdet (float):
            log of the absolute value of the overlap
    """
    if (isinstance(V, jnp.ndarray) and isinstance(U, jnp.ndarray) and V.ndim == U.ndim == 2):
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
        V, U (tuple of array):
            pair of array for spin up (first) and spin down (second) matrix
            representation of the bra(V) and ket(U) in calculate the RDM, 
            with row index (-2) for basis and column index (-1) for electrons. 
        
    Returns:
        rdm (array):
            spin up and spin down one-particle reduced density matrix in computing basis
    """
    if (isinstance(V, jnp.ndarray) and isinstance(U, jnp.ndarray) and V.ndim == U.ndim == 2):
        return calc_rdm_ns(V, U)
    Va, Vb = V
    Ua, Ub = U
    return jnp.stack((calc_rdm_ns(Va, Ua), calc_rdm_ns(Vb, Ub)), 0)


def calc_e1b(h1e, rdm):
    # jnp.einsum("ij,ij", h1e, rdm)
    return (h1e * rdm).sum()


def calc_e2b(eri, rdm):
    if eri.ndim == 4:
        return calc_e2b_dense(eri, rdm)
    elif eri.ndim == 3:
        return calc_e2b_chol(eri, rdm)
    else:
        raise RuntimeError(f"invalid shape of ERI: {eri.shape}")

def calc_e2b_dense(eri, rdm):
    if rdm.ndim == 3:
        ga, gb = rdm
    else:
        ga, gb = rdm*.5, rdm*.5
    gt = ga + gb
    ej = 0.5 * jnp.einsum("prqs,pr,qs", eri, gt, gt)
    ek = 0.5 * (jnp.einsum("prqs,ps,qr", eri, ga, ga) 
              + jnp.einsum("prqs,ps,qr", eri, gb, gb))
    return ej - ek

def calc_e2b_chol(ceri, rdm):
    if rdm.ndim == 3:
        ga, gb = rdm
    else:
        ga, gb = rdm*.5, rdm*.5
    gt = ga + gb
    # ej = 0.5 * jnp.einsum("kpr,kqs,pr,qs", ceri, ceri, gt, gt)
    chol_j = jnp.einsum("kpr,pr->k", ceri, gt)
    ej = 0.5 * (jnp.einsum("k,k", chol_j, chol_j))
    # ek = 0.5 * (jnp.einsum("kpr,kqs,ps,qr", ceri, ceri, ga, ga)
    #           + jnp.einsum("kpr,kqs,ps,qr", ceri, ceri, gb, gb))
    chol_ka = jnp.einsum("kpr,ps->krs", ceri, ga)
    chol_kb = jnp.einsum("kpr,ps->krs", ceri, gb)
    ek = 0.5 * (jnp.einsum("krs,ksr", chol_ka, chol_ka) 
              + jnp.einsum("krs,ksr", chol_kb, chol_kb))
    return ej - ek


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

    calc_ovlp = staticmethod(calc_ovlp)
    calc_slov = staticmethod(calc_slov)
    calc_rdm  = staticmethod(calc_rdm)

    def local_energy(self, bra=None, ket=None):
        """the normalized energy from two slater determinants"""
        bra = bra if bra is not None else self.wfn0
        ket = ket if ket is not None else self.wfn0
        rdm = calc_rdm(bra, ket)
        return self.enuc + self.calc_e1b(rdm) + self.calc_e2b(rdm)

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
                   chol_cut=1e-6, orth_ao=None, full_eri=False, with_cc=False):
        if not hasattr(mol_or_mf, "mo_coeff"):
            mf = mol_or_mf.HF().run()
        else:
            mf = mol_or_mf
        if with_cc:
            assert orth_ao is None, "only support MO basis for CCSD amplitudes"
            mcc = with_cc if hasattr(with_cc, "t1") else mf.CCSD().run()
            aux = {"cc_t1": mcc.t1, "cc_t2": mcc.t2}
        else:
            aux = {}
        ints = integrals_from_scf(mf, 
            use_mcd=True, chol_cut=chol_cut, orth_ao=orth_ao)
        wfn0 = initwfn_from_scf(mf, orth_ao)
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
