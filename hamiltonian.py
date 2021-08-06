import jax
from jax.config import config
config.update("jax_enable_x64", True)
from jax import numpy as jnp

from .utils import integrals_from_scf

@jax.tree_util.register_pytree_node_class
class Hamiltonian(object):
    use_mcd = False

    def __init__(self, h1e, eri, enuc=0.):
        self.h1e = jnp.asarray(h1e)
        self.eri = jnp.asarray(eri)
        self.enuc = enuc

    def calc_e1b(self, rdm):
        return calc_e1b(self.h1e, rdm)
    
    def calc_e2b(self, rdm):
        return calc_e2b(self.eri, rdm)
    
    def local_energy(self, bra, ket):
        """the normalized energy from two slater determinants"""
        rdm = calc_rdm(bra, ket)
        return self.enuc + self.calc_e1b(rdm) + self.calc_e2b(rdm)

    def energy_ovlp(self, bra, ket):
        """the (unnormalized) energy and overlap from two determinants"""
        raw_ene = self.local_energy(bra, ket)
        ovlp = calc_ovlp(bra, ket)
        return raw_ene * ovlp, ovlp
    
    @classmethod
    def from_pyscf(cls, mol_or_mf, **kwargs):
        if not hasattr(mol_or_mf, "mo_coeff"):
            mf = mol_or_mf.HF()
        else:
            mf = mol_or_mf
        return cls(*integrals_from_scf(mf, use_mcd=cls.use_mcd, **kwargs))
    
    def tree_flatten(self):
        children = (self.h1e, self.eri, self.enuc)
        aux_data = ("h1e", "eri", "enuc")
        return (children, aux_data)
    
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = object.__new__(cls)
        for name, data in zip(aux_data, children):
            setattr(obj, name, data)
        return obj


@jax.tree_util.register_pytree_node_class
class HamiltonianMCD(Hamiltonian):
    use_mcd = True

    def calc_e2b(self, rdm):
        return calc_e2b_chol(self.eri, rdm)

    
def calc_ovlp_ns(V, U):
    r"""
    Overlap of two (non-orthogonal) Slater determinants V and U, no spin index

    .. math::
        \langle \phi_V | \phi_U \rangle =
        \mathrm{det} \left( V^{\dagger} U \right)

    Parameters
    ----------
    V, U : array
        matrix representation of the bra(V) and ket(U) in calculate the RDM, 
        with row index (-2) for basis and column index (-1) for electrons. 

    Returns
    -------
    ovlp : flpat
        overlap of the two Slater determinants
    """
    return jnp.linalg.det( V.conj().T @ U )


def calc_ovlp(V, U):
    r"""
    Overlap of two (non-orthogonal) Slater determinants V and U with spin components

    .. math::
        \langle \phi_V | \phi_U \rangle =
        \mathrm{det} \left( V^{\dagger} U \right)

    Parameters
    ----------
    V, U : tuple of array
        pair of array for spin up (first) and spin down (second) matrix
        representation of the bra(V) and ket(U) in calculate the RDM, 
        with row index (-2) for basis and column index (-1) for electrons. 

    Returns
    -------
    ovlp : flpat
        overlap of the two Slater determinants
    """
    if (isinstance(V, jnp.ndarray) and isinstance(U, jnp.ndarray) and V.ndim == U.ndim == 2):
        return calc_ovlp_ns(V, U)
    Va, Vb = V
    Ua, Ub = U
    return calc_ovlp_ns(Va, Ua) * calc_ovlp_ns(Vb, Ub)


def calc_rdm_ns(V, U):
    r"""
    One-particle reduced density matrix. No spin index.
    
    Calculate the one particle RDM from two (non-orthogonal) Slater determinants (V for bra and U for ket). 
    Use the following formula:
    
    .. math::
        \langle \phi_V | c_i^{\dagger} c_j | \phi_U \rangle =
        [U (V^{\dagger}U)^{-1} V^{\dagger}]_{ji}
        
    where :math:`U` stands for the matrix representation of Slater determinant :math:`|\psi_U\rangle`.
    
    
    Parameters
    ----------
    V, U : array
        matrix representation of the bra(V) and ket(U) in calculate the RDM, 
        with row index (-2) for basis and column index (-1) for electrons. 
        
    Returns
    -------
    rdm : array
        one-particle reduced density matrix in computing basis
    """
    V_h = V.conj().T
    inv_O = jnp.linalg.inv(V_h @ U)
    rdm = U @ inv_O @ V_h
    return rdm.T


def calc_rdm(V, U):
    r"""
    One-particle reduced density matrix, with both spin components in a tuple
    
    Calculate the one particle RDM from two (non-orthogonal) Slater determinants (V for bra and U for ket) for each spin components.
    Use the following formula:
    
    .. math::
        \langle \phi_V | c_i^{\dagger} c_j | \phi_U \rangle =
        [U (V^{\dagger}U)^{-1} V^{\dagger}]_{ji}
        
    where :math:`U` stands for the matrix representation of Slater determinant :math:`|\psi_U\rangle`.
    
    
    Parameters
    ----------
    V, U : tuple of array
        pair of array for spin up (first) and spin down (second) matrix
        representation of the bra(V) and ket(U) in calculate the RDM, 
        with row index (-2) for basis and column index (-1) for electrons. 
        
    Returns
    -------
    rdm : array
        spin up and spin down one-particle reduced density matrix in computing basis
    """
    if (isinstance(V, jnp.ndarray) and isinstance(U, jnp.ndarray) and V.ndim == U.ndim == 2):
        return calc_rdm_ns(V, U)
    Va, Vb = V
    Ua, Ub = U
    return jnp.stack((calc_rdm_ns(Va, Ua), calc_rdm_ns(Vb, Ub)), 0)


def calc_e1b(h1e, rdm):
    if rdm.ndim == 3:
        rdm = rdm.sum(0)
    return jnp.einsum("ij,ij", h1e, rdm)


def calc_e2b(eri, rdm):
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