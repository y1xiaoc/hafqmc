import jax
import numpy as np
from jax import numpy as jnp
from jax import scipy as jsp
from jax import lax


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