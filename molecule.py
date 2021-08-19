import numpy as np
from .pauxy_scripts import chunked_cholesky


def rotate_wfn(coeff, X, thld=1e-12):
    if isinstance(coeff, (list, tuple)):
        return tuple(rotate_wfn(c, X, thld) for c in coeff)
    new_coeff = np.linalg.solve(X, coeff)
    new_coeff[np.abs(new_coeff) < thld] = 0
    return new_coeff

def rotate_h1e(h1e, X):
    return X.conj().T @ h1e @ X

def rotate_eri(eri, X):
    from pyscf import ao2mo
    nb = X.shape[1]
    return ao2mo.full(eri, X, compact=False).reshape(nb,nb,nb,nb)

def rotate_eri_chol(ceri, X):
    return np.einsum("kpr,pi,rl->kil", ceri, X.conj(), X)

def rotate_integrals(h1e, eri, X):
    # the shape of X has to be [nao x nb]
    # normally nb is nao but can also be less
    # 1-body term
    assert X.ndim == 2
    h1e = rotate_h1e(h1e, X)
    # eri, in cholesky vector
    if isinstance(eri, np.ndarray) and eri.ndim == 3:
        eri = rotate_eri_chol(eri, X)
    # eri, in dense form or just mol
    else:
        eri = rotate_eri(eri, X)
    # h1e: [nb x nb]
    # eri: [nchol x nb x nb] or [nb x nb x nb x nb]
    return h1e, eri


def get_orth_ao(mf, method=None):
    from pyscf import lo
    if isinstance(method, np.ndarray):
        X = method
    elif method is None:
        if mf.mo_coeff is None:
            mf.run()
        X = mf.mo_coeff
    else:
        X = lo.orth_ao(mf, method)
    if X.ndim == 3:
        X = X[0]
    return X


def integrals_from_scf(mf, use_mcd=True, chol_cut=1e-6, orth_ao=None):
    mol = mf.mol
    enuc = mf.energy_nuc()
    h1e = mf.get_hcore()
    if use_mcd:
        eri = chunked_cholesky(mol, max_error=chol_cut)
        eri = eri.reshape(eri.shape[0], mol.nao, mol.nao)
    else:
        eri = mf._eri if mf._eri is not None else mf.mol
    X = get_orth_ao(mf, orth_ao)
    h1e, eri = rotate_integrals(h1e, eri, X)
    return h1e, eri, enuc


def initwfn_from_scf(mf, orth_ao=None):
    if mf.mo_coeff is None:
        mf.run()
    X = get_orth_ao(mf, orth_ao)
    if mf.mo_coeff.ndim == 3:
        mo_a = mf.mo_coeff[0, :, mf.mo_occ[0] > 0]
        mo_b = mf.mo_coeff[1, :, mf.mo_occ[1] > 0]
    else:
        mo_a = mf.mo_coeff[:, mf.mo_occ > 0]
        mo_b = mf.mo_coeff[:, mf.mo_occ > 0]
    assert (mo_a.shape[1], mo_b.shape[1]) == mf.mol.nelec
    return rotate_wfn((mo_a, mo_b), X)