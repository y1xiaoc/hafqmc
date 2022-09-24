import jax
import numpy as onp
from jax import numpy as jnp
from jax import scipy as jsp

from .utils import tree_map, scatter
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


class Hamiltonian:

    def __init__(self, h1e, ceri, enuc, wfn0, aux=None, *, full_eri=False):
        self.h1e = jnp.asarray(h1e)
        self.ceri = jnp.asarray(ceri)
        self._eri = jnp.einsum("kpr,kqs->prqs", ceri, ceri) if full_eri else None
        self.enuc = enuc
        self.wfn0 = tree_map(jnp.asarray, wfn0)
        self.aux = aux if aux is not None else {}
        self.nbasis = self.h1e.shape[-1]

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


# below are methods for plane wave UEG calculations
from .utils import symrange, rawcorr, fftconvolve


def make_pw_basis(ecut, nmax=None):
    if nmax is None:
        nmax = int((onp.sqrt((2*ecut))))
    lingrid = symrange(nmax, onp.int32)
    kmesh = onp.meshgrid(lingrid, lingrid, lingrid, copy=False, indexing='ij')
    kall = jnp.stack(kmesh, axis=-1)
    # here ecut is in the kmesh unit
    k2 = 0.5 * (kall**2).sum(-1)
    kmask = k2 <= ecut
    return kall[kmask], kmask


def make_ke(kvec):
    return 0.5 * (kvec**2).sum(-1)


def make_vq(qvec):
    q2 = (qvec**2).sum(-1)
    return jnp.where(q2 > 1e-10, 4*jnp.pi / q2, 0.)


def make_pw_rhf(ke, nelec):
    if not isinstance(nelec, int):
        return tuple(make_pw_rhf(ke, ne) for ne in nelec)
    nbas = ke.shape[0]
    sort_idx = jnp.argsort(ke)
    eye = jnp.eye(nbas)
    return eye[:, sort_idx[:nelec]]


def mod_h1e(vol, kvec):
    vfac = 1. / (2 * vol)
    qmat = kvec - kvec[:, None]
    return vfac * make_vq(qmat).sum(-1)


def calc_v0_pw(vq, kmask, qmask):
    vq_mesh = scatter(vq, qmask)
    v0_mesh = rawcorr(vq_mesh, kmask, 'valid')
    return v0_mesh[kmask]


def calc_madelung(rs, nelec):
    """Use expression in Schoof et al. (PhysRevLett.115.130402) for the
    Madelung contribution to the total energy fitted to L.M. Fraser et al.
    Phys. Rev. B 53, 1814.
    Parameters
    ----------
    rs : float
        Wigner-Seitz radius.
    ne : int
        Number of electrons.
    Returns
    -------
    v_M: float
        Madelung potential (in Hartrees).
    """
    c1 = -2.837297
    c2 = (3/(4*onp.pi))**(1/3)
    return c1 * c2 / (nelec**(1/3) * rs)


def calc_e1b_pw(ke, bra, theta):
    ekin = 0.
    for s in (0, 1):
        ekin += jnp.einsum('k,ki,ki', ke, bra[s].conj(), theta[s])
    return ekin


def calc_e2b_pw(vq, bra, theta, kmask, qmask):
    # can be viewed as matmul in basis axis
    def corr1ele(bi, tj):
        # assume bra is already conjugated
        bi_mesh = scatter(bi, kmask)
        tj_mesh = scatter(tj, kmask)
        # G_q = \sum_k bra_k * theta_{k-q}
        # gq_mesh = rawcorr(bi_mesh, tj_mesh, 'full')
        gq_mesh = fftconvolve(bi_mesh, jnp.flip(tj_mesh))
        gq = gq_mesh[qmask]
        return gq
    # like calculating `c` and `f` in LCAO cases 
    gq_trace = gq_exchange = 0.
    for s in (0, 1):
        gq_half = jax.vmap(jax.vmap(corr1ele, (1, None), -1), (None, 1), -1)(bra[s].conj(), theta[s])
        gq_trace += gq_half.trace(0, -1, -2)
        gq_exchange += jnp.einsum('qij,qji->q', gq_half, jnp.flip(gq_half, axis=0))
    gq_coulomb = gq_trace * jnp.flip(gq_trace)
    # follow arxiv:1905.04361 eq (23)
    return jnp.einsum('q,q', vq, gq_coulomb-gq_exchange)


class HamiltonianPW:

    def __init__(self, ke, vq, kmask, qmask, ecore, wfn0, aux=None):
        self.ke = jnp.asarray(ke)
        self.vq = jnp.asarray(vq)
        self.kmask = jnp.asarray(kmask)
        self.qmask = jnp.asarray(qmask)
        self.ecore = ecore
        self.wfn0 = tree_map(jnp.asarray, wfn0)
        self.aux = aux
        self.nbasis = self.ke.shape[-1]

    calc_ovlp = staticmethod(calc_ovlp)
    calc_slov = staticmethod(calc_slov)
    calc_rdm  = staticmethod(calc_rdm)

    def calc_e1b(self, bra, theta):
        return calc_e1b_pw(self.ke, bra, theta)
    
    def calc_e2b(self, bra, theta):
        return calc_e2b_pw(self.vq, bra, theta, self.kmask, self.qmask)
    
    def local_energy(self, bra=None, ket=None, optimize=True):
        """the normalized energy from two slater determinants"""
        assert optimize, 'UEG hamiltonian must use PW optimization'
        bra = bra if bra is not None else self.wfn0
        ket = ket if ket is not None else self.wfn0
        bra, ket = _align_wfn(bra, ket)
        theta = calc_theta(bra, ket)
        return self.calc_e1b(bra, theta) + self.calc_e2b(bra, theta) + self.ecore

    def make_proj_op(self):
        hmf = self.ke - calc_v0_pw(self.vq, self.kmask, self.qmask)
        vhs = jnp.sqrt(1/2 * self.vq)
        return hmf, vhs, self.kmask, self.qmask

    def to_tuple(self):
        return (self.ke, self.vq, self.kmask, self.qmask, self.ecore, self.wfn0, self.aux)

    @classmethod
    def from_ueg(cls, nelec, rs, ecut, with_uhf=False):
        nup, ndown = nelec # only support uhf for now
        ne = nup + ndown
        # basis constants
        emdl = 0.5 * ne * calc_madelung(rs, ne) # madelung energy from pauxy
        rho = 1 / (4/3*onp.pi * rs**3.0) # Density
        box_size = rs * (4/3*onp.pi * ne)**(1/3.) 
        vol = box_size**3.0 # Volume
        kfac = 2*onp.pi / box_size # k-space grid spacing
        # k space grid vectors and masks
        kvec, kmask = make_pw_basis(ecut)
        qvec, qmask = make_pw_basis(ecut * 4)
        # kinetic and potential term for each k or q
        ke = make_ke(kfac * kvec)
        vq = make_vq(kfac * qvec) / (2 * vol)
        wfn0 = make_pw_rhf(ke, nelec)
        hamiltonian = cls(ke=ke, vq=vq, kmask=kmask, qmask=qmask, 
                          ecore=0., wfn0=wfn0, aux={"emdl": emdl})
        if with_uhf:
            rkey = jax.random.PRNGKey(0) # hardcoded key since this is not important
            key0, key1 = jax.random.split(rkey)
            init_wfn = (wfn0[0] + 1e-3*(jax.random.uniform(key0)-0.5),
                        wfn0[1] + 1e-3*(jax.random.uniform(key1)-0.5))
            nene, nwfn, conv = solve_UHF(hamiltonian, init_wfn, verbose=False)
            hamiltonian.wfn0 = nwfn
        return hamiltonian


def solve_UHF(hamiltonian, init_wfn, 
              opt_method="adabelief", start_lr=1e-2,
              conv_tol=1e-8, max_iter=1000, 
              ortho_intvl=10, verbose=True):
    import optax, time
    from .propagator import orthonormalize
    from .train import make_optimizer, make_lr_schedule, ensure_mapping

    energy_fn = lambda wfn: hamiltonian.local_energy(wfn, wfn).real
    ene_and_grad = jax.value_and_grad(energy_fn)
    lr_schedule = make_lr_schedule(start_lr, delay=max_iter//5)
    optimizer = make_optimizer(lr_schedule=lr_schedule, **ensure_mapping(opt_method, 'name'))
    ensure_ortho = jax.jit(lambda wfn: orthonormalize(wfn)[0])

    @jax.jit
    def iter_step(wfn, opt_state):
        energy, grads = ene_and_grad(wfn)
        grads = tree_map(jnp.conj, grads) # for complex parameters
        updates, opt_state = optimizer.update(grads, opt_state, wfn)
        wfn = optax.apply_updates(wfn, updates)
        return energy, grads, wfn, opt_state

    wfn = tree_map(jnp.asarray, init_wfn)
    opt_state = optimizer.init(wfn)
    prev_ene = 0

    converged = False
    for ii in range(max_iter):
        tick = time.perf_counter()
        lr = lr_schedule(ii)
        curr_ene, grads, wfn, opt_state = iter_step(wfn, opt_state)
        if ii > 0 and ii % ortho_intvl == 0:
            wfn = ensure_ortho(wfn)
        delta = (curr_ene - prev_ene)
        tock = time.perf_counter()
        if verbose and ii % int(onp.ceil(1/verbose)) == 0:
            max_grad = abs(onp.max(tree_map(jnp.max, grads)))
            print(f"iter: {ii}, prev: {prev_ene:.4e}, curr: {curr_ene:.4e}, diff: {delta:.4e}, max_grad: {max_grad:.4e}, time: {tock-tick:.3f}")
        if abs(delta) < conv_tol * lr:
            converged = True
            if verbose:
                print(f'converged at iteration {ii}')
            break
        prev_ene = curr_ene
    
    return curr_ene, ensure_ortho(wfn), converged

