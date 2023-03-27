# this file borrows script from pauxy repo
# https://github.com/pauxy-qmc/pauxy/

import time
import h5py
import numpy as np


def cholesky(mol, filename='hamil.h5', max_error=1e-6, verbose=False, cmax=20,
             CHUNK_SIZE=2.0, MAX_SIZE=20.0):
    nao = mol.nao_nr()
    if nao*nao*cmax*nao*8.0 / 1024.0**3 > MAX_SIZE:
        if verbose:
            print("# Approximate memory for cholesky > MAX_SIZE ({} GB)."
                  .format(MAX_SIZE))
            print("# Using out of core algorithm.")
            return chunked_cholesky_outcore(mol, filename=filename,
                                            max_error=max_error,
                                            verbose=verbose,
                                            cmax=cmax,
                                            CHUNK_SIZE=CHUNK_SIZE)
        else:
            return chunked_cholesky(mol, max_error=max_error, verbose=verbose,
                                    cmax=cmax)


def chunked_cholesky(mol, max_error=1e-6, verbose=False, cmax=10):
    """Modified cholesky decomposition from pyscf eris.

    See, e.g. [Motta17]_

    Only works for molecular systems.

    Parameters
    ----------
    mol : :class:`pyscf.mol`
        pyscf mol object.
    max_error : float
        Accuracy desired.
    verbose : bool
        If true print out convergence progress.
    cmax : int
        nchol = cmax * M, where M is the number of basis functions.
        Controls buffer size for cholesky vectors.

    Returns
    -------
    chol_vecs : :class:`np.ndarray`
        Matrix of cholesky vectors in AO basis.
    """
    nao = mol.nao_nr()
    diag = np.zeros(nao*nao)
    nchol_max = cmax * nao
    # This shape is more convenient for pauxy.
    chol_vecs = np.zeros((nchol_max, nao*nao))
    ndiag = 0
    dims = [0]
    nao_per_i = 0
    for i in range(0,mol.nbas):
        l = mol.bas_angular(i)
        nc = mol.bas_nctr(i)
        nao_per_i += (2*l+1)*nc
        dims.append(nao_per_i)
    # print (dims)
    for i in range(0,mol.nbas):
        shls = (i,i+1,0,mol.nbas,i,i+1,0,mol.nbas)
        buf = mol.intor('int2e_sph', shls_slice=shls)
        di, dk, dj, dl = buf.shape
        diag[ndiag:ndiag+di*nao] = buf.reshape(di*nao,di*nao).diagonal()
        ndiag += di * nao
    nu = np.argmax(diag)
    delta_max = diag[nu]
    if verbose:
        print("# Generating Cholesky decomposition of ERIs."%nchol_max)
        print("# max number of cholesky vectors = %d"%nchol_max)
        print("# iteration %5d: delta_max = %f"%(0, delta_max))
    j = nu // nao
    l = nu % nao
    sj = np.searchsorted(dims, j)
    sl = np.searchsorted(dims, l)
    if dims[sj] != j and j != 0:
        sj -= 1
    if dims[sl] != l and l != 0:
        sl -= 1
    Mapprox = np.zeros(nao*nao)
    # ERI[:,jl]
    eri_col = mol.intor('int2e_sph',
                         shls_slice=(0,mol.nbas,0,mol.nbas,sj,sj+1,sl,sl+1))
    cj, cl = max(j-dims[sj],0), max(l-dims[sl],0)
    chol_vecs[0] = np.copy(eri_col[:,:,cj,cl].reshape(nao*nao)) / delta_max**0.5

    nchol = 0
    while abs(delta_max) > max_error:
        # Update cholesky vector
        start = time.time()
        # M'_ii = L_i^x L_i^x
        Mapprox += chol_vecs[nchol] * chol_vecs[nchol]
        # D_ii = M_ii - M'_ii
        delta = diag - Mapprox
        nu = np.argmax(np.abs(delta))
        delta_max = np.abs(delta[nu])
        # Compute ERI chunk.
        # shls_slice computes shells of integrals as determined by the angular
        # momentum of the basis function and the number of contraction
        # coefficients. Need to search for AO index within this shell indexing
        # scheme.
        # AO index.
        j = nu // nao
        l = nu % nao
        # Associated shell index.
        sj = np.searchsorted(dims, j)
        sl = np.searchsorted(dims, l)
        if dims[sj] != j and j != 0:
            sj -= 1
        if dims[sl] != l and l != 0:
            sl -= 1
        # Compute ERI chunk.
        eri_col = mol.intor('int2e_sph',
                            shls_slice=(0,mol.nbas,0,mol.nbas,sj,sj+1,sl,sl+1))
        # Select correct ERI chunk from shell.
        cj, cl = max(j-dims[sj],0), max(l-dims[sl],0)
        Munu0 = eri_col[:,:,cj,cl].reshape(nao*nao)
        # Updated residual = \sum_x L_i^x L_nu^x
        R = np.dot(chol_vecs[:nchol+1,nu], chol_vecs[:nchol+1,:])
        chol_vecs[nchol+1] = (Munu0 - R) / (delta_max)**0.5
        nchol += 1
        if verbose:
            step_time = time.time() - start
            info = (nchol, delta_max, step_time)
            print ("# iteration %5d: delta_max = %13.8e: time = %13.8e"%info)

    return chol_vecs[:nchol]


def chunked_cholesky_outcore(mol, filename='hamil.h5', max_error=1e-6,
                             verbose=False, cmax=20, CHUNK_SIZE=2.0):
    """Modified cholesky decomposition from pyscf eris.

    See, e.g. [Motta17]_

    Only works for molecular systems.

    Parameters
    ----------
    mol : :class:`pyscf.mol`
        pyscf mol object.
    max_error : float
        Accuracy desired.
    verbose : bool
        If true print out convergence progress.
    cmax : int
        nchol = cmax * M, where M is the number of basis functions.
        Controls buffer size for cholesky vectors.

    Returns
    -------
    chol_vecs : :class:`np.ndarray`
        Matrix of cholesky vectors in AO basis.
    """
    nao = mol.nao_nr()
    diag = np.zeros(nao*nao)
    nchol_max = cmax * nao
    mem = 8.0*nchol_max*nao*nao / 1024.0**3
    chunk_size = min(int(CHUNK_SIZE*1024.0**3/(8*nao*nao)),nchol_max)
    if verbose:
        print("# Number of AOs: {}".format(nao))
        print("# Writing AO Cholesky to {:s}.".format(filename))
        print("# Max number of Cholesky vectors: {}".format(nchol_max))
        print("# Max memory required for Cholesky tensor: {} GB".format(mem))
        print("# Splitting calculation into chunks of size: {} / GB"
              .format(chunk_size, 8*chunk_size*nao*nao/(1024.0**3)))
        print("# Generating diagonal.")
    chol_vecs = np.zeros((chunk_size,nao*nao))
    ndiag = 0
    dims = [0]
    nao_per_i = 0
    start = time.time()
    for i in range(0,mol.nbas):
        l = mol.bas_angular(i)
        nc = mol.bas_nctr(i)
        nao_per_i += (2*l+1)*nc
        dims.append(nao_per_i)
    for i in range(0,mol.nbas):
        shls = (i,i+1,0,mol.nbas,i,i+1,0,mol.nbas)
        buf = mol.intor('int2e_sph', shls_slice=shls)
        di, dk, dj, dl = buf.shape
        diag[ndiag:ndiag+di*nao] = buf.reshape(di*nao,di*nao).diagonal()
        ndiag += di * nao
    nu = np.argmax(diag)
    delta_max = diag[nu]
    with h5py.File(filename, 'w') as fh5:
        fh5.create_dataset('Lao',
                           shape=(nchol_max, nao*nao),
                           dtype=np.float64)
    end = time.time()
    if verbose:
        print("# Time to generate diagonal {} s.".format(end-start))
        print("# Generating Cholesky decomposition of ERIs.")
        print("# iteration {:5d}: delta_max = {:13.8e}".format(0, delta_max))
    j = nu // nao
    l = nu % nao
    sj = np.searchsorted(dims, j)
    sl = np.searchsorted(dims, l)
    if dims[sj] != j and j != 0:
        sj -= 1
    if dims[sl] != l and l != 0:
        sl -= 1
    Mapprox = np.zeros(nao*nao)
    eri_col = mol.intor('int2e_sph',
                        shls_slice=(0,mol.nbas,0,mol.nbas,sj,sj+1,sl,sl+1))
    cj, cl = max(j-dims[sj],0), max(l-dims[sl],0)
    chol_vecs[0] = np.copy(eri_col[:,:,cj,cl].reshape(nao*nao)) / delta_max**0.5

    def compute_residual(chol, ichol, nchol, nu):
        # Updated residual = \sum_x L_i^x L_nu^x
        # R = np.dot(chol_vecs[:nchol+1,nu], chol_vecs[:nchol+1,:])
        R = 0.0
        with h5py.File(filename, 'r') as fh5:
            for ic in range(0, ichol):
                # Compute dot product from file.
                # print(ichol*chunk_size, (ichol+1)*chunk_size)
                # import sys
                # sys.exit()
                # print(ic*chunk_size, (ic*chunk_size)
                L = fh5['Lao'][ic*chunk_size:(ic+1)*chunk_size,:]
                R += np.dot(L[:,nu], L[:,:])
        R += np.dot(chol[:nchol+1,nu], chol[:nchol+1,:])
        return R

    nchol = 0
    ichunk = 0
    while abs(delta_max) > max_error:
        # Update cholesky vector
        start = time.time()
        # M'_ii = L_i^x L_i^x
        Mapprox += chol_vecs[nchol%chunk_size] * chol_vecs[nchol%chunk_size]
        # D_ii = M_ii - M'_ii
        delta = diag - Mapprox
        nu = np.argmax(np.abs(delta))
        delta_max = np.abs(delta[nu])
        # Compute ERI chunk.
        # shls_slice computes shells of integrals as determined by the angular
        # momentum of the basis function and the number of contraction
        # coefficients. Need to search for AO index within this shell indexing
        # scheme.
        # AO index.
        j = nu // nao
        l = nu % nao
        # Associated shell index.
        sj = np.searchsorted(dims, j)
        sl = np.searchsorted(dims, l)
        if dims[sj] != j and j != 0:
            sj -= 1
        if dims[sl] != l and l != 0:
            sl -= 1
        # Compute ERI chunk.
        eri_col = mol.intor('int2e_sph',
                            shls_slice=(0,mol.nbas,0,mol.nbas,sj,sj+1,sl,sl+1))
        # Select correct ERI chunk from shell.
        cj, cl = max(j-dims[sj],0), max(l-dims[sl],0)
        Munu0 = eri_col[:,:,cj,cl].reshape(nao*nao)
        # Updated residual = \sum_x L_i^x L_nu^x
        startr = time.time()
        R = compute_residual(chol_vecs, ichunk, nchol%chunk_size, nu)
        endr = time.time()
        if nchol > 0 and (nchol + 1) % chunk_size == 0:
            startw = time.time()
            # delta = L[ichunk*chunk_size:(ichunk+1)*chunk_size]-chol_vecs
            with h5py.File(filename, 'r+') as fh5:
                fh5['Lao'][ichunk*chunk_size:(ichunk+1)*chunk_size] = chol_vecs
            endw = time.time()
            if verbose:
                print("# Writing Cholesky chunk {} to file".format(ichunk))
                print("# Time to write {}".format(endw-startw))
            ichunk += 1
            chol_vecs[:] = 0.0
        chol_vecs[(nchol+1)%chunk_size] = (Munu0 - R) / (delta_max)**0.5
        nchol += 1
        if verbose:
            step_time = time.time() - start
            # info = (nchol, delta_max, step_time, endr-startr)
            print("iteration {:5d} : delta_max = {:13.8e} : step time ="
                  " {:13.8e} : res time = {:13.8e} "
                  .format(nchol, delta_max, step_time, endr-startr))
    with h5py.File(filename, 'r+') as fh5:
        fh5['dims'] = np.array([nao*nao, nchol])
    return nchol