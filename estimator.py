import jax
import numpy as np
from jax import lax
from jax import numpy as jnp
from functools import partial

from .utils import paxis
from .hamiltonian import Hamiltonian
from .propagator import Propagator


def exp_shifted(x, normalize=None):
    stblz = paxis.all_max(x)
    exp = jnp.exp(x - stblz)
    if normalize:
        assert normalize.lower() in ("sum", "mean"), "invalid normalize option"
        reducer = getattr(paxis, f"all_{normalize.lower()}")
        total = reducer(lax.stop_gradient(exp))
        exp /= total
    return exp


def make_eval_local(hamil: Hamiltonian, prop: Propagator):
    """Create a function that evaluates local energy, sign and log of the overlap.

    Args:
        hamil (Hamiltonian): 
            the hamiltonian of the system contains 1- and 2-body integrals.
        prop (Propagator): 
            the propagator ansatz that generate a Slater determinant from aux fields.

    Returns:
        eval_local (callable): 
            a function that takes parameters of the propagator and the 
            field configurations for bra and ket (shape: [2 x n_tstep x n_site]) 
            and returns the local energy, sign and overlap from the bra and ket.
    """

    vapply = jax.vmap(prop.apply, in_axes=(None, 0))

    def eval_local(params, fields):
        r"""evaluate the local energy, sign and log-overlap of the bra and ket.

        Args:
            params (dict): 
                the parameter of the propagator ansatz (as a flax linen model)
            fields (array): 
                the field configurations `[2 x n_tstep x n_site]` for both bra and ket

        Returns:
            eloc (float): 
                the local energy :math:`\frac{ <\Phi(\sigma)|H|\Psi(\sigma)> }{ <\Phi(\sigma)|\Psi(\sigma)> }`
            sign (float): 
                the sign of the overlap :math:`\frac{ <\Phi(\sigma)|\Psi(\sigma)> }{ |<\Phi(\sigma)|\Psi(\sigma)>| }`
            logov (float): 
                the log of absolute value of the overlap :math:`\log{ |<\Phi(\sigma)|\Psi(\sigma)>| }`
        """
        res = vapply(params, fields)
        bra, bra_lw = jax.tree_map(lambda x: x[0], res)
        ket, ket_lw = jax.tree_map(lambda x: x[1], res)
        eloc = hamil.local_energy(bra, ket)
        sign, logov = hamil.calc_slov(bra, ket)
        return eloc, sign, logov + bra_lw + ket_lw

    return eval_local


def make_eval_total(hamil: Hamiltonian, prop: Propagator, 
                    default_batch: int = 100, calc_stds: bool = False):
    """Create a function that evaluates the total energy from a batch of field configurations.

    Args:
        hamil (Hamiltonian): 
            the hamiltonian of the system contains 1- and 2-body integrals.
        prop (Propagator): 
            the propagator ansatz that generate a Slater determinant from aux fields.
        default_batch (int, optional): 
            the batch size to use if there is no pre-spiltted batch in the data. Defaults to 100.

    Returns:
        eval_total (callable): 
            a function that takes parameters and (batched) data that contains 
            field configurations (shape: `[(n_loop x) n_batch x 2 x n_tstep x n_site]`)
            and the corresponding (unnormalized) log density that they are sampled from,
            and returns the estimated total energy and auxiliary estimations. 
    """

    eval_local = make_eval_local(hamil, prop)
    batch_eval = jax.vmap(eval_local, in_axes=(None, 0))
        
    def eval_total(params, data):
        r"""evaluate the total energy and the auxiliary estimations from batched data.

        Args:
            params (dict): 
                the parameter of the propagator ansatz (as a flax linen model)
            data (tuple of array): 
                a tuple like (fields, logsw), for field configurations and corresponding log density
                that they are sampled from (for important sampling purpose). The fields are of shape
                `[(n_loop x) n_batch x 2 x n_tstep x n_site]` and logsw of shape `[(n_loop x) n_batch]`.
                The function would loop for `n_loop` times with each time eval a batch size n_batch.
                If n_loop is not given, calculated from `default_batch` (as the maximum batch size).

        Returns:
            e_tot (float): 
                the estimated total energy :math:`\frac{ <\Phi|H|\Psi> }{ <\Phi|\Psi> }`
            aux_data (tuple): 
                the dict for auxiliary estimated data, for now it is {e_tot, exp_es, exp_s}
                where exp_es is the the estimated `(eloc * sign).real` 
                and exp_s is the estimated `sign.real`.
        """
        fields, logsw = data
        if fields.ndim == 4:
            batch_size = np.min((fields.shape[0], default_batch))
            fields = fields.reshape(-1, batch_size, *fields.shape[1:])
            if logsw is not None:
                logsw = logsw.reshape(-1, batch_size)
        assert fields.ndim == 5
        eval_fn = partial(batch_eval, params)
        if fields.shape[0] > 1:
            eval_fn = jax.checkpoint(eval_fn, prevent_cse=False)
        eloc, sign, logov = lax.map(eval_fn, fields)
        logsw = lax.stop_gradient(logsw) if logsw is not None else 0.
        rel_w = exp_shifted(logov - logsw, normalize="mean")
        exp_es = paxis.all_mean((eloc * sign) * rel_w)
        exp_s = paxis.all_mean(sign * rel_w)
        etot = exp_es.real / exp_s.real
        aux_data = {"e_tot": etot, 
                    "exp_es": exp_es.real, 
                    "exp_s": exp_s.real}
        if calc_stds:
            tot_w = paxis.all_mean(rel_w) # should be just 1, but provide correct gradient
            var_es = paxis.all_mean(jnp.abs(eloc*sign - exp_es/tot_w)**2 * rel_w) / tot_w
            var_s = paxis.all_mean(jnp.abs(sign - exp_s/tot_w)**2 * rel_w) / tot_w
            aux_data.update(std_es=jnp.sqrt(var_es), std_s=jnp.sqrt(var_s))
        return etot, aux_data
            
    return eval_total

