"""
helpers/vmc_is_firstorder.py
============================
Importance-sampled VMC driven by a FIRST-ORDER optimizer (ADAM, plain SGD, ...).

WHY THIS EXISTS
---------------
`advd.driver.VMC_NG` bundles two things that are mathematically independent:

  1. IMPORTANCE SAMPLING: draw configurations from q = |psi|^alpha instead of
     p = |psi|^2, and reweight every expectation value with w(x) = p(x)/q(x)
     (the overdispersed sampling this project is about);
  2. NATURAL GRADIENT / SR: solve  (S + lambda I) delta = F  for the update.

`advd.driver.VMC` gives you (2)-free bare gradients, but it has no notion of a
sampling distribution at all, so switching to it silently throws away (1).

This driver keeps (1) and drops (2): it computes *exactly* the same reweighted
gradient F that VMC_NG hands to its linear solver, and passes it straight to the
optax optimizer instead. So `optax.adam` sees the importance-sampled energy
gradient, and alpha keeps doing what it did before.

HOW
---
Subclass VMC_NG and override the single property `update_fn`. Everything
upstream of the linear solve is inherited unchanged:

  * sampling from q = |psi|^alpha                (VMC_NG._prepare_derivatives)
  * the weights w = p/q, normalised to mean 1    (idem)
  * the reweighted energy statistics             (is_stats.statistics)
  * the reweighted centering of BOTH the jacobian and the local energies, and
    the sqrt(w/N_mc) scaling                     (advd's own _prepare_input)

and the gradient itself is built with advd's own helpers:

    O_L, dv = _prepare_input(jacobian, e_loc, weights=w, mode=mode)
    F       = O_L.T @ dv
            = sum_x (2 w_x / N_mc) (O_x - <O>_w)^* (e_loc(x) - <e_loc>_w)

which is the same line of code SR runs before its solve (see
advanced_drivers/_src/driver/ngd/sr.py::_compute_sr_update, where this vector is
called `grad`). Hence "identical gradient, no preconditioner" is exact here, not
approximate.

COST
----
CHEAPER per step than VMC_NG: the dense jacobian is still evaluated (same as
your current `use_ntk=True, on_the_fly=False` runs), but the NTK contraction
(2Ns x 2Ns) and the Cholesky solve are gone. Use `chunk_size_bwd` to trade time
for peak memory in the jacobian, exactly as with VMC_NG.

WHAT IS NOT SUPPORTED
---------------------
`diag_shift`, `proj_reg`, `momentum` (SPRING) and `linear_solver_fn` are
properties of the SR solve and are meaningless without it -- passing them raises
rather than silently ignoring them. Adam has its own momentum; use `b1`.
`collect_quadratic_model` needs S, which is never built, so it also raises.

`auto_is=True` DOES work: the SNR-based automatic tuning of alpha only needs the
gradient and its per-sample variance, both available here, so `info["grad_snr"]`
is still produced (via advd's own `_compute_snr_derivative`).
"""

from typing import Callable, Optional
from functools import partial

import jax
from jax.flatten_util import ravel_pytree

import netket.jax as nkjax
from netket.utils import mpi, timing
from netket.utils.types import Array, Optimizer, PyTree, ScalarOrSchedule
from netket.operator import AbstractOperator
from netket.vqs.mc import MCState
from netket.jax._jacobian.default_mode import JacobianMode

from advanced_drivers._src.driver.ngd.driver_vmc_ngd import VMC_NG
from advanced_drivers._src.driver.ngd.sr_srt_common import _prepare_input
from advanced_drivers._src.driver.ngd.sr import _compute_snr_derivative
from advanced_drivers._src.driver.ngd.distribution_constructors.abstract_distribution import (
    AbstractDistribution,
)


@partial(
    jax.jit,
    static_argnames=("log_psi", "mode", "chunk_size", "want_snr"),
)
def _compute_is_gradient(
    log_psi,
    local_grad,
    parameters,
    model_state,
    samples,
    *,
    weights: Optional[Array],
    mode: str,
    chunk_size: Optional[int],
    want_snr: bool,
    is_jac: Optional[PyTree],
):
    r"""
    The importance-sampling-reweighted gradient of the energy, as a pytree
    matching `parameters`.

    This is advd's `_sr_srt_common` with the curvature matrix and the linear
    solve deleted; every other line is theirs, so the vector returned here is
    bit-for-bit the `grad` that SR would have preconditioned.
    """
    _, unravel_params_fn = ravel_pytree(parameters)
    params_structure = jax.tree_util.tree_map(
        lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), parameters
    )

    jacobians = nkjax.jacobian(
        log_psi,
        parameters,
        samples,
        model_state,
        mode=mode,
        dense=True,
        center=False,
        chunk_size=chunk_size,
    )  # NOT centered here: _prepare_input centers it *with the weights*

    O_L, dv = _prepare_input(jacobians, local_grad, weights=weights, mode=mode)

    # (np, #ns) x (#ns) -> (np). Under sharding the sample sum is automatic;
    # under MPI it needs the explicit allreduce.
    grad, token = mpi.mpi_allreduce_sum_jax(O_L.T @ dv, token=None)

    info = {}
    if want_snr:
        # Per-parameter signal-to-noise ratio of the gradient (and, if `is_jac`
        # is given, d(SNR)/d(alpha) for auto_is). Uses the un-repacked `grad`,
        # matching how sr.py calls it.
        info.update(_compute_snr_derivative(O_L, dv, grad, weights, is_jac, token=token))

    # complex parameters are carried as [real_part, imag_part] in the real-ified
    # jacobian; repack them the same way _compute_sr_update repacks its updates.
    if mode == "complex" and nkjax.tree_leaf_iscomplex(params_structure):
        num_p = grad.shape[-1] // 2
        grad = grad[:num_p] + 1j * grad[num_p:]

    return unravel_params_fn(grad), info


@timing.timed
def is_gradient(
    log_psi,
    local_grad,
    parameters,
    model_state,
    samples,
    *,
    weights: Optional[Array] = None,
    mode: str,
    chunk_size: Optional[int] = None,
    collect_gradient_statistics: bool = False,
    is_jac: Optional[PyTree] = None,
    # --- SR-only knobs. Accepted because AbstractNGDDriver always passes them,
    #     but a non-trivial value is an error rather than a silent no-op. ---
    diag_shift=None,
    solver_fn: Optional[Callable] = None,
    proj_reg=None,
    momentum=None,
    old_updates: Optional[PyTree] = None,
    collect_quadratic_model: bool = False,
):
    """Drop-in replacement for advd's `sr` / `srt` that skips the preconditioner."""
    if momentum is not None:
        raise ValueError(
            "`momentum` (SPRING) is part of the SR solve and is not available "
            "without it. Use the optimizer's own momentum instead (optax.adam's `b1`)."
        )
    if proj_reg is not None:
        raise ValueError("`proj_reg` regularizes the SR solve, which is not performed here.")
    if collect_quadratic_model:
        raise ValueError(
            "`collect_quadratic_model` needs the QGT/NTK, which this driver never builds."
        )

    dp, info = _compute_is_gradient(
        log_psi,
        local_grad,
        parameters,
        model_state,
        samples,
        weights=weights,
        mode=mode,
        chunk_size=chunk_size,
        want_snr=bool(collect_gradient_statistics) or (is_jac is not None),
        is_jac=is_jac,
    )
    return dp, old_updates, info


class VMC_IS(VMC_NG):
    r"""
    Energy minimization with importance sampling and a first-order optimizer.

    Same sampling, weighting and estimators as :class:`VMC_NG`, but the
    reweighted gradient is passed to the optimizer directly instead of being
    preconditioned by the (regularized) quantum geometric tensor.

    Args:
        hamiltonian: The Hamiltonian to minimize.
        optimizer: An optax optimizer, e.g. ``optax.adam(...)``. Unlike with
            VMC_NG, this is expected to be a real optimizer and not ``optax.sgd``
            standing in for a learning rate.
        sampling_distribution: The distribution q to sample from, e.g.
            ``advd.driver.overdispersed_distribution(alpha=0.6)``. Defaults to
            ``|psi|^2`` (no reweighting), in which case this is ordinary VMC.
        variational_state: The :class:`netket.vqs.MCState` to optimize.
        chunk_size_bwd: Row-chunking for the jacobian evaluation (time vs memory).
        collect_gradient_statistics: Log the gradient SNR into ``info``.
        auto_is: Let the sampling distribution tune itself from the gradient SNR.
        mode: Jacobian mode, ``'real'`` or ``'complex'`` (default: inferred).
    """

    def __init__(
        self,
        hamiltonian: AbstractOperator,
        optimizer: Optimizer,
        *,
        sampling_distribution: Optional[AbstractDistribution] = None,
        variational_state: MCState = None,
        chunk_size_bwd: Optional[int] = None,
        collect_gradient_statistics: bool = False,
        auto_is: bool = False,
        mode: Optional[JacobianMode] = None,
        # rejected loudly: see the module docstring
        diag_shift: Optional[ScalarOrSchedule] = None,
        proj_reg: Optional[ScalarOrSchedule] = None,
        momentum: Optional[ScalarOrSchedule] = None,
    ):
        for name, value in (("diag_shift", diag_shift), ("proj_reg", proj_reg),
                            ("momentum", momentum)):
            if value is not None:
                raise ValueError(
                    f"`{name}` belongs to the SR solve, which VMC_IS does not perform. "
                    f"Remove it (or use VMC_NG if you want stochastic reconfiguration)."
                )

        super().__init__(
            hamiltonian,
            optimizer,
            sampling_distribution=sampling_distribution,
            diag_shift=0.0,          # unused; the base class requires a value
            variational_state=variational_state,
            chunk_size_bwd=chunk_size_bwd,
            collect_gradient_statistics=collect_gradient_statistics,
            auto_is=auto_is,
            mode=mode,
            use_ntk=False,
            on_the_fly=False,
        )

    @property
    def update_fn(self) -> Callable:
        """The bare importance-sampled gradient: no QGT, no NTK, no solve."""
        return is_gradient

    def __repr__(self):
        return (
            "VMC_IS("
            + f"\n  step_count = {self.step_count},"
            + f"\n  sampling_distribution = {self.sampling_distribution.name},"
            + f"\n  state = {self.state})"
        )
