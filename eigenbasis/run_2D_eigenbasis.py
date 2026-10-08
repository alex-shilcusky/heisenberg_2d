"""
run_2D_eigenbasis.py
====================
2D square-lattice J1-J2 in the 2x2-plaquette CLUSTER EIGENBASIS, VMC with the
2D eigenbasis ViT (learnable embedding), plus an exact-diagonalization
reference for small L so you can watch the VMC energy descend toward E0.

Structured after z_EB_optim.py / z_mainbash.sh:
  * N_samples is the sample count itself (it used to be a 2048 multiplier)
  * lr and diag_shift are LINEAR optax schedules (lr1->lr2, ds1->ds2), and are
    OFFSET-AWARE so a resubmit continues the schedule from the global step
    rather than re-annealing from lr1.
  * checkpoint / .progress / .done / .timeout sentinels + a wall-clock budget,
    so the bash wrapper can auto-resubmit exactly as z_mainbash.sh does.

CHANGES vs the previous version
-------------------------------
1. ONE CONTINUOUS LOG (0 -> iters across all resubmissions):
     - the driver's step counter is set to `step_offset` before run(), so the
       iterations the JsonLog records are GLOBAL, not local-to-this-submission
       (the counter otherwise restarts at 0 every job -> the ~3500 cap).
     - JsonLog opens in mode='append' on a resume, so every window lands in the
       SAME .log file instead of overwriting it.
     - the old ".segK.log" archiving is removed (no longer needed, and it made
       the notebook's `*.log` glob pick up each window as a separate run).
   >>> VERIFY once on your build (see the 2-run test in the chat) that
       `driver._step_count` is honored: iters in the single .log should exceed
       one wall-clock window. If not, tell me and I'll switch to a callback-side
       history writer (ProgressWriter already knows the true global step).

2. n_discard_per_chain is now a tunable arg (--n_discard_per_chain), passed to
   the MCState for the metro sampler, and tagged into the filename (it changes
   the sampling trajectory and you're sweeping it).

3. FULL provenance -> <fname>.config.json written at startup with EVERY input
   parameter (incl. chunk_size and on_the_fly). The filename stays limited to
   run *identity*; performance knobs live in the sidecar so they don't fork the
   checkpoint path. See the TAG_* switches below.

4. ADAPTIVE alpha ([distribution] auto_alpha = true): pass `auto_is=True` to
   VMC_NG, which is the package's own adaptive importance sampling — it moves
   the overdispersion exponent every step along the gradient of the
   gradient-SNR. `alpha` in the config becomes the INITIAL value only. This is
   exactly the usage in importance_sampling_nqs/notebooks/j1j2_vmc.ipynb; the
   tuning rate and per-step clip live in `overdispersed_distribution.update_params`
   and are deliberately not re-exposed here.

   What this script adds on top is only what the resubmit structure needs:
     - the live alpha is written into .progress and restored on resume, so it
       continues across submissions instead of re-annealing from the config
       value each job (the .mpack holds vstate.variables only, and alpha is not
       one of them);
     - SamplingDiag recomputes its importance weights with the CURRENT alpha,
       not the startup one, so ess_w stays meaningful once alpha moves.
   The trajectory itself is logged by the package as log["info"]["alpha"], as
   plotted at the end of the notebook.

   Requires on_the_fly=False and mode='complex'; both are checked at startup.
   With auto_alpha=false (the default) nothing about the run changes.

Self-contained EXCEPT for things already in helpers (imported, not modified):
  helpers.hamiltonians_graphs : GraphCust, GraphOperatorCust
  helpers.EigenHilbert        : EigenHilbert
  helpers.multilayer_2D_vit_Eigenbasis : ViT_2D_EB   (the new wavefunction)

python 3.13 / netket 3.16.1.post1
"""

import os
import sys
import time
import json
import shutil
from functools import reduce
from datetime import datetime

import numpy as np
import netket as nk

import jax
import jax.numpy as jnp
import optax
import flax
import scipy.sparse.linalg as sla
from netket.callbacks import InvalidLossStopping

import advanced_drivers as advd 

from helpers.hamiltonians_graphs import GraphCust, GraphOperatorCust
from helpers.EigenHilbert import EigenHilbert
from helpers.multilayer_2D_vit_Eigenbasis import ViT_2D_EB
from helpers.symmetries import (SymmProject, build_symmetry_tables,
                                C4V_ELEMENTS)

import inspect

from jax.tree_util import register_pytree_node_class

job_start = time.time()
print('\n#############')
print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
print(jax.devices())
print('#############\n', flush=True)


# ===========================================================================
# What goes in the FILENAME (run identity) vs the config.json (everything)
# ---------------------------------------------------------------------------
# Identity params fork the checkpoint path: change one -> fresh run. Keep this
# to things that define the *experiment*. Performance/memory knobs are NOT here
# by default:
#   * chunk_size  -> if it were in the name, lowering it after an OOM would fork
#                    the path and you'd lose the resume (start from step 0).
#   * on_the_fly  -> pinned False right now (srt_onthefly is_jac bug) -> tagging
#                    a constant is just noise.
# Both are still fully recorded in <fname>.config.json. Flip a switch to True
# only if you deliberately want to fork runs on that knob.
# ===========================================================================
TAG_N_DISCARD  = True     # n_discard_per_chain changes sampling -> identity
TAG_CHUNK_SIZE = False    # perf knob; True => forks path on chunk_size
TAG_ON_THE_FLY = False    # constant right now; True => tag it anyway

DATA_ROOT = sys.argv[2]

# ===========================================================================
# args 
# ===========================================================================
def parse_config():
    global config
    with open(sys.argv[1], "rb") as f:
        config = json.load(f)

    # System
    global dimensions, L, J2, J1, sign_rule, transl_invariant
    system = config["system"]
    # One config per basis: refuse a config written for the other basis. A
    # config without the key (written before the field existed) is accepted.
    if system.get("basis", "eigen") != "eigen":
        raise SystemExit(f"this is the eigen-basis entry point, but [system] basis = "
                         f"{system['basis']!r}; use spinbasis/run_2D_spinbasis.py")
    dimensions = system["dimensions"]
    L = system["L"]
    J2 = system["J2"]
    J1 = system["J1"]
    sign_rule = system["sign_rule"]
    transl_invariant = system["transl_invariant"]

    # Sampler
    global SAMPLER, N_SAMPLES, seed, chains, discard_per_chain, sweep_size
    sampler = config["sampler"]
    SAMPLER = sampler["SAMPLER"]

    # Number of Monte Carlo samples per step, stated directly.
    #
    # This used to be MONTECARLO, a multiplier against a hidden 2048, so the
    # config said "4" when the run actually drew 8192 samples. The number that
    # matters is now the one in the file. A config.json written before this
    # change replays unchanged via the fallback.
    if "N_SAMPLES" in sampler:
        N_SAMPLES = int(sampler["N_SAMPLES"])
    else:
        N_SAMPLES = int(2048 * sampler["MONTECARLO"])
    seed = sampler["seed"]
    chains = sampler["chains"]
    discard_per_chain = sampler["discard_per_chain"]
    # Metropolis moves between recorded samples (metro sampler only). NetKet's
    # default is hilbert.size = the number of CLUSTERS (9 on 6x6), which left
    # the energy autocorrelation at tau ~ 7-10. Optional: absent or -1 keeps
    # that default, so older configs replay unchanged.
    sweep_size = int(sampler.get("sweep_size", -1))

    # Distribution
    # `alpha` is the INITIAL overdispersion exponent. With auto_alpha=True it is
    # only a starting point: VMC_NG's `auto_is` machinery then moves alpha every
    # step by ascending the gradient of the gradient-SNR, so q = |psi|^alpha
    # tracks the wavefunction as it sharpens instead of sitting at a value
    # guessed up front. All the tuning knobs are optional, so a config.json
    # written before this change replays with the old fixed-alpha behaviour.
    global alpha, use_ntk, on_the_fly
    global auto_alpha
    distr = config["distribution"]
    alpha = distr["alpha"]
    use_ntk = distr["use_ntk"]
    on_the_fly = distr["on_the_fly"]
    # optional => a config.json written before this change keeps the old
    # fixed-alpha behaviour
    auto_alpha = bool(distr.get("auto_alpha", False))

    # Model
    global iters, layers, ratio, heads, atn_mode, embedding
    model = config["model"]
    iters = model["iter"]
    layers = model["layers"]
    ratio = model["ratio"]
    heads = model["heads"]
    atn_mode = model["attention_mode"]
    embedding = model.get("embedding", "learnable")   # 'learnable' | 'qn'
    if embedding not in ("learnable", "qn"):
        raise ValueError(f"[model] embedding must be 'learnable' or 'qn', got {embedding!r}")

    # Symmetry projection (optional; absent in configs written before this
    # change, which then keep the old unsymmetrised behaviour).
    global symm_elements, symm_spin_flip
    symm = config.get("symmetry", {})
    symm_elements = tuple(symm.get("elements", []))
    symm_spin_flip = bool(symm.get("spin_flip", False))
    bad = [e for e in symm_elements if e not in C4V_ELEMENTS]
    if bad:
        raise ValueError(f"[symmetry] unknown element(s) {bad}; "
                         f"valid names are {list(C4V_ELEMENTS)}")
    if symm_elements and "I" not in symm_elements:
        raise ValueError("[symmetry] elements must contain the identity 'I'.")

    # Optimizer
    global learn_rate1, learn_rate2, diag_shift1, diag_shift2, lr_step, ds_step
    opt = config["optimizer"]
    learn_rate1 = opt["initial_learning_rate"]
    learn_rate2 = opt["final_learning_rate"]
    diag_shift1 = opt["initial_diagonal_shift"]
    diag_shift2 = opt["final_diagonal_shift"]
    lr_step = opt["lr_transition_steps"]
    ds_step = opt["ds_transition_steps"]

    # Run
    global max_runtime, save_every, print_every, diag_every, chunk_size
    run = config["run"]
    max_runtime = run["MAX_RUNTIME_MIN"]
    save_every = run["SAVE_EVERY"]
    print_every = run["PRINT_EVERY"]
    diag_every = run["DIAG_EVERY"]
    chunk_size = run["CHUNK_SIZE"]

    # Misc
    global EXACT_SAMPLER_MAX_STATES
    misc = config["misc"]
    EXACT_SAMPLER_MAX_STATES = misc["EXACT_SAMPLER_MAX_STATES"]

parse_config()

# Config checks and derived parameters
assert L % 2 == 0, "L must be even (2x2 clusters)"
b = 4
Lx = Ly = L
Cx, Cy = Lx // 2, Ly // 2
sign_rule = [c.upper() == 'T' for c in sign_rule]
assert len(sign_rule) == 2, "sign_rule must be 2 chars T/F, e.g. 'FF' for [J1,J2]"
sr_tag = ''.join('T' if s else 'F' for s in sign_rule)   # e.g. 'FF'
d_model = ratio * heads                        # divisible by n_heads by construction
n_discard = discard_per_chain if discard_per_chain >= 0 else None

# Log params for debugging purposes
print(f'L={Lx}x{Ly}  clusters={Cx}x{Cy}={Cx*Cy}  J1={J1} J2={J2}  sign_rule={sr_tag} {sign_rule}  iters={iters} (total)')
print(f'N_samples={N_SAMPLES}  seed={seed}  '
      f'n_chains={chains}  n_discard_per_chain={n_discard if n_discard is not None else "auto"}')
print(f'vit: [layers,r,heads]=[{layers},{ratio},{heads}] d_model={d_model} '
      f'attn={atn_mode} transl_inv={transl_invariant} embedding={embedding}')
print(f'lr=[{learn_rate1},{learn_rate2}] ds=[{diag_shift1},{diag_shift2}] trans(lr/ds)={lr_step}/{ds_step}')
# Adaptive alpha has two hard requirements in advanced_drivers, both of which
# fail deep inside a jit if you get them wrong, so check them here instead.
#   * on_the_fly=True routes the update through `srt_onthefly`, which does not
#     accept the `is_jac` argument the SNR derivative is carried in.
#   * `_compute_snr_derivative` reads the jacobian as interleaved real/imag rows
#     (O_L.T[:, ::2] / [1::2]); in mode='real' that slicing is silently wrong.
if auto_alpha:
    if on_the_fly:
        raise ValueError("auto_alpha=True is incompatible with on_the_fly=True "
                         "(srt_onthefly takes no is_jac argument). Set "
                         "[distribution] on_the_fly = false.")

print(f'driver: overdispersed VMC_NG  ALPHA={alpha} '
      f'({"initial, auto_is tunes it" if auto_alpha else "fixed"})  use_ntk={use_ntk}  '
      f'on_the_fly={on_the_fly}  chunk_size={chunk_size}', flush=True)


# ===========================================================================
# 2x2 plaquette cluster ops + 2D Hamiltonian   (inline; validated construction)
#
# THE LOCAL BASIS IS CANONICAL, not whatever LAPACK returned.
# Diagonalising the plaquette h leaves three things undetermined:
#   1. which eigenvalue each column belongs to  -> fixed by h itself;
#   2. which vector inside a DEGENERATE block   -> eigh returns an arbitrary
#      orthogonal mixture, chosen by LAPACK's pivoting;
#   3. the overall SIGN of each column          -> arbitrary even for
#      non-degenerate levels; eigh makes no promise.
# The old `eigh(h + 1e-4*sz_total)` only partially addressed (2) and ignored
# (3). Sz_tot cannot split states that share an (E, Sz): for J1=1, J2=0 the
# 7-fold E=0 level of the 4-ring is 1 singlet + 2 triplets, so Sz alone leaves
# blocks of size 2, 3, 2 -- five degenerate pairs whose eigenvectors were a
# LAPACK-dependent, non-symmetry-adapted mixture.
#
# Fix: exact hierarchical simultaneous diagonalisation against a mutually
# commuting set of operators that all commute with h (verified numerically):
#   Sz_tot, S2_tilde, sigma_d.
# Note S2_tilde, NOT plain S^2: with sign_rule[0]=True we are in the MARSHALL
# frame, where ||[h, S^2]|| = 2 but ||[h, U_M^dag S^2 U_M]|| = 0 exactly.
# U_M is the pi z-rotation on sublattice B. Under k = dx + 2*dy the plaquette
# bipartition is A={0,3}, B={1,2}: all four J1 bonds are A-B and both J2
# diagonals are intra-sublattice, which is why sign_rule[1] is False.
#
# S2_tilde peels the singlet off the two triplets; C2 is useless (C2 = C4^2
# and both triplets sit at C2 = -1); sigma_d finishes the job. The leftover
# pair is a 2D E-irrep of C4v, and sigma_d is the one operator here that does
# NOT commute with C4 -- so no basis is adapted to both the rotation and the
# reflection and a choice is unavoidable. sigma_d is chosen because it keeps v
# REAL (the C4-momentum alternative has eigenvalues +-i). Either is fine; the
# point is that it is FIXED and reproducible.
#
# Measured on the exact 4x4 ground state, vs the old eps-lifted basis:
# support 7394 -> 2814 configs, reweighting ESS(alpha=0.6) 0.044 -> 0.070,
# std(log|psi|) under q 1.61 -> 1.45, <H> identical to 10 digits (it is a
# unitary change of local basis, so the physics is untouched).
#
# NOTE this changes the local labelling, so .mpack checkpoints written with the
# old basis are NOT transferable: the ViT's learned local embedding is tied to
# the old column order. Start such runs fresh.
# ===========================================================================
def _perm_op_2x2(pi):
    """16x16 permutation moving the content of site k to site pi[k].

    Basis index convention matches `site()` in get_ops_2x2: site 0 is the most
    significant factor, i.e. b = s0*8 + s1*4 + s2*2 + s3.
    """
    P = np.zeros((16, 16))
    for b in range(16):
        s = [(b >> (3 - k)) & 1 for k in range(4)]
        t = [0] * 4
        for k in range(4):
            t[pi[k]] = s[k]
        P[sum(t[k] << (3 - k) for k in range(4)), b] = 1.0
    return P


def _refine_blocks(blocks, A, tol=1e-8):
    """Split each still-degenerate block of columns by diagonalising A inside it.

    Only ever rotates WITHIN a degenerate block of the previous operator, so
    every label applied earlier (starting with h) stays exactly diagonal.
    """
    out = []
    for blk in blocks:
        Ab = blk.conj().T @ A @ blk
        w, u = np.linalg.eigh(0.5 * (Ab + Ab.conj().T))
        nb, st = blk @ u, 0
        for i in range(1, len(w) + 1):
            if i == len(w) or w[i] - w[i - 1] > tol:
                out.append(nb[:, st:i]); st = i
    return out


def get_ops_2x2(sign_rule, J1, J2, tol=1e-8, verbose=True, return_qn=False,
                return_basis=False):
    I2 = np.eye(2)
    sz = 0.5 * np.array([[1, 0], [0, -1]])
    sx = 0.5 * np.array([[0, 1], [1, 0]]); sy = 0.5 * np.array([[0, -1j], [1j, 0]])
    sp = np.real(sx + 1j * sy); sm = np.real(sx - 1j * sy)
    site = lambda o, k: reduce(np.kron, [o if s == k else I2 for s in range(4)])
    SZ = [site(sz, k) for k in range(4)]; SP = [site(sp, k) for k in range(4)]; SM = [site(sm, k) for k in range(4)]
    SX = [site(sx, k) for k in range(4)]; SY = [site(sy, k) for k in range(4)]
    sz_total = sum(SZ)

    def bond(i, j, sr):
        zz = SZ[i] @ SZ[j]; exch = 0.5 * (SP[i] @ SM[j] + SM[i] @ SP[j])
        return (zz - exch) if sr else (zz + exch)

    h = np.zeros((16, 16))
    for (i, j) in [(0, 1), (2, 3), (0, 2), (1, 3)]:   # J1: square edges
        h += J1 * bond(i, j, sign_rule[0])
    for (i, j) in [(0, 3), (1, 2)]:                    # J2: diagonals
        h += J2 * bond(i, j, sign_rule[1])

    # ---- candidate labels ------------------------------------------------
    # Plain total S^2, and total S^2 in the MARSHALL frame. U_M is the pi
    # z-rotation on sublattice B: under k = dx + 2*dy the plaquette
    # bipartition is A={0,3}, B={1,2}, so all four J1 bonds are A-B and both
    # J2 diagonals are intra-sublattice (which is why sign_rule[1] is
    # normally False). WHICH of the two commutes with h depends on sign_rule,
    # so we do not guess -- see the numerical filter below.
    S2 = np.real(sum(SX) @ sum(SX) + sum(SY) @ sum(SY) + sum(SZ) @ sum(SZ))
    U_M = np.diag(np.real(np.exp(1j * np.pi * (np.diag(SZ[1]) + np.diag(SZ[2])))))
    S2_marshall = np.real(U_M.conj().T @ S2 @ U_M)
    C2 = _perm_op_2x2([3, 2, 1, 0])        # 180 deg: 0<->3, 1<->2  (preserves A,B)
    sigma_d = _perm_op_2x2([0, 2, 1, 3])   # reflect through the 0-3 diagonal (1<->2)

    candidates = [("Sz_tot", sz_total), ("S2", S2), ("S2_marshall", S2_marshall),
                  ("C2", C2), ("sigma_d", sigma_d)]
    symm = [(n, A) for n, A in candidates if np.abs(h @ A - A @ h).max() < 1e-10]
    rest = [(n, A) for n, A in candidates if (n, A) not in symm]

    # ---- exact hierarchical simultaneous diagonalisation (no epsilon) -----
    w, u = np.linalg.eigh(h)
    blocks, st = [], 0
    for i in range(1, 17):
        if i == 16 or w[i] - w[i - 1] > tol:
            blocks.append(u[:, st:i]); st = i
    used = []
    for name, A in symm:
        if all(bl.shape[1] == 1 for bl in blocks):
            break
        blocks = _refine_blocks(blocks, A, tol); used.append((name, A))
    # Anything left is an accidental degeneracy no symmetry of h resolves. Fall
    # back to the remaining candidates purely as deterministic tie-breakers:
    # the basis stays reproducible, but it is no longer symmetry-adapted.
    fallback = []
    for name, A in rest:
        if all(bl.shape[1] == 1 for bl in blocks):
            break
        blocks = _refine_blocks(blocks, A, tol); used.append((name, A)); fallback.append(name)
    if not all(bl.shape[1] == 1 for bl in blocks):
        raise RuntimeError(
            f"plaquette basis still degenerate (block sizes "
            f"{[bl.shape[1] for bl in blocks]}) at J1={J1} J2={J2} "
            f"sign_rule={sign_rule}, after labels {[n for n, _ in used]}. "
            f"Add another operator commuting with h, or the local basis is "
            f"not well defined and the run would depend on LAPACK internals."
        )
    v = np.hstack(blocks)

    # ---- deterministic column sign: largest-|.| component positive -------
    for n in range(16):
        j = np.argmax(np.abs(v[:, n]) - 1e-12 * np.arange(16))
        if v[j, n] < 0:
            v[:, n] *= -1

    # ---- canonical column ORDER, so the local labels are reproducible too -
    qn = np.round([[v[:, n] @ A @ v[:, n] for A in [h] + [A for _, A in used]]
                   for n in range(16)], 9) + 0.0        # +0.0 normalises -0.0
    order = np.lexsort(tuple(qn[:, ::-1].T))            # primary key = energy
    v = v[:, order]

    if verbose:
        print(f'plaquette basis: symmetry labels {[n for n, _ in used if n not in fallback]}'
              + (f'  + TIE-BREAKERS {fallback} (NOT symmetries of h -- basis is '
                 f'reproducible but not symmetry-adapted)' if fallback else '')
              + f'  [available symmetries: {[n for n, _ in symm]}]', flush=True)

    rot = lambda o: np.around(np.conj(v.T) @ o @ v, 12)
    ops = [rot(o) for o in ([h] + SZ + SP + SM + [sz_total])]
    if not return_qn:
        return (ops, v) if return_basis else ops

    # ---- physical labels of the 16 local states: (E, Sz, S~, sigma_d, C2) ----
    # Reported on a FIXED set of five operators regardless of which subset the
    # splitting above happened to need, so the table means the same thing for
    # every (J1, J2, sign_rule). Used by the quantum-number embedding in
    # helpers.multilayer_2D_vit_Eigenbasis.ViT_2D_EB(embedding='qn').
    #
    # For the spin label we report whichever S^2 variant is conserved: plain
    # S^2 in the un-rotated frame, S^2 in the Marshall frame when sign_rule[0]
    # flips the J1 exchange. With sign_rule[1]=True neither commutes with h and
    # the spin is genuinely not a good quantum number -- we still return the
    # expectation value (a usable feature) but say so.
    s2_ok = [(nm, A) for nm, A in (("S2", S2), ("S2_marshall", S2_marshall))
             if np.abs(h @ A - A @ h).max() < 1e-10]
    s2_name, s2_op = s2_ok[0] if s2_ok else ("S2_marshall (NOT conserved)", S2_marshall)
    exp = lambda A: np.array([v[:, n] @ A @ v[:, n] for n in range(16)])
    s2_val = np.clip(exp(s2_op), 0.0, None)
    qn_table = np.round(np.stack([
        exp(h),                                   # E      plaquette energy
        exp(sz_total),                            # Sz     total magnetisation
        (-1.0 + np.sqrt(1.0 + 4.0 * s2_val)) / 2, # S~     total spin from S(S+1)
        exp(sigma_d),                             # +-1    diagonal mirror parity
        exp(C2),                                  # +-1    180 deg rotation parity
    ], axis=1), 9) + 0.0
    if verbose:
        sharp = np.abs(np.array([v[:, n] @ (s2_op @ s2_op) @ v[:, n] for n in range(16)])
                       - exp(s2_op) ** 2).max()
        print(f'plaquette quantum numbers: (E, Sz, S~, sigma_d, C2) using {s2_name}'
              f'  [max spin spread {sharp:.1e}]', flush=True)
    # v is the local eigenbasis itself; helpers.symmetries needs it to build the
    # C4v / spin-flip gather tables.
    return (ops, qn_table, v) if return_basis else (ops, qn_table)


def get_piecewise_Hamiltonian_2D(new_ops, J1, J2, sign_rule):
    (hlocal, sz0, sz1, sz2, sz3, sp0, sp1, sp2, sp3, sm0, sm1, sm2, sm3, sz_tot) = new_ops
    SZ = [sz0, sz1, sz2, sz3]; SP = [sp0, sp1, sp2, sp3]; SM = [sm0, sm1, sm2, sm3]

    def hb(A, C, sr):
        zz = np.kron(SZ[A], SZ[C]); exch = 0.5 * np.kron(SP[A], SM[C]); exch = exch + exch.T
        return (zz - exch) if sr else (zz + exch)

    s1, s2 = sign_rule
    right = J1 * (hb(1, 0, s1) + hb(3, 2, s1)) + J2 * (hb(1, 2, s2) + hb(3, 0, s2))
    up = J1 * (hb(2, 0, s1) + hb(3, 1, s1)) + J2 * (hb(2, 1, s2) + hb(3, 0, s2))
    up_right = J2 * hb(3, 0, s2); down_right = J2 * hb(1, 2, s2)
    return [hlocal], [right, up, up_right, down_right]


def get_cluster_graph_2D_J2(Cx, Cy, PBC=True):
    if PBC and (Cx < 2 or Cy < 2):
        raise ValueError("Under PBC need Cx>=2 and Cy>=2.")
    node = lambda cx, cy: (cx % Cx) + Cx * (cy % Cy)
    edges = []
    for cy in range(Cy):
        for cx in range(Cx):
            u = node(cx, cy)
            if PBC or cx + 1 < Cx:                    edges.append((u, node(cx + 1, cy), 0))
            if PBC or cy + 1 < Cy:                    edges.append((u, node(cx, cy + 1), 1))
            if PBC or (cx + 1 < Cx and cy + 1 < Cy):  edges.append((u, node(cx + 1, cy + 1), 2))
            if PBC or (cx + 1 < Cx and cy - 1 >= 0):  edges.append((u, node(cx + 1, cy - 1), 3))
    return GraphCust(n_nodes=Cx * Cy, edges=edges)

@register_pytree_node_class
class Heisenberg_cluster_2D(GraphOperatorCust):
    def __init__(self, hilbert, graph, site_ops, bond_ops,
                 bond_ops_colors=(0, 1, 2, 3), J=(1.0, 1.0),
                 sign_rule=(False, False), dtype=None, acting_on_subspace=None):
        self._J = J; self._sign_rule = sign_rule
        self.site_ops = site_ops; self.bond_ops = bond_ops
        super().__init__(hilbert, graph, bond_ops=bond_ops, site_ops=site_ops,
                         bond_ops_colors=list(bond_ops_colors),
                         dtype=dtype, acting_on_subspace=acting_on_subspace)


new_ops, qn_table, eigenbasis = get_ops_2x2(sign_rule, J1, J2, return_qn=True,
                                            return_basis=True)
hlocal, sz_tot = new_ops[0], new_ops[-1]
site_ops, bond_ops = get_piecewise_Hamiltonian_2D(new_ops, J1, J2, sign_rule)
graph = get_cluster_graph_2D_J2(Cx, Cy, PBC=True)
hi = EigenHilbert(total_sz=0, N=graph.n_nodes, Nlocal=b, Sz_operator=sz_tot)
ha = Heisenberg_cluster_2D(hi, graph, site_ops=site_ops, bond_ops=bond_ops,
                           bond_ops_colors=[0, 1, 2, 3], J=[J1, J2], sign_rule=sign_rule)
ha_jax = ha.to_jax_operator()

try:
    n_states = hi.n_states
except Exception:
    n_states = None
print('n_states (Sz=0) =', n_states if n_states is not None else 'too large to index')

# resolve auto sampler now that we know the sector size
if SAMPLER == 'auto':
    sampler_name = 'exact' if (n_states is not None and n_states <= EXACT_SAMPLER_MAX_STATES) else 'metro'
else:
    sampler_name = SAMPLER


# ===========================================================================
# ED reference (small systems only)
# ===========================================================================
E0 = None
if n_states is not None and n_states <= EXACT_SAMPLER_MAX_STATES:
    t = time.time()
    E0 = float(sla.eigsh(ha.to_sparse(), k=1, which='SA', return_eigenvectors=False)[0])
    print(f'ED ground-state energy  E0 = {E0:.8f}   ({time.time()-t:.1f}s)', flush=True)
else:
    print('ED skipped (sector too large).')


# ===========================================================================
# output paths + sentinels
# ===========================================================================

fname = os.path.join(DATA_ROOT, "run")
print(f'\n{fname=}')

PROGRESS = fname + '.progress'
DONE = fname + '.done'
CKPT = fname + '.mpack'
TIMEOUT = fname + '.timeout'
CONFIG = fname + '.config.json'
DIAG = fname + '.diag.jsonl'
if os.path.exists(TIMEOUT):
    os.remove(TIMEOUT)

def _read_progress_json():
    if os.path.exists(PROGRESS):
        try:
            with open(PROGRESS) as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def read_progress():
    return int(_read_progress_json().get('cum_steps', 0))

def read_progress_alpha():
    """Adaptive alpha lives OUTSIDE vstate.variables, so the .mpack checkpoint
    does not carry it. Without this the exponent would re-anneal from the
    config value on every resubmission, exactly the bug the offset-aware lr /
    diag_shift schedules exist to avoid. Returns None when absent (fresh run,
    or a .progress written before adaptive alpha existed)."""
    a = _read_progress_json().get('alpha', None)
    return None if a is None else float(a)

def write_progress(cum, alpha_now=None):
    tmp = PROGRESS + '.tmp'
    rec = {'cum_steps': int(cum)}
    if alpha_now is not None:
        rec['alpha'] = float(alpha_now)
    with open(tmp, 'w') as f:
        json.dump(rec, f)
    os.replace(tmp, PROGRESS)

jobid = os.environ.get('SLURM_JOB_ID', str(os.getpid()))
with open(os.path.join(DATA_ROOT, f'.runinfo_{jobid}'), 'w') as f:
    f.write(fname + '\n')


# ===========================================================================
# fresh vs resume
# ===========================================================================
do_resume = os.path.exists(CKPT)

# The .diag.jsonl is append-only and keyed on the GLOBAL step, so it continues
# cleanly across resubmissions. On a fresh (non-resume) run, start it empty.
if not do_resume and os.path.exists(DIAG):
    os.remove(DIAG)

step_offset = read_progress() if do_resume else 0
print(f'do_resume={do_resume}  step_offset={step_offset}')
remaining = max(0, iters - step_offset)
if remaining == 0:
    print(f'\n*** Target {iters} already reached (cum={step_offset}). Writing .done. ***')
    open(DONE, 'w').close(); sys.exit(0)
print(f'remaining this submission = {remaining}')

# ===========================================================================
# OFFSET-AWARE linear schedules for lr and diag_shift  (the requested feature)
# ===========================================================================
_lr = optax.linear_schedule(init_value=learn_rate1, end_value=learn_rate2, transition_steps=lr_step)
_ds = optax.linear_schedule(init_value=diag_shift1, end_value=diag_shift2, transition_steps=ds_step)
lr = lambda c: _lr(c + step_offset)
diag_shift = lambda c: _ds(c + step_offset)

print(f'\nSchedules @ global step {step_offset}:')
print(f'  lr         = {float(lr(0)):.4e}   ({learn_rate1} -> {learn_rate2} over {lr_step})')
print(f'  diag_shift = {float(diag_shift(0)):.4e}   ({diag_shift1} -> {diag_shift2} over {ds_step})', flush=True)


# ===========================================================================
# model + sampler + variational state
# ===========================================================================
# qn_table must be HASHABLE to be a flax static attribute -> nested tuple.
# Ignored entirely when embedding='learnable'.
qn_static = tuple(tuple(float(x) for x in row) for row in qn_table)
model = ViT_2D_EB(num_layers=layers, d_model=d_model, n_heads=heads,
                  local_dim=2 ** b, attn_mode=atn_mode,
                  transl_invariant=transl_invariant,
                  embedding=embedding, qn_table=qn_static)
if embedding == 'qn':
    print('quantum-number embedding: (E, Sz, S~, sigma_d, C2) per local state')
    print(f"{'n':>4} {'E':>8} {'Sz':>6} {'S~':>5} {'sigma_d':>8} {'C2':>5}")
    for n, row in enumerate(qn_table):
        print(f'{n:>4} {row[0]:+8.3f} {row[1]:+6.1f} {row[2]:5.1f} {row[3]:+8.1f} {row[4]:+5.1f}')
    print(flush=True)

# ---------------------------------------------------------------------------
# Optional C4v / spin-flip projection.
#
# Cluster TRANSLATIONS are already exact via the circulant attention; this adds
# the point group, following Eq. (A1) of Viteritti et al. Costs |G| forward
# passes per sample, so the paper's staged protocol (Appendix D) -- optimise
# unsymmetrised, switch symmetries on for a short final phase -- is usually the
# right way to spend the budget.
#
# NOTE this changes the parameter tree (everything moves under a submodule), so
# .mpack checkpoints are NOT interchangeable between symmetrised and
# unsymmetrised runs. Turning this on mid-run means starting fresh.
# ---------------------------------------------------------------------------
if symm_elements:
    inv_cperms, local_maps, local_signs, gnames = build_symmetry_tables(
        eigenbasis, Cx, elements=symm_elements, spin_flip=symm_spin_flip)
    model = SymmProject(module=model, inv_cperms=inv_cperms,
                        local_maps=local_maps, local_signs=local_signs)
    print(f'symmetry projection: |G|={len(gnames)}  {list(gnames)}')
    print(f'  -> {len(gnames)}x forward passes per sample', flush=True)
else:
    print('symmetry projection: OFF (cluster translations only)', flush=True)

if sampler_name == 'exact':
    sampler = nk.sampler.ExactSampler(hilbert=hi)
else:
    sampler = nk.sampler.MetropolisHamiltonian(
        hilbert=hi, hamiltonian=ha_jax, n_chains=chains,
        **({"sweep_size": sweep_size} if sweep_size > 0 else {}))
    print(f'metro sampler: sweep_size={sampler.sweep_size} '
          f'({"config" if sweep_size > 0 else "NetKet default = n_clusters"})', flush=True)

# n_discard_per_chain only matters for the (Markov-chain) metro sampler; the
# exact sampler draws i.i.d. so there is nothing to burn in.
mcstate_kwargs = dict(sampler=sampler, model=model, n_samples=N_SAMPLES, seed=seed)
if sampler_name != 'exact' and n_discard is not None:
    mcstate_kwargs['n_discard_per_chain'] = n_discard
for k, v in mcstate_kwargs.items():
    print(f"key={k}, val={v}, val_type={type(v)}")
vstate = nk.vqs.MCState(**mcstate_kwargs)
if chunk_size > 0:
    vstate.chunk_size = chunk_size
print('n_params =', nk.jax.tree_size(vstate.parameters),
      ' n_discard_per_chain =', getattr(vstate, 'n_discard_per_chain', 'n/a'), flush=True)

if do_resume:
    print('\n### resuming from checkpoint ###')
    with open(CKPT, 'rb') as f:
        vstate.variables = flax.serialization.from_bytes(vstate.variables, f.read())
    # NOTE: no more .segK.log archiving — the JsonLog below appends to the one
    # continuous log instead of overwriting it.


# ===========================================================================
# driver: overdispersed VMC_NG (advanced_drivers) — mirrors z_EB_optim.py
#   - opt carries the offset-aware LINEAR lr schedule (optax.sgd)
#   - diag_shift is the offset-aware LINEAR schedule callable
#   - sampling is overdispersed with exponent alpha (in the savename + logs)
# ===========================================================================
opt = optax.sgd(learning_rate=lr)

# Adaptive alpha is entirely the package's: `auto_is=True` makes VMC_NG compute
# d(SNR)/d(alpha) each step and call `overdispersed_distribution.update_params`
# with it. This is the usage in importance_sampling_nqs/notebooks/j1j2_vmc.ipynb.
modulus_distribution = advd.driver.overdispersed_distribution(alpha=alpha)

# Resume the exponent where the previous submission left it (see
# read_progress_alpha). Only meaningful for an adaptive run: with a fixed alpha
# the config value IS the truth and must win, e.g. if you edit it and resubmit.
if auto_alpha and do_resume:
    _a_prev = read_progress_alpha()
    if _a_prev is not None:
        modulus_distribution.q_variables = {"alpha": jnp.array([_a_prev])}
        print(f'resumed adaptive alpha = {_a_prev:.4f} (config initial was {alpha})')
    else:
        print(f'no alpha in .progress; adaptive alpha restarts from {alpha}')

driver = advd.driver.VMC_NG(hamiltonian=ha_jax,
                            optimizer=opt,
                            sampling_distribution=modulus_distribution,
                            variational_state=vstate,
                            diag_shift=diag_shift,
                            auto_is=auto_alpha,
                            use_ntk=use_ntk,
                            on_the_fly=on_the_fly)

if auto_alpha and driver.mode != 'complex':
    raise ValueError(f"auto_alpha=True needs the driver in mode='complex' "
                     f"(the SNR derivative assumes interleaved real/imag "
                     f"jacobian rows), but the driver picked mode={driver.mode!r} "
                     f"from the model dtype.")


def current_alpha():
    """Live overdispersion exponent as a scalar (mean over components, so the
    mixture form would also report something sane)."""
    try:
        return float(np.asarray(modulus_distribution.q_variables["alpha"]).mean())
    except Exception:
        return float(alpha)


# ===========================================================================
# callbacks
# ===========================================================================
class ProgressWriter:
    def __init__(self, every, print_every, E0):
        self.every = every
        self.print_every = print_every
        self.E0 = E0

    def __call__(self, step, log_data, driver):
        self.step = step + 1
        a = current_alpha() if auto_alpha else None
        if (step + 1) % self.every == 0:
            write_progress(self.step, a)
        if step % self.print_every == 0:
            try:
                e = float(np.real(log_data["Energy"].mean))
                tail = f'  alpha={a:.4f}' if a is not None else ''
                if self.E0 is not None:
                    print(f'[{self.step:6d}] E={e:.8f}  E0={self.E0:.8f}  '
                          f'rel.err={abs(e - self.E0) / abs(self.E0):.3e}{tail}', flush=True)
                else:
                    print(f'[{self.step:6d}] E={e:.8f}{tail}', flush=True)
            except Exception:
                pass
        return True

class TimeBudget:
    def __init__(self, minutes):
        self.deadline = time.time() + minutes * 60 if minutes > 0 else None
        self.stopped = False
    def __call__(self, step, log_data, driver):
        if self.deadline is not None and time.time() > self.deadline:
            self.stopped = True
            print(f'\n*** wall-clock budget reached at local step {step}; stopping gracefully ***', flush=True)
            return False
        return True

class SamplingDiag:
    """Cheap sampling-health diagnostics -> <fname>.diag.jsonl (append-only,
    keyed on the GLOBAL step). One extra forward pass (log|psi| of the current
    samples) every `every` steps; everything else is free.

    Logged fields:
      ess_w / ess_w_frac : effective sample size of the IMPORTANCE weights
                           w ∝ |psi|^(2-alpha). This is the overdispersion (alpha)
                           health check — a collapse here is the alpha=0.6 failure.
                           At alpha=2 the weights are identically 1, so ess_w should
                           report ~n_samples. NOTE this recomputes the weights from
                           the samples, independently of the driver, so it is also a
                           cross-check that the driver is really sampling |psi|^alpha.
      rhat_logpsi        : split-R-hat of log|psi| ACROSS the MCMC chains. ~1.0 =
                           chains agree; >>1 = chains stuck in different modes / not
                           mixing, i.e. effective N well below nominal.
      acceptance         : Metropolis acceptance fraction (None for the exact sampler).
      var_eloc / _per_site : Var(E_loc) and Var(E_loc)/N_sites, from the driver's own
                           energy stats (free) — the size-fair variance for 4x4 vs 6x6.

    A broken diagnostic must never kill a run, so every step is guarded; on any
    failure it records the error string and continues.
    """
    def __init__(self, alpha, every, diag_path, n_sites):
        # `alpha` here is only the fallback: the importance weights must be
        # recomputed with whatever exponent the driver is sampling at RIGHT NOW,
        # otherwise ess_w silently drifts into nonsense once alpha starts moving.
        self.alpha = float(alpha)
        self.every = max(1, int(every))
        self.path = diag_path
        self.n_sites = int(n_sites)

    def _alpha(self, driver):
        try:
            return float(np.asarray(
                driver.sampling_distribution.q_variables["alpha"]).mean())
        except Exception:
            return self.alpha

    @staticmethod
    def _split_rhat(x):
        # x: (n_chains, n_per_chain) real -> standard split-R-hat (Gelman-Rubin).
        x = np.asarray(x, dtype=float)
        if x.ndim != 2 or x.shape[0] < 2 or x.shape[1] < 4:
            return float('nan')
        m = x.shape[1] // 2
        h = np.concatenate([x[:, :m], x[:, m:2 * m]], axis=0)   # (2*n_chains, m)
        n = h.shape[1]
        means = h.mean(axis=1); vars = h.var(axis=1, ddof=1)
        W = float(means.size and vars.mean())
        B = float(n * means.var(ddof=1))
        if W <= 0:
            return float('nan')
        var_hat = (n - 1) / n * W + B / n
        return float(np.sqrt(var_hat / W))

    def __call__(self, step, log_data, driver):
        if step % self.every:
            return True
        # `step` is already GLOBAL (driver._step_count is seeded with
        # step_offset below), so adding step_offset here would double-count it.
        gstep = step + 1
        rec = {"step": int(gstep), "alpha": self._alpha(driver)}
        # --- free: energy variance from the driver's own stats ---
        try:
            v = float(np.real(log_data["Energy"].variance))
            rec["var_eloc"] = v
            rec["var_eloc_per_site"] = v / self.n_sites
        except Exception:
            pass
        vs = driver.state
        # The chain to diagnose is the one the GRADIENT used: q = |psi|^alpha,
        # NOT vs.samples (which is the |psi|^2 chain and, being invalidated every
        # iteration, would trigger a whole extra MCMC pass here). The driver just
        # sampled q this step, so this call hits its cache and costs nothing.
        chain = getattr(driver.sampling_distribution, 'name', None)
        try:
            afun_q, vars_q = driver.sampling_distribution(vs._apply_fun, vs.variables)
            samp = np.asarray(vs.samples_distribution(afun_q, variables=vars_q,
                                                      chain_name=chain))
            rec["chain"] = chain
        except Exception as ex:
            samp = np.asarray(vs.samples)               # fall back to |psi|^2
            rec["chain"] = "default"
            rec["chain_error"] = repr(ex)
        # --- one forward pass: log|psi| of those samples ---
        try:
            flat = samp.reshape(-1, samp.shape[-1])
            logpsi = np.asarray(vs.log_value(flat)).real            # log|psi| (+const)
            wl = (2.0 - rec["alpha"]) * logpsi
            wl = wl - wl.max()                                      # normalization-invariant
            w = np.exp(wl)
            sw = float(w.sum()); sw2 = float(np.square(w).sum())
            rec["ess_w"] = (sw * sw / sw2) if sw2 > 0 else float('nan')
            rec["ess_w_frac"] = rec["ess_w"] / w.size
            rec["n_samples"] = int(w.size)
            if samp.ndim >= 3:                                      # chain structure present
                nchains = int(np.prod(samp.shape[:-2])); nper = samp.shape[-2]
                rec["rhat_logpsi"] = self._split_rhat(logpsi.reshape(nchains, nper))
        except Exception as ex:
            rec["logpsi_error"] = repr(ex)
        # --- free-ish: Metropolis acceptance (guarded across NetKet versions) ---
        # again for the q chain: netket_pro keeps one sampler state per chain.
        try:
            ss = getattr(vs, 'sampler_states', {}).get(chain) or vs.sampler_state
            acc = getattr(ss, "acceptance", None)
            if acc is None:
                na = getattr(ss, "n_accepted", None); ns = getattr(ss, "n_steps", None)
                if na is not None and ns:
                    acc = float(np.sum(na)) / float(np.sum(ns))
            if acc is not None:
                rec["acceptance"] = float(np.asarray(acc).mean())
        except Exception:
            pass
        # --- append one line, keyed on the global step ---
        try:
            with open(self.path, "a") as f:
                f.write(json.dumps(rec) + "\n")
        except Exception:
            pass
        # compact console echo
        msg = f'    [diag {gstep:6d}]'
        if "alpha" in rec:             msg += f' alpha={rec["alpha"]:.4f}'
        if "ess_w_frac" in rec:        msg += f' ESS_w={rec["ess_w"]:.0f} ({100 * rec["ess_w_frac"]:.1f}%)'
        if "rhat_logpsi" in rec:       msg += f' Rhat|psi|={rec["rhat_logpsi"]:.3f}'
        if "acceptance" in rec:        msg += f' acc={rec["acceptance"]:.2f}'
        if "var_eloc_per_site" in rec: msg += f' Var/site={rec["var_eloc_per_site"]:.3e}'
        print(msg, flush=True)
        return True

progress_cb = ProgressWriter(save_every, print_every, E0)
time_cb = TimeBudget(max_runtime)

# sampling-health diagnostics: -1 => follow print_every, 0 => off
diag_every = print_every if diag_every < 0 else diag_every
diag_cb = None
if diag_every > 0:
    diag_cb = SamplingDiag(alpha=alpha, every=diag_every,
                           diag_path=DIAG, n_sites=Lx * Ly)
    print(f'sampling diagnostics -> {os.path.basename(DIAG)} every {diag_every} steps', flush=True)

# ---- ONE CONTINUOUS LOG ----------------------------------------------------
# 1) make the iterations the logger records GLOBAL: the driver's step counter
#    otherwise restarts at 0 every submission, which is what capped the plotted
#    x-axis at one wall-clock window. Setting it to step_offset means this
#    submission logs iters step_offset .. step_offset+remaining-1.
#    (This does NOT affect the optax schedules — those use the optimizer's own
#    per-run counter plus step_offset, independently of driver._step_count.)
try:
    driver._step_count = int(step_offset)
    print(f'driver step counter set to global offset {step_offset}', flush=True)
except Exception as e:
    print(f'WARNING: could not set driver._step_count ({e}); '
          f'log iters may restart at 0 this submission.', flush=True)

if do_resume:
    fname += " step " + str(step_offset)
# 2) append on a resume so every window lands in the SAME .log file.
logger = nk.logging.JsonLog(fname,
                            mode='write',
                            save_params_every=save_every, 
                            write_every=save_every)

driver.run(n_iter=remaining, out=logger,
           callback=[InvalidLossStopping(monitor='mean', patience=5), progress_cb, time_cb]
                    + ([diag_cb] if diag_cb is not None else []))


# ===========================================================================
# finalize  (checkpoint matched to recorded progress; sentinels for resubmit)
# ===========================================================================
with open(CKPT, 'wb') as f:
    f.write(flax.serialization.to_bytes(vstate.variables))
alpha_final = current_alpha()
write_progress(progress_cb.step, alpha_final if auto_alpha else None)

Ef = float(np.real(vstate.expect(ha_jax).mean))
print('\n===== summary =====')
if auto_alpha:
    print(f'alpha: {alpha} (initial) -> {alpha_final:.4f} (final, adaptive)  '
          f'use_ntk={use_ntk}  on_the_fly={on_the_fly}  (overdispersed VMC_NG)')
else:
    print(f'alpha={alpha} (fixed)  use_ntk={use_ntk}  on_the_fly={on_the_fly}  (overdispersed VMC_NG)')
print(f'final VMC energy E = {Ef:.8f}')
if E0 is not None:
    print(f'ED reference    E0 = {E0:.8f}   rel.err = {abs(Ef - E0) / abs(E0):.3e}')
    print(f'E/site (VMC) = {Ef/(Lx*Ly):.8f}   E0/site = {E0/(Lx*Ly):.8f}')
else:
    print(f'E/site (VMC) = {Ef/(Lx*Ly):.8f}')

with open(fname + '.summary.json', 'w') as f:
    json.dump({"basis": "eigen", "L": Lx, "J2": J2, "E_vmc": Ef, "E0": E0,
               "rel_err": (abs(Ef - E0) / abs(E0)) if E0 is not None else None,
               "alpha": alpha, "alpha_final": alpha_final, "auto_alpha": bool(auto_alpha),
               "use_ntk": bool(use_ntk), "on_the_fly": bool(on_the_fly),
               "embedding": embedding,
               "chunk_size": chunk_size, "n_discard_per_chain": n_discard,
               "lr_step": lr_step, "ds_trans": ds_step,
               "cum_steps": progress_cb.step, "N_iter_target": iters}, f, indent=2)

if progress_cb.step >= iters:
    open(DONE, 'w').close()
    print(f'\n*** Target reached: cum={progress_cb.step} >= {iters}. Wrote .done ***')
elif time_cb.stopped:
    open(TIMEOUT, 'w').close()
    print(f'\n*** Clean wall-clock stop at cum={progress_cb.step}/{iters}. Wrote .timeout (resubmit). ***')
else:
    print(f'\n*** Stopped early at cum={progress_cb.step}/{iters} WITHOUT clean time-stop '
          f'(divergence?). No resubmission. ***')

print('\nRuntime:', (time.time() - job_start) / 60, 'mins')
print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
jax.clear_caches()
