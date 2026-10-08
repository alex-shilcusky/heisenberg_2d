"""
learned_basis_4x4.py
====================
Is there a better local basis for the plaquette ViT than the plaquette
eigenbasis? Fit the 4x4 exact ground state while LEARNING the 16x16 local basis
jointly with the network.

sign_diagnostics_4x4.py showed that in the plaquette eigenbasis the ViT fails
on the SIGN of the ground state (given the exact signs, it fits as well as the
spin-basis ViT). Here the basis itself is a trainable orthogonal matrix,

    U = U0 @ expm(M * (A - A^T)),      columns of U = the 16 local states,

where M allows rotations only between local states with the SAME Sz. Every
local state therefore keeps a definite Sz, the Sz = 0 sector of index
configurations is unchanged, and a basis found here can be used in VMC as is
(EigenConstraint still applies). The family contains both the plaquette
eigenbasis and the plain spin product basis: Sz blocks of size 1, 4, 6, 4, 1,
so 27 free parameters.

Objective, exact (no sampling): with T(n1..n4) = exp(ViT(n)) the network
amplitudes, the represented spin-basis state is U^{x4} T. U is orthogonal, so

    fidelity = |<(U^T)^{x4} psi | T>|^2 / <T|T>,

which needs one 16^4 contraction of the exact psi per step.

Fit specs (--fits) are '<init>:<frozen|learned>':
  init   eigen   the plaquette eigenbasis (get_ops_2x2), what production uses
         spin    the computational (spin product) basis
         random  a random Sz-preserving rotation (seeded by the fit seed)
  frozen = basis fixed at init (a baseline); learned = basis trained too.

The network is ViT_2D_EB with a LEARNABLE embedding: the quantum-number
embedding means nothing once the basis moves. spin:frozen is therefore the
same spin basis as the spin-basis ViT, but fed as categorical indices. It
separates "which basis" from "how the input is encoded".

Per fit it records the infidelity (best and current; NaN-flagged), dE/E0,
and where the basis ended up:
  eigen_likeness  mean over local states of the max |overlap| with an
                  eigenbasis state (1 = the eigenbasis, up to order and sign)
  spin_likeness   the same against computational states (1 = spin basis)
  wrong_sign      weight of the exact GS whose sign no per-state rule predicts
  configs_99      configs holding 99% of the exact GS weight; a concentration
                  proxy, since concentration is what kills MCMC acceptance
plus the learned U itself (U_*.npy).

Usage:
    cluster:  DIAG=learned_basis ./submit_diag.sh
    local:    PYTHONPATH=../importance_sampling_nqs/packages .venv/bin/python \
                  learned_basis_4x4.py --j2 0.0 --seeds 0 --steps 300 --out /some/dir
"""
import argparse
import json
import os
import platform
import sys
import time

import numpy as np
import scipy.sparse.linalg as sla

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.scipy.linalg as jsl
import optax

from sign_diagnostics_4x4 import (EIGEN_DIR, HERE, NS, _git_info, _load_basis_functions, _load_module,
                                  build_hamiltonian, energy_error, wrong_sign_weight_eigen)

INITS = ("eigen", "spin", "random")


def parse_spec(spec):
    try:
        init, mode = spec.split(":")
    except ValueError:
        raise ValueError(f"fit spec {spec!r} must be '<init>:<frozen|learned>'")
    if init not in INITS or mode not in ("frozen", "learned"):
        raise ValueError(f"fit spec {spec!r}: init in {INITS}, mode in (frozen, learned)")
    return init, mode == "learned", spec


# ---------------------------------------------------------------------------
# Local bases
# ---------------------------------------------------------------------------
# Computational plaquette state b in [0,16): site k is bit (3-k), bit 0 = up.
SZ_COMP = np.array([sum(0.5 - ((b >> (3 - k)) & 1) for k in range(4)) for b in range(16)])


def initial_basis(init, eig_v, seed):
    """(U0, column Sz labels). U0's columns are local states, each of definite Sz."""
    if init == "eigen":
        U0 = np.asarray(eig_v, float)
        sz = np.round(np.einsum("bn,b,bn->n", U0, SZ_COMP, U0), 9)
    else:
        U0 = np.eye(16); sz = SZ_COMP.copy()
        if init == "random":
            rng = np.random.default_rng(1000 + seed)
            K = rng.normal(size=(16, 16)) * block_mask(sz)
            U0 = U0 @ np.asarray(jsl.expm(jnp.asarray(K - K.T)))
    return U0, sz


def block_mask(sz):
    """1 where two local states share an Sz label (rotations allowed), else 0."""
    return (np.abs(sz[:, None] - sz[None, :]) < 1e-9).astype(float)


def rotation(A, mask):
    K = mask * (A - A.T)
    return jsl.expm(K)


# ---------------------------------------------------------------------------
def fit_one(model, x_sector, flat_idx, psi_t, U0, mask, learn_basis, steps, lr, basis_lr,
            seed, checkpoints):
    """Joint fit of the ViT (and, if learn_basis, the local basis rotation)."""
    params = model.init(jax.random.PRNGKey(seed), x_sector[:2])
    n_params = sum(int(np.prod(a.shape)) for a in jax.tree_util.tree_leaves(params))
    A = jnp.zeros((16, 16))
    U0j, maskj = jnp.asarray(U0), jnp.asarray(mask)
    idx = jnp.asarray(flat_idx)

    def rotated_target(A):
        U = U0j @ rotation(A, maskj)
        Psi = jnp.einsum("an,bm,co,dp,abcd->nmop", U, U, U, U, psi_t)
        return Psi.reshape(-1)[idx], U

    def loss(params, A):
        t, _ = rotated_target(A)
        f = model.apply(params, x_sector)
        T = jnp.exp(f - jax.lax.stop_gradient(jnp.real(f).max()))
        fid = jnp.abs(jnp.sum(t * T)) ** 2 / (jnp.sum(jnp.abs(T) ** 2) * jnp.sum(t ** 2))
        return -jnp.log(fid)

    opt_n, opt_b = optax.adam(lr), optax.adam(basis_lr)
    sn, sb = opt_n.init(params), opt_b.init(A)

    @jax.jit
    def step(params, A, sn, sb):
        val, (gn, gb) = jax.value_and_grad(loss, argnums=(0, 1))(params, A)
        un, sn = opt_n.update(gn, sn, params)
        params = optax.apply_updates(params, un)
        if learn_basis:
            ub, sb = opt_b.update(gb, sb, A)
            A = optax.apply_updates(A, ub)
        return params, A, sn, sb, val

    best, best_state = np.inf, (params, A)
    curve_best, curve_now, nan_at = {}, {}, None
    for i in range(1, steps + 1):
        params, A, sn, sb, val = step(params, A, sn, sb)
        v = float(val)
        if not np.isfinite(v):
            nan_at = i
            break
        if v < best:
            best, best_state = v, (params, A)
        if i in checkpoints:
            curve_best[i] = 1 - np.exp(-best)
            curve_now[i] = 1 - np.exp(-v)
    params, A = best_state
    f = np.asarray(model.apply(params, x_sector))
    T = np.exp(f - f.real.max())
    _, U = rotated_target(A)
    return dict(n_params=n_params, infidelity=1 - np.exp(-best), curve_best=curve_best,
                curve_now=curve_now, nan_at=nan_at, T=T, U=np.asarray(U))


def basis_report(U, eig_v, psi_t):
    """Where a basis sits, and what the exact GS looks like in it."""
    Psi = np.einsum("an,bm,co,dp,abcd->nmop", U, U, U, U, psi_t)
    w = np.sort((Psi ** 2).ravel())[::-1]
    return {"eigen_likeness": float(np.abs(eig_v.T @ U).max(axis=0).mean()),
            "spin_likeness": float(np.abs(U).max(axis=0).mean()),
            "wrong_sign": float(wrong_sign_weight_eigen(Psi)),
            "configs_99": int(np.searchsorted(np.cumsum(w), 0.99) + 1),
            "orth_err": float(np.abs(U.T @ U - np.eye(16)).max())}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--j2", type=float, nargs="+", default=[0.0, 0.5])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--basis-lr", type=float, default=3e-3)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--d-model", type=int, default=72)
    ap.add_argument("--heads", type=int, default=12)
    ap.add_argument("--fits", nargs="+",
                    default=["eigen:frozen", "eigen:learned", "spin:frozen", "spin:learned", "random:learned"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--require-gpu", action="store_true")
    args = ap.parse_args()
    specs = [parse_spec(s) for s in args.fits]

    devices = [str(d) for d in jax.devices()]
    print("JAX devices:", devices, flush=True)
    if args.require_gpu and jax.default_backend() != "gpu":
        raise SystemExit(f"--require-gpu: JAX backend is {jax.default_backend()!r}, not 'gpu'.")
    out = args.out
    if out:
        os.makedirs(out, exist_ok=True)
        json.dump({"argv": sys.argv, "args": vars(args), "devices": devices,
                   "backend": jax.default_backend(), "host": platform.node(),
                   "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "jax": jax.__version__,
                   "git": _git_info(), "started": time.strftime("%Y-%m-%d %H:%M:%S")},
                  open(os.path.join(out, "meta.json"), "w"), indent=2)

    basis = _load_basis_functions()
    vit_eig = _load_module("vit_eig", os.path.join(EIGEN_DIR, "helpers", "multilayer_2D_vit_Eigenbasis.py"))
    model = vit_eig.ViT_2D_EB(num_layers=args.layers, d_model=args.d_model, n_heads=args.heads,
                              local_dim=16, attn_mode="factored", transl_invariant=True,
                              embedding="learnable", output_head="logcosh")
    checkpoints = sorted({c for c in (100, 300, 1000, 2000, 5000, args.steps) if c <= args.steps})
    grid = np.stack(np.meshgrid(*[np.arange(16)] * 4, indexing="ij"), -1).reshape(-1, 4)
    results = []

    for J2 in args.j2:
        H, up = build_hamiltonian(J2)
        w, V = sla.eigsh(H, k=1, which="SA")
        E0 = float(w[0]); psi = V[:, 0]
        psi_t = psi.reshape(16, 16, 16, 16)          # axis k = plaquette k's computational state
        psi_tj = jnp.asarray(psi_t)
        _, _, eig_v = basis["get_ops_2x2"]((False, False), 1.0, J2, verbose=False,
                                           return_qn=True, return_basis=True)
        eig_v = np.asarray(eig_v, float)
        print(f"\n=== J2 = {J2}   E0/site = {E0 / NS:+.6f}", flush=True)
        print(f"  {'fit':15s} {'seed':>4s} | infid@ " + " ".join(f"{c:>7d}" for c in checkpoints)
              + " | dE/E0    eig_like spin_like wrong_sign cfg99  (init->final)", flush=True)

        for init, learn, label in specs:
            for seed in args.seeds:
                U0, sz = initial_basis(init, eig_v, seed)
                mask = block_mask(sz)
                keep = np.abs(sz[grid].sum(axis=1)) < 1e-9
                assert keep.sum() == 12870, keep.sum()
                flat_idx = np.flatnonzero(keep)
                x_sector = jnp.asarray(grid[keep], dtype=jnp.int32)
                rep0 = basis_report(U0, eig_v, psi_t)
                # sanity: the basis is orthogonal, the GS lives entirely in the
                # Sz = 0 index sector, and rotating it back reproduces E0
                Psi0 = np.einsum("an,bm,co,dp,abcd->nmop", U0, U0, U0, U0, psi_t).reshape(-1)
                assert rep0["orth_err"] < 1e-10 and abs(np.sum(Psi0[keep] ** 2) - 1) < 1e-9

                t1 = time.time()
                r = fit_one(model, x_sector, flat_idx, psi_tj, U0, mask, learn, args.steps,
                            args.lr, args.basis_lr, seed, checkpoints)
                full = np.zeros(16 ** 4, complex); full[keep] = r["T"]
                U = r["U"]
                spin_state = np.einsum("an,bm,co,dp,nmop->abcd", U, U, U, U,
                                       full.reshape(16, 16, 16, 16)).reshape(-1)
                de = energy_error(spin_state, H, E0)
                rep = basis_report(U, eig_v, psi_t)
                dt = time.time() - t1
                cb = r["curve_best"]
                print(f"  {label:15s} {seed:4d} | " + " ".join(f"{cb.get(c, float('nan')):7.1e}" for c in checkpoints)
                      + f" | {de:7.1e}  {rep0['eigen_likeness']:.2f}->{rep['eigen_likeness']:.2f}"
                      f"  {rep0['spin_likeness']:.2f}->{rep['spin_likeness']:.2f}"
                      f"  {rep0['wrong_sign']:.3f}->{rep['wrong_sign']:.3f}"
                      f"  {rep0['configs_99']}->{rep['configs_99']}"
                      + (f"  NaN@{r['nan_at']}" if r["nan_at"] else "") + f"  {dt:.0f}s", flush=True)
                rec = {"J2": J2, "fit": label, "init": init, "learned_basis": learn, "seed": seed,
                       "n_params_net": r["n_params"], "steps": args.steps, "lr": args.lr,
                       "basis_lr": args.basis_lr, "infidelity": float(r["infidelity"]),
                       "rel_energy_error": float(de), "nan_at_step": r["nan_at"],
                       "infidelity_curve_best": {str(k): float(v) for k, v in cb.items()},
                       "infidelity_curve_current": {str(k): float(v) for k, v in r["curve_now"].items()},
                       "basis_init": rep0, "basis_final": rep, "seconds": round(dt, 1)}
                results.append(rec)
                if out:
                    tag = f"J2_{J2}_{init}_{'learned' if learn else 'frozen'}_s{seed}"
                    np.save(os.path.join(out, f"U_{tag}.npy"), U)
                    with open(os.path.join(out, "results.jsonl"), "a") as f:
                        f.write(json.dumps(rec) + "\n"); f.flush(); os.fsync(f.fileno())

    if out:
        lines = []
        for J2 in sorted({r["J2"] for r in results}):
            lines.append(f"J2 = {J2}")
            lines.append(f"  {'fit':15s} seeds  {'infidelity mean (min-max)':27s} dE/E0 mean  "
                         f"eig_like spin_like wrong_sign cfg99   (final basis, mean)")
            groups = {}
            for r in results:
                if r["J2"] == J2:
                    groups.setdefault(r["fit"], []).append(r)
            for name, rs in sorted(groups.items(), key=lambda kv: np.mean([r["infidelity"] for r in kv[1]])):
                inf = [r["infidelity"] for r in rs]
                fb = lambda k: np.mean([r["basis_final"][k] for r in rs])
                lines.append(f"  {name:15s} {len(rs):5d}  {np.mean(inf):8.2e} ({min(inf):.1e}-{max(inf):.1e})"
                             f"   {np.mean([r['rel_energy_error'] for r in rs]):8.2e}   "
                             f"{fb('eigen_likeness'):.2f}     {fb('spin_likeness'):.2f}      "
                             f"{fb('wrong_sign'):.3f}    {fb('configs_99'):.0f}"
                             + ("  [NaN in some seed]" if any(r["nan_at_step"] for r in rs) else ""))
            lines.append("")
        open(os.path.join(out, "summary.txt"), "w").write("\n".join(lines))
        print("\n" + "\n".join(lines) + f"\nresults -> {out}", flush=True)


if __name__ == "__main__":
    main()
