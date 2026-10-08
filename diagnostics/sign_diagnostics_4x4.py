"""
sign_diagnostics_4x4.py
=======================
Is the eigenbasis ViT limited by the SIGN STRUCTURE of the ground state in the
plaquette eigenbasis, or by sampling / optimization? Settled on the 4x4 lattice,
where the exact ground state is available, by taking VMC out of the picture.

The same ViT architecture is fit DIRECTLY to the exact ground state (maximum
fidelity over the full Sz = 0 sector, 12,870 configurations, no sampling):

  spin       spin-basis ViT (spinbasis/helpers/vit.py) -> psi in the spin basis
  eigen      eigenbasis ViT (helpers/multilayer_2D_vit_Eigenbasis.py) -> Psi in
             the plaquette eigenbasis
  eigen+sign the same eigenbasis ViT learning AMPLITUDES only: the exact sign of
             Psi is supplied ("sign oracle")

Both bases describe the same state: Psi is psi with every plaquette rotated
into its eigenbasis, so the two targets are the same vector written in two
local bases. If eigen+sign ~ spin while eigen >> spin, the sign is what the
network cannot learn. If eigen ~ spin, the basis is not the problem on 4x4.

Also printed: how much of the ground state's weight has a sign no simple rule
predicts (best of all 2^16 rules in each basis), and the plaquette occupations.

Usage:
    On the cluster (GPU, results organised for you):   ./submit_diag.sh
    Locally (CPU, slow -- ~0.5 s per step):
        PYTHONPATH=../importance_sampling_nqs/packages .venv/bin/python \
            sign_diagnostics_4x4.py --j2 0.0 --seeds 0 --steps 1500 --out /some/dir

Fit specs (--fits): spin | eigen | eigen+sign, and for the eigenbasis fits an
optional network variant ':<attn_mode>:<output_head>' (e.g. eigen:s:amp_phase).

Read-only with respect to runs. With --out it writes, into that directory:
meta.json (args, devices, git commit), sign_diagnostics.json, results.jsonl
(one line per finished fit, appended as it goes), and summary.txt.
"""
import argparse
import ast
import importlib.util
import itertools
import json
import os
import platform
import sys
import time
from functools import reduce

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import optax

HERE = os.path.dirname(os.path.abspath(__file__))              # diagnostics/
REPO = os.path.dirname(HERE)                                     # heisenberg_j1j2/
EIGEN_DIR = os.path.join(REPO, "eigenbasis")
SPIN_DIR = os.path.join(REPO, "spinbasis")
L, C, NS = 4, 2, 16                   # 4x4 lattice, 2x2 grid of 2x2 plaquettes


# ---------------------------------------------------------------------------
# Reuse the production basis construction without running the entry point.
# ---------------------------------------------------------------------------
def _load_basis_functions():
    """Exec only the named top-level functions of run_2D_eigenbasis.py (found
    by name with `ast`, so line edits there can't break this)."""
    path = os.path.join(EIGEN_DIR, "run_2D_eigenbasis.py")
    tree = ast.parse(open(path).read(), path)
    want = {"_perm_op_2x2", "_refine_blocks", "get_ops_2x2", "get_piecewise_Hamiltonian_2D"}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in want]
    missing = want - {n.name for n in nodes}
    if missing:
        raise RuntimeError(f"run_2D_eigenbasis.py no longer defines {sorted(missing)}")
    ns = {"np": np, "reduce": reduce}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), path, "exec"), ns)
    return ns


def _load_module(name, path):
    """Import a file as a standalone module. Both projects have a package called
    `helpers`, so normal imports would collide."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Lattice and exact ground state
# ---------------------------------------------------------------------------
def site(X, Y):
    """Plaquette-major site label: cluster (cx + C*cy) * 4 + intra-cluster k,
    with k = dx + 2*dy -- the convention of get_ops_2x2 / get_cluster_graph_2D_J2."""
    cx, dx = divmod(X % L, 2)
    cy, dy = divmod(Y % L, 2)
    return (cx + C * cy) * 4 + (dx + 2 * dy)


def build_hamiltonian(J2):
    """Sparse 2^16 x 2^16 J1-J2 Heisenberg in S = sigma/2 units, no sign rule.
    Bit (NS-1-i) of a basis index is site i; bit 0 = spin up."""
    idx = np.arange(2 ** NS)
    up = [1 - ((idx >> (NS - 1 - i)) & 1) for i in range(NS)]
    families = [(1.0, ((1, 0), (0, 1))), (J2, ((1, 1), (1, -1)))]
    diag = np.zeros(2 ** NS)
    rows, cols, vals = [], [], []
    for J, dirs in families:
        if J == 0:
            continue
        bonds = {tuple(sorted((site(X, Y), site(X + dx, Y + dy))))
                 for X in range(L) for Y in range(L) for dx, dy in dirs}
        for i, j in bonds:
            diag += 0.25 * J * np.where(up[i] == up[j], 1, -1)
            anti = up[i] != up[j]
            rows.append(idx[anti] ^ ((1 << (NS - 1 - i)) | (1 << (NS - 1 - j))))
            cols.append(idx[anti])
            vals.append(np.full(anti.sum(), 0.5 * J))
    H = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(2 ** NS, 2 ** NS)) + sp.diags(diag)
    return H, np.stack(up)                                     # up: (NS, 2^NS)


# ---------------------------------------------------------------------------
# Sign diagnostics
# ---------------------------------------------------------------------------
_SIGNS16 = np.array(list(itertools.product((1, -1), repeat=16)), float)


def wrong_sign_weight_spin(psi, up):
    """Best of all 2^16 rules (-1)^{sum_{i in S} n_up_i} (includes Marshall):
    fraction of |psi|^2 whose sign the rule gets wrong."""
    w = np.sign(psi) * psi ** 2
    best = 0.0
    for ch in range(0, 65536, 2048):
        subset = (_SIGNS16[ch:ch + 2048] < 0).astype(float)
        rule = 1 - 2 * ((subset @ up) % 2)
        best = max(best, np.abs(rule @ w).max())
    return (1 - best) / 2


def wrong_sign_weight_eigen(Psi):
    """Best of all 2^16 per-eigenstate sign gauges s(n), applied as
    s(n1) s(n2) s(n3) s(n4) (the same rule on every plaquette)."""
    w = np.sign(Psi) * Psi ** 2
    best = 0.0
    for ch in range(0, 65536, 4096):
        g = _SIGNS16[ch:ch + 4096]
        x = np.einsum("ga,abcd->gbcd", g, w)
        x = np.einsum("gb,gbcd->gcd", g, x)
        x = np.einsum("gc,gcd->gd", g, x)
        best = max(best, np.abs(np.einsum("gd,gd->g", g, x)).max())
    return (1 - best) / 2


# ---------------------------------------------------------------------------
# Fidelity fitting
# ---------------------------------------------------------------------------
def make_loss(apply_fn, x, target, sign_oracle):
    """-log fidelity between exp(model(x)) and the real target vector, exact
    over the sector. With sign_oracle the model supplies only log|psi|."""
    t = jnp.asarray(target)
    t_norm2 = jnp.sum(t ** 2)
    sgn = jnp.where(t >= 0, 1.0, -1.0)                       # exact zeros -> +1

    def psi_of(params):
        f = apply_fn(params, x)
        if sign_oracle:
            a = jnp.real(f)
            return jnp.exp(a - jax.lax.stop_gradient(a.max())) * sgn
        return jnp.exp(f - jax.lax.stop_gradient(jnp.real(f).max()))

    def loss(params):
        p = psi_of(params)
        ov = jnp.sum(t * p)
        fid = jnp.abs(ov) ** 2 / (jnp.sum(jnp.abs(p) ** 2) * t_norm2)
        return -jnp.log(fid)

    return loss, psi_of


def fit(model, x, target, sign_oracle, steps, lr, seed, checkpoints):
    params = model.init(jax.random.PRNGKey(seed), x[:2])
    n_params = sum(int(np.prod(a.shape)) for a in jax.tree_util.tree_leaves(params))
    loss, psi_of = make_loss(model.apply, x, target, sign_oracle)
    opt = optax.adam(lr)
    state = opt.init(params)

    @jax.jit
    def step(params, state):
        val, g = jax.value_and_grad(loss)(params)
        upd, state = opt.update(g, state, params)
        return optax.apply_updates(params, upd), state, val

    curve, best, best_params = {}, np.inf, params
    for i in range(1, steps + 1):
        params, state, val = step(params, state)
        v = float(val)
        if v < best:
            best, best_params = v, params
        if i in checkpoints:
            curve[i] = 1 - np.exp(-best)
    return n_params, curve, 1 - np.exp(-best), np.asarray(psi_of(best_params))


def energy_error(psi_full, H, E0):
    """Relative energy error of a (complex) spin-basis vector."""
    Hp = H @ psi_full
    E = np.real(np.vdot(psi_full, Hp)) / np.real(np.vdot(psi_full, psi_full))
    return abs(E - E0) / abs(E0)


# ---------------------------------------------------------------------------
DEFAULT_FITS = ["spin", "eigen", "eigen+sign"]


def parse_fit_spec(spec):
    """'spin' | 'eigen' | 'eigen+sign', optionally ':<attn_mode>:<output_head>'
    for the eigenbasis fits, e.g. 'eigen:s:amp_phase'. Returns
    (kind, attn_mode, output_head, label)."""
    parts = spec.split(":")
    kind = parts[0]
    if kind not in ("spin", "eigen", "eigen+sign"):
        raise ValueError(f"unknown fit {spec!r}")
    if kind == "spin":
        if len(parts) > 1:
            raise ValueError(f"{spec!r}: the spin fit takes no variant")
        return kind, None, None, "spin"
    attn = parts[1] if len(parts) > 1 else "factored"
    head = parts[2] if len(parts) > 2 else "logcosh"
    if attn not in ("factored", "s", "d") or head not in ("logcosh", "amp_phase") or len(parts) > 3:
        raise ValueError(f"{spec!r}: expected {kind}[:factored|s|d[:logcosh|amp_phase]]")
    label = kind if (attn, head) == ("factored", "logcosh") else \
        f"{kind}[{attn},{'amp' if head == 'amp_phase' else 'lc'}]"
    return kind, attn, head, label


def _git_info():
    import subprocess
    def run(*a):
        try:
            return subprocess.run(["git", *a], cwd=HERE, capture_output=True, text=True).stdout.strip()
        except Exception:
            return None
    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain"))}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--j2", type=float, nargs="+", default=[0.0, 0.5])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--d-model", type=int, default=72)
    ap.add_argument("--heads", type=int, default=12)
    ap.add_argument("--fits", nargs="+", default=DEFAULT_FITS,
                    help="fit specs: spin | eigen | eigen+sign, the eigen ones optionally "
                         "':<factored|s|d>:<logcosh|amp_phase>', e.g. eigen:s:amp_phase")
    ap.add_argument("--out", default=None,
                    help="directory for results.jsonl (one line per finished fit), "
                         "sign_diagnostics.json, meta.json and summary.txt")
    ap.add_argument("--require-gpu", action="store_true",
                    help="exit immediately if JAX does not see a GPU (so a broken CUDA "
                         "install fails in seconds instead of running for hours on CPU)")
    args = ap.parse_args()
    specs = [parse_fit_spec(s) for s in args.fits]            # validate before any work

    devices = [str(d) for d in jax.devices()]
    print("JAX devices:", devices, flush=True)
    if args.require_gpu and jax.default_backend() != "gpu":
        raise SystemExit(f"--require-gpu: JAX backend is {jax.default_backend()!r}, not 'gpu'. "
                         f"Check that the venv has jax[cuda] and the job got a GPU.")

    out = args.out
    if out:
        os.makedirs(out, exist_ok=True)
        meta = {"argv": sys.argv, "args": vars(args), "devices": devices, "backend": jax.default_backend(),
                "host": platform.node(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "jax": jax.__version__, "git": _git_info(),
                "started": time.strftime("%Y-%m-%d %H:%M:%S")}
        json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=2)
    results, sign_diag = [], {}

    basis = _load_basis_functions()
    spin_vit_path = os.path.join(SPIN_DIR, "helpers", "vit.py")
    if not os.path.exists(spin_vit_path):
        raise SystemExit(f"spin-basis ViT not found at {os.path.abspath(spin_vit_path)}; "
                         f"spinbasis/ must sit next to eigenbasis/ in this repo.")
    vit_spin = _load_module("vit_spin", spin_vit_path)
    vit_eig = _load_module("vit_eig", os.path.join(EIGEN_DIR, "helpers", "multilayer_2D_vit_Eigenbasis.py"))
    checkpoints = sorted({c for c in (100, 300, 1000, 2000, 5000, args.steps) if c <= args.steps})

    # spin configs in the Sz = 0 sector, as ViT inputs: row-major with
    # row = X, col = Y, so each 2x2 patch of extract_patches2d IS a plaquette
    pos = np.array([[X * L + Y for X in range(L) for Y in range(L)]]).ravel()
    lab = np.array([site(X, Y) for X in range(L) for Y in range(L)])
    order = np.empty(NS, int); order[pos] = lab               # row-major slot -> site label

    for J2 in args.j2:
        t0 = time.time()
        H, up = build_hamiltonian(J2)
        w, V = sla.eigsh(H, k=1, which="SA")
        E0 = float(w[0]); psi = V[:, 0]
        sector = np.flatnonzero(up.sum(axis=0) == NS // 2)
        assert len(sector) == 12870

        spins = (2 * up[:, sector] - 1).T                       # (12870, 16) in site-label order
        x_spin = jnp.asarray(spins[:, order], dtype=jnp.float64)
        t_spin = psi[sector]

        ops, qn, v = basis["get_ops_2x2"]((False, False), 1.0, J2, verbose=False,
                                          return_qn=True, return_basis=True)
        Psi = np.einsum("an,bm,co,dp,abcd->nmop", v, v, v, v, psi.reshape(16, 16, 16, 16),
                        optimize=True)
        sz = np.round(np.diag(ops[-1]), 9)
        grid = np.stack(np.meshgrid(*[np.arange(16)] * 4, indexing="ij"), -1).reshape(-1, 4)
        keep = np.abs(sz[grid].sum(axis=1)) < 1e-9
        x_eig = jnp.asarray(grid[keep], dtype=jnp.int32)
        t_eig = Psi.reshape(-1)[keep]
        assert len(t_eig) == 12870 and abs(np.sum(t_eig ** 2) - 1) < 1e-9

        def eig_to_spin(p_sector):
            full = np.zeros(16 ** 4, complex); full[keep] = p_sector
            return np.einsum("an,bm,co,dp,nmop->abcd", v, v, v, v, full.reshape(16, 16, 16, 16),
                             optimize=True).reshape(-1)

        def spin_to_full(p_sector):
            full = np.zeros(2 ** NS, complex); full[sector] = p_sector
            return full

        # sanity: the exact vectors give fidelity 1 and the exact energy
        assert energy_error(eig_to_spin(t_eig), H, E0) < 1e-10
        assert energy_error(spin_to_full(t_spin), H, E0) < 1e-10

        occ = (Psi ** 2).sum(axis=(1, 2, 3))
        P = np.sort(t_eig ** 2)[::-1]; Q = np.sort(t_spin ** 2)[::-1]
        diag = {"E0_per_site": E0 / NS,
                "wrong_sign_weight_spin": float(wrong_sign_weight_spin(psi, up)),
                "wrong_sign_weight_eigen": float(wrong_sign_weight_eigen(Psi)),
                "configs_99pct_spin": int(np.searchsorted(np.cumsum(Q), .99) + 1),
                "configs_99pct_eigen": int(np.searchsorted(np.cumsum(P), .99) + 1),
                "plaquette_occupation": [float(z) for z in occ]}
        sign_diag[str(J2)] = diag
        print(f"\n=== J2 = {J2}   E0/site = {E0 / NS:+.6f}   ({time.time() - t0:.0f}s setup)")
        print(f"  wrong-sign weight, best rule:  spin {diag['wrong_sign_weight_spin']:.3f}   "
              f"eigen {diag['wrong_sign_weight_eigen']:.3f}")
        print(f"  configs holding 99% weight:   spin {diag['configs_99pct_spin']}   "
              f"eigen {diag['configs_99pct_eigen']}")
        print(f"  plaquette occupation: GS {occ[0]:.3f}, next three {occ[1]:.3f} {occ[2]:.3f} {occ[3]:.3f}")
        if out:
            json.dump(sign_diag, open(os.path.join(out, "sign_diagnostics.json"), "w"), indent=2)

        qn_static = tuple(tuple(float(z) for z in row) for row in qn)
        print(f"  {'fit':19s} {'seed':>4s} {'params':>7s} | infidelity at step "
              + " ".join(f"{c:>8d}" for c in checkpoints) + " | dE/E0     time", flush=True)
        for kind, attn, head, label in specs:
            if kind == "spin":
                model = vit_spin.ViT(num_layers=args.layers, d_model=args.d_model, n_heads=args.heads,
                                     patch_size=2, transl_invariant=True)
                x, target, oracle, to_spin = x_spin, t_spin, False, spin_to_full
            else:
                model = vit_eig.ViT_2D_EB(num_layers=args.layers, d_model=args.d_model, n_heads=args.heads,
                                          local_dim=16, attn_mode=attn, transl_invariant=True,
                                          embedding="qn", qn_table=qn_static, output_head=head)
                x, target, oracle, to_spin = x_eig, t_eig, kind == "eigen+sign", eig_to_spin
            for seed in args.seeds:
                t1 = time.time()
                n_par, curve, inf, p = fit(model, x, target, oracle, args.steps, args.lr, seed, checkpoints)
                de = energy_error(to_spin(p), H, E0)
                dt = time.time() - t1
                print(f"  {label:19s} {seed:4d} {n_par:7d} | " + " ".join(f"{curve[c]:8.1e}" for c in checkpoints)
                      + f" | {de:8.1e}  {dt:4.0f}s", flush=True)
                rec = {"J2": J2, "fit": label, "kind": kind, "attn_mode": attn, "output_head": head,
                       "seed": seed, "n_params": n_par, "steps": args.steps, "lr": args.lr,
                       "infidelity": float(inf), "rel_energy_error": float(de),
                       "infidelity_curve": {str(k): float(val) for k, val in curve.items()},
                       "seconds": round(dt, 1)}
                results.append(rec)
                if out:                                    # one line per finished fit, survives a timeout
                    with open(os.path.join(out, "results.jsonl"), "a") as f:
                        f.write(json.dumps(rec) + "\n"); f.flush(); os.fsync(f.fileno())

    if out:
        write_summary(results, sign_diag, os.path.join(out, "summary.txt"))
        print(f"\nresults -> {out}", flush=True)


def write_summary(results, sign_diag, path):
    """Seed-averaged table, one row per (J2, fit), ranked within each J2."""
    lines = []
    for J2 in sorted({r["J2"] for r in results}):
        d = sign_diag.get(str(J2), {})
        lines.append(f"J2 = {J2}   E0/site = {d.get('E0_per_site', float('nan')):+.6f}   "
                     f"wrong-sign weight: spin {d.get('wrong_sign_weight_spin', float('nan')):.3f}  "
                     f"eigen {d.get('wrong_sign_weight_eigen', float('nan')):.3f}")
        rows = {}
        for r in results:
            if r["J2"] == J2:
                rows.setdefault(r["fit"], []).append(r)
        lines.append(f"  {'fit':19s} {'params':>7s} {'seeds':>5s}  {'infidelity (mean, min-max)':28s}  dE/E0 (mean)")
        for fit_name, rs in sorted(rows.items(), key=lambda kv: np.mean([r["infidelity"] for r in kv[1]])):
            inf = [r["infidelity"] for r in rs]; de = [r["rel_energy_error"] for r in rs]
            lines.append(f"  {fit_name:19s} {rs[0]['n_params']:7d} {len(rs):5d}  "
                         f"{np.mean(inf):8.2e} ({min(inf):.1e}-{max(inf):.1e})      {np.mean(de):8.2e}")
        lines.append("")
    open(path, "w").write("\n".join(lines))
    print("\n" + "\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
