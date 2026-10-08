# 2D Eigenbasis

Variational Monte Carlo for the 2D J1-J2 Heisenberg model on the square
lattice, using a **cluster eigenbasis** representation and a Vision Transformer
wave function.

Instead of working in the computational spin basis, the lattice is tiled with
2x2 plaquettes. Each plaquette's 4-site Hamiltonian is diagonalised exactly and
its 16 eigenstates become the local degrees of freedom, so a configuration is a
length-`(L/2)^2` vector of integers in `[0, 16)` rather than `L^2` spins. The
inter-cluster terms are rotated into that basis and the whole thing is handed to
NetKet as a custom Hilbert space (`EigenHilbert`) plus a graph operator.

The ansatz is a ViT over the cluster grid (one token per plaquette), optimised
by stochastic reconfiguration with overdispersed importance sampling.

- **Stack**: Python 3.13, JAX, Flax, NetKet 3.17+, `advanced_drivers`
- **Reference**: Viteritti, Rende, Parola, Goldt, Becca, *Transformer Wave
  Function for two dimensional frustrated magnets* (2025)

## Layout

```
eigenbasis/
├── run_2D_eigenbasis.py      entry point: builds H, model, driver; runs VMC
├── hami_svd.jl               SVD-basis exploration (Julia, standalone)
├── helpers/
│   ├── EigenHilbert.py                  custom Hilbert space + Sz constraint
│   ├── hamiltonians_graphs.py           directed graph + cluster operator
│   ├── multilayer_2D_vit_Eigenbasis.py  the ViT (factored attention)
│   ├── symmetries.py                    C4v / spin-flip projection
│   ├── vmc_is_firstorder.py             IS-VMC without the SR preconditioner
│   └── ...                              older variants, unused by the entry point
├── debugging/                local debug runs (gitignored)
└── legacy/                   pre-consolidation scripts, not maintained
```

Configuration: `../config_eigenbasis.toml` (`[system] basis = "eigen"`).
Submitting, clusters, monitoring and branching are shared with the spin
basis and described in the top-level `README.md`:

```bash
CLUSTER=aicr-batch ./submit.sh config_eigenbasis.toml
```

Output: `data/eigenbasis/<sweep-id>/run_XXXX/`.

## Configuration notes

| Key | Note |
|---|---|
| `[sampler] N_SAMPLES` | MC samples per step, stated directly (8192). Was `MONTECARLO = 4`, a multiplier against a hidden 2048; configs using the old key still replay. Keep it a multiple of `chains`. |
| `[model] layers` | Depth dominates accuracy; the reference uses 8 |
| `[model] ratio`, `heads` | `d_model = ratio * heads`; keep `d_head = ratio >= 6` |
| `[model] embedding` | `learnable` (free 16xd table) or `qn` (linear map of E, Sz, S~, sigma_d, C2) |
| `[distribution] alpha` | Sample `|psi|^alpha`; `auto_alpha` tunes it by gradient SNR |
| `[symmetry] elements` | C4v projection; costs `|G|` forward passes per sample |
| `[run] MAX_RUNTIME_MIN` | **Must stay below the profile's `CLUSTER_WALLTIME`** or Slurm hard-kills before the checkpoint is written. `submit.sh` enforces this and refuses to submit. |

## Status

Best 6x6 J2=0 result so far is `E/site = -0.6748` against the exact
`-0.678872`, i.e. **rel. err ~6e-3**; the reference achieves ~1e-4 on
comparable systems. Measured throughput: **11.56 s/step** at 8 layers / d=72
(H200 and B200 are within noise of each other — the model is small and
`float64`, so it is launch-bound, not FLOP-bound).

Open items, roughly in order of expected payoff:

1. **Depth.** `layers` is currently 2; 8 is where the reference gets its accuracy.
2. **Importance sampling.** ESS decays to ~7%, giving `M_eff ~ 550` against
   433k parameters. The reference reaches 1e-4 with plain Born sampling at
   M = 6000. A control run at `alpha = 2.0, auto_alpha = false` would settle
   whether the IS is helping or capping accuracy here.
3. **Symmetry.** See below — implemented, off by default on cost grounds.
4. **Odd translations are unrecoverable.** The 2x2 tiling keeps only
   `(L/2)^2` of the `L^2` translations. Unlike Shastry-Sutherland, where the
   4-site patch *is* the crystallographic unit cell, the square lattice has a
   one-site cell, so patching here discards 75% of the translation group and
   projection cannot restore it (odd translations re-tile the lattice and do
   not map cluster-eigenbasis configurations to one another).

---

## Recent changes

### Symmetry projection (new)

`helpers/symmetries.py` adds projection over the plaquette point group C4v and
optionally the global spin flip, implementing Eq. (A1) of the reference. Cluster
*translations* were already exact via the circulant attention (`roll2d`); this
covers the point group, which was missing entirely.

The key enabling fact, verified numerically: in the eigenbasis produced by
`get_ops_2x2`, **every C4v element is a signed permutation of the 16 local
states** — `sigma_d` and `C2` are diagonal (the basis is adapted to them),
and `C4` merely exchanges the two partners of each 2D E-irrep. So a symmetry
operation is an index gather plus a sign, not a matrix product.
`build_symmetry_tables` asserts this and raises if it ever stops holding.

`SymmProject` wraps any log-amplitude module, mirroring the `FlipExpSum`
pattern in `deepnets`. Each `-1` is carried as `+i*pi` in the complex
log-amplitude, then combined by a stable complex log-mean-exp. Validated over
thousands of random configuration/element pairs: `|psi|` invariant to ~1e-15,
amplitude ratio exactly `±1`, full orbit size (8 for C4v, 16 with spin flip),
for both `sign_rule = FF` and `TF`.

Off by default. It costs `|G|` forward passes per sample, so the staged
protocol of Appendix D — optimise unsymmetrised, enable symmetries for a short
final phase — is the affordable way to use it. Note that wrapping the model
changes the flax parameter tree, so `.mpack` checkpoints do **not** transfer
between symmetrised and unsymmetrised runs.

Supporting change: `get_ops_2x2(..., return_basis=True)` now also returns the
local eigenbasis `v`, which the table builder needs.

### Optimizer schedule

`final_learning_rate` `1e-3 -> 2e-2`. With `lr_transition_steps = 400`, the old
value left 96% of the run at roughly 1/20 of the reference step size
(`tau = 0.02-0.03`). The symptom was a 8-layer run descending at `-8.6e-6`
E/step after step ~400 with `Var/site` flat at ~0.006 for 3400 steps —
under-optimised rather than at a variational floor.

### Cluster profiles (new)

Cluster-specific resources moved out of `run.sh`'s `#SBATCH` header into
`clusters/aicr.sh` and `clusters/explorer.sh`, so switching machines is
`CLUSTER=explorer ./submit.sh config_eigenbasis.toml` instead of hand-editing the header.
`run.sh` passes the same flags when it resubmits itself, and `debug.sh` —
which was still loading Explorer's `python/3.13.5` on AICR — now takes its
module from the profile.

`submit.sh` additionally enforces `MAX_RUNTIME_MIN < CLUSTER_WALLTIME`, a
coupling that was previously only a comment in two files.

### AICR migration

`run.sh` / `submit.sh` retargeted from Discovery:

| | Discovery | AICR |
|---|---|---|
| partition | `gpu` | `b200-devel` |
| GPU request | `--gres=gpu:h200:1` | `--gpus=1` |
| account | (none) | `--account=p2026_0109_neu` (**required**) |
| memory | `16GB` | `64G` (AICR defaults to 1 GB/CPU) |
| cores | `4` | `16` (per-GPU share of a 128-core node) |
| module | `python/3.13.5` | `miniforge3` |

`MAX_RUNTIME_MIN` now documents its coupling to `--time` in both files.

### Repo hygiene

`.gitignore` now covers `.DS_Store`, `*.tar.gz`, and `*.pdf` (reference papers
are kept locally; `git add -f` to track one deliberately). `submit.sh` treats
untracked files as a dirty tree, so these previously blocked submission.
