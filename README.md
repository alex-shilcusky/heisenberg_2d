# Heisenberg J1-J2 (2D)

Variational Monte Carlo for the 2D J1-J2 Heisenberg model on the periodic
square lattice, with a Vision Transformer wave function, in two bases:

- **spin basis** (`spinbasis/`): plain sigma^z configurations, the NetKet
  tutorial ViT (plus Jastrow and RBM for comparison);
- **plaquette eigenbasis** (`eigenbasis/`): the lattice tiled with 2x2
  plaquettes, each replaced by its 16 local eigenstates, and a ViT over the
  cluster grid.

Both use the same driver (overdispersed `VMC_NG` with optional adaptive
alpha), the same Slurm wrappers, checkpoint/resubmit sentinels and output
formats. Energies are in S = sigma/2 units in both.

This repo replaces `2d_eigenbasis` and `spinbasis_2d`, which are kept
unchanged (on GitHub too) as the record of everything run before 2026-10-08.

## Layout

```
heisenberg_j1j2/
├── submit.sh                 login node: validate, expand sweep, sbatch
├── run.sh                    per-task Slurm script, dispatches on the basis (+ auto-resubmit)
├── config_spinbasis.toml     spin-basis base config + sweeps
├── config_eigenbasis.toml    eigenbasis base config + sweeps
├── submit_branch.sh          continue a finished run into a new folder
├── branch_run.py             (the work behind submit_branch.sh)
├── submit_diag.sh            4x4 exact-state diagnostics on a GPU
├── clusters/                 per-cluster resource profiles (aicr, aicr-batch, explorer)
├── spinbasis/                run_2D_spinbasis.py + helpers/   (see spinbasis/README.md)
├── eigenbasis/               run_2D_eigenbasis.py + helpers/  (see eigenbasis/README.md)
├── diagnostics/              sign_diagnostics_4x4.py, learned_basis_4x4.py
├── notebooks/                analysis and plotting notebooks
├── tools/import_data.py      copy data from the old folders (see Migration)
├── data/                     ALL run output (gitignored)
│   ├── spinbasis/<sweep>/
│   ├── eigenbasis/<sweep>/
│   └── diagnostics/<sign_4x4|learned_basis_4x4>/<run>/
├── run_hist/                 Slurm logs written before run.sh redirects (gitignored)
└── docs/                     reference papers (gitignored)
```

One sibling is required: `master/importance_sampling_nqs/`, an editable path
dependency (see `pyproject.toml`).

## Setup (once per machine)

```bash
cd ~/master/heisenberg_j1j2
uv sync --python 3.13          # 3.14 has no wheels for this lockfile
```

## Running

```bash
CLUSTER=aicr-batch ./submit.sh config_eigenbasis.toml
CLUSTER=aicr-batch ./submit.sh config_spinbasis.toml
```

`[system] basis` in the config (`"spin"` or `"eigen"`) decides everything
else: which entry point runs (`<basis>basis/run_2D_<basis>basis.py`), the
Slurm job name (`SB2D` / `EB2D`) and where output goes
(`data/<basis>basis/<timestamp>G<commit>P<pid>/run_XXXX/`). It cannot be swept;
use the other config. Override the output folder with `RUN_ROOT=<dir>`.

`submit.sh` refuses to run with a dirty git tree, and records the commit in
every sweep folder. A refused or cancelled submission leaves no folder behind.
`run.sh` auto-resubmits on a clean wall-clock stop, continuing from the
checkpoint, until `[model] iter` is reached.

**Sweeps.** `[sweep.<section>]` tables expand as a Cartesian product,
`[sweep_zip.<section>]` as zipped lists. A swept key must exist in the base
config. Examples are at the bottom of each config.

**Model tables.** The spin config keeps only the `[model.<type>]` table it
uses; the other wave functions are commented templates in the file. Sweeping
`model.type` needs each swept type's table present, but `submit.sh` drops the
unselected tables from every run, so `config.json`, `manifest.json` and
`matrix.json` only ever show the model that ran.

### Monitoring

```bash
d=data/eigenbasis/<sweep-id>/run_0000
tail -c 300 $d/stderr.txt     # tqdm: s/it
tail -f    $d/stdout.txt      # energy + sampling diagnostics
```

Job state is **not** a reliable success signal: `run.sh` always exits 0. Use
the `status` field in `manifest.json` (`completed` / `timeout` / `failed` /
`frozen` / `crashed_before_setup`). Every `manifest.json`, `config.json` and
`matrix.json` also records `basis`.

### Continuing a run

```bash
./submit_branch.sh --from data/eigenbasis/<sweep-id> --run 0 --extra 5000
```

Copies the parent's checkpoint into a new one-run folder next to the parent
and submits it; the parent stays intact. The basis comes from the parent's
config. `--set key=value` overrides config values. Options that change the
parameter tree need `--fresh`. Full options are at the top of `submit_branch.sh`.

### Diagnostics

```bash
CLUSTER=aicr-batch ./submit_diag.sh
DIAG=learned_basis CLUSTER=aicr-batch ./submit_diag.sh
```

Output: `data/diagnostics/<sign_4x4|learned_basis_4x4>/<timestamp>G<commit>/`;
read `summary.txt`.

## Clusters

Everything that differs between machines lives in `clusters/<name>.sh`. Pick
the profile per submission with `CLUSTER=...`; without it, the cluster is
autodetected (on AICR that means `aicr`, the 4 h devel partition).

| | `explorer` | `aicr` | `aicr-batch` |
|---|---|---|---|
| partition | `gpu` | `b200-devel` (4 h, max 4 concurrent) | `b200-batch` (24 h) |
| walltime | `04:00:00` | `04:00:00` | `24:00:00` |
| `MAX_RUNTIME_MIN` | 230 | 230 | 1380 |
| GPU request | `--gres=gpu:h200:1` | `--gpus=1` | as `aicr` |
| account | (none) | `p2026_0109_neu` (**required**) | as `aicr` |
| memory | `16GB` | `64G` | as `aicr` |
| cores | `4` | `16` | as `aicr` |
| module | `python/3.13.5` | `miniforge3` | as `aicr` |

`MAX_RUNTIME_MIN` in the config is what stops a run gracefully. It must sit
below the profile's walltime: `submit.sh` refuses if it does not, and warns
if it is under half. Both configs are currently set to 1380, so they need
`CLUSTER=aicr-batch`. For devel, set 230.

The chosen profile is written to `<sweep>/cluster.txt`, and `run.sh` reloads it
on every resubmission, so a sweep never switches partition mid-run.

To add a cluster: copy a profile, set `CLUSTER_NAME` to the filename, and add
the name to `CLUSTER_PROFILES` in `clusters/common.sh`.

Validating a GPU (on a compute node, never the login node):

```bash
srun --partition=b200-devel --account=p2026_0109_neu --gpus=1 --cpus-per-task=8 --mem=32G --time=00:10:00 .venv/bin/python -c "import jax; print(jax.devices())"
```

## Notebooks

| notebook | reads |
|---|---|
| `EIGENBASIS_VS_SPINBASIS.ipynb` | both bases + `data/diagnostics/sign_4x4`; the advisor summary |
| `ANALYSIS.ipynb`, `plot_energy_vs_iteration.ipynb`, `plot_depth_vs_width.ipynb` | `data/eigenbasis` |
| `plot_ansatz_comparison.ipynb` | `data/spinbasis` |

Each one finds `data/` by walking up from the notebook, so they run unchanged
on the Mac and on the cluster.

## Migration from the old folders

The data was copied, not moved, from `master/2D_eigenbasis_data`,
`master/spinbasis_data_2d` and `master/2D_diagnostics`; those folders were not
touched. The copy adds `basis` to every `matrix.json`, `config.json`,
`manifest.json` and `*.summary.json`. Each sweep's archived `config.toml` is
left verbatim, as the record of what was submitted. Older spin-basis configs
therefore still show all three model tables.

```bash
.venv/bin/python tools/import_data.py --from ../2D_eigenbasis_data --basis eigen
.venv/bin/python tools/import_data.py --from ../spinbasis_data_2d --basis spin
.venv/bin/python tools/import_data.py --from ../2D_diagnostics --basis diagnostics
```

It is safe to re-run: sweeps already in `data/` are skipped. A sweep that is
still being written (an unfinished run touched in the last 12 h) is skipped
too. Import it once it finishes:

```bash
.venv/bin/python tools/import_data.py --from ../spinbasis_data_2d --basis spin --sweeps <sweep-id> --include-running
```

Naming a sweep with `--sweeps` re-copies it over the existing one. Add `--dry-run` to
see what would happen.
