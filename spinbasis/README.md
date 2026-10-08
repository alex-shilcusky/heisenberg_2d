# 2D Spin Basis

Variational Monte Carlo for the 2D J1-J2 Heisenberg model on the periodic
square lattice, in the plain sigma^z spin basis, with a choice of wavefunction:
Jastrow, RBM (optionally translation- or space-group-symmetric), or the Vision
Transformer from the NetKet tutorial.

Shares the driver, Slurm wrappers, checkpoint/resubmit sentinels and output
formats with `../eigenbasis`. Energies are in S = sigma/2 units in both.

## Layout

```
spinbasis/
├── run_2D_spinbasis.py    entry point: builds H, model, driver; runs VMC
└── helpers/
    ├── wavefunctions.py   [model] config -> Flax module (jastrow | rbm | vit)
    └── vit.py             the tutorial's spin-basis ViT
```

Configuration: `../config_spinbasis.toml` (`[system] basis = "spin"`).
Submitting, clusters, monitoring and branching are shared with the
eigenbasis and described in the top-level `README.md`:

```bash
CLUSTER=aicr-batch ./submit.sh config_spinbasis.toml
```

Output: `data/spinbasis/<sweep-id>/run_XXXX/`.

## Choosing a wavefunction

Set `[model] type` and give it a `[model.<type>]` table. The config keeps only
the table in use; the Jastrow and RBM tables are commented templates below it.
To sweep `model.type`, uncomment the tables for every swept type: `submit.sh`
still writes only the selected table into each run's config and manifest.

Adding a wavefunction is one branch in `helpers/wavefunctions.py:build_model`
plus a `[model.<name>]` table.

## Differences from the eigenbasis code worth knowing

- **Checkpoints**: `run.mpack` and `run.progress` are written together, atomically,
  every `SAVE_EVERY` steps. A hard kill or `scancel` loses at most `SAVE_EVERY`
  steps, and resubmitting the array task resumes correctly with no manual repair.
- **Divergence** (`InvalidLossStopping`) never overwrites the checkpoint.
- **Log files** are `run step 000000.log`, `run step 004400.log`, ...: zero-padded,
  so sorting by name sorts by time.
