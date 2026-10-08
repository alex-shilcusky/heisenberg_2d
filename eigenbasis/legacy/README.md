# Legacy scripts (not maintained)

Kept for reference from the old `2d_eigenbasis` repo; they predate the move
into `heisenberg_2d` and still use that repo's paths. Do not use them:

- `run_2D_eigenbasis.sh`: the original standalone Slurm wrapper, replaced by
  the shared `run.sh` + `submit.sh` at the repo root.
- `debug.sh`: an old copy of `submit.sh` that wrote into `debugging/`. The same
  is now done with the shared script and a run-root override:

  ```bash
  RUN_ROOT=eigenbasis/debugging ./submit.sh config_eigenbasis.toml
  ```
