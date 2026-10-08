#!/bin/bash
# ==========================================================================
# submit_diag.sh -- run the 4x4 sign diagnostics on a cluster GPU
#
# Fits the same ViT to the exact 4x4 ground state in the spin basis, in the
# plaquette eigenbasis, in the eigenbasis with the exact signs supplied, and
# with several eigenbasis network variants (see sign_diagnostics_4x4.py).
#
# USAGE (login node, from the heisenberg_j1j2 checkout):
#
#     ./submit_diag.sh                       # sign diagnostics (default)
#     DIAG=learned_basis ./submit_diag.sh    # learned local basis (learned_basis_4x4.py)
#     CLUSTER=aicr-batch ./submit_diag.sh    # pick a profile (24 h partition)
#     STEPS=3000 SEEDS="0 1 2" ./submit_diag.sh
#     DIAG=learned_basis BASIS_LR=1e-2 ./submit_diag.sh
#
# Like submit.sh, it requires a clean git tree, so every result is tied to a
# commit.
#
# EVERYTHING lands in ONE folder, printed at submission:
#
#     heisenberg_j1j2/data/diagnostics/<sign_4x4|learned_basis_4x4>/<YYYY-MM-DD_HH-MM-SS>G<commit>/
#         STATUS                 queued -> running -> done | failed (exit N)
#         submit_info.txt        commit, cluster, resources, the exact command
#         slurm-<jobid>.out/.err full job log (the tables appear here live)
#         summary.txt            final seed-averaged table, ranked  <- read this
#         results.jsonl          one line per finished fit (survives a timeout)
#         sign_diagnostics.json  sign / concentration numbers per J2
#         meta.json              args, GPU, JAX version, git commit
#         U_*.npy                (learned_basis only) the final 16x16 local basis per fit
#
# It is a sibling of data/spinbasis and data/eigenbasis rather than inside
# them: the notebooks treat every folder there as a VMC sweep.
# ==========================================================================
set -euo pipefail

# --------------------------------------------------------------------------
# JOB MODE -- Slurm runs this same file with --job <out_dir> <python args...>
# --------------------------------------------------------------------------
if [[ "${1:-}" == "--job" ]]; then
    OUT="$2"; SCRIPT="$3"; shift 3
    set +e
    REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
    cd "$REPO_ROOT" || { echo "failed (cannot cd to $REPO_ROOT)" > "$OUT/STATUS"; exit 1; }
    echo "running (job ${SLURM_JOB_ID:-local} on ${SLURMD_NODENAME:-$(hostname)})" > "$OUT/STATUS"

    # Same interpreter handling as run.sh: purge modules, call the venv's
    # python by absolute path, fall back to the profile's python module only if
    # the venv's base interpreter vanished with the purge.
    if command -v module >/dev/null 2>&1; then module purge; fi
    VENV_PY="$REPO_ROOT/.venv/bin/python"
    if ! "$VENV_PY" -c 'pass' >/dev/null 2>&1; then
        PYMOD="$(cat "$OUT/python_module.txt" 2>/dev/null || true)"
        if [[ -n "$PYMOD" ]] && command -v module >/dev/null 2>&1; then module load "$PYMOD"; fi
    fi
    if ! "$VENV_PY" -c 'import sys; sys.exit(0 if sys.version_info[:2] >= (3, 10) else 1)' 2>/dev/null; then
        echo "ERROR: venv interpreter missing or broken: $VENV_PY" >&2
        echo "       Recreate it:  cd $REPO_ROOT && rm -rf .venv && uv sync" >&2
        echo "failed (no usable venv python)" > "$OUT/STATUS"; exit 1
    fi

    export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
    export XLA_PYTHON_CLIENT_PREALLOCATE=false     # the model is tiny; don't grab the whole card
    export JAX_TRACEBACK_FILTERING=off

    echo "Repo:    $REPO_ROOT"
    echo "Python:  $("$VENV_PY" -V 2>&1)  ($VENV_PY)"
    echo "Output:  $OUT"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "(nvidia-smi unavailable)"
    echo "Command: $VENV_PY -u $SCRIPT --out $OUT $*"
    echo

    "$VENV_PY" -u "$SCRIPT" --out "$OUT" "$@"
    CODE=$?
    if [[ $CODE -eq 0 ]]; then echo "done" > "$OUT/STATUS"; else echo "failed (exit $CODE)" > "$OUT/STATUS"; fi
    exit $CODE
fi

# --------------------------------------------------------------------------
# SUBMIT MODE -- login node
# --------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "ERROR: $SCRIPT_DIR is not inside a git repository." >&2; exit 1
fi
REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

if [[ -n "$(git status --porcelain)" ]]; then
    echo "ERROR: Git working directory is dirty:" >&2
    git status --short >&2
    echo "Commit or stash your changes before submitting, so the results are tied to a commit." >&2
    exit 1
fi
command -v sbatch >/dev/null 2>&1 || { echo "ERROR: sbatch not found. Are you on the login node?" >&2; exit 1; }
[[ -x "$REPO_ROOT/.venv/bin/python" ]] || {
    echo "ERROR: $REPO_ROOT/.venv is missing. Run:  uv sync" >&2; exit 1; }
[[ -f "$REPO_ROOT/spinbasis/helpers/vit.py" ]] || {
    echo "ERROR: spinbasis/helpers/vit.py not found in $REPO_ROOT." >&2; exit 1; }

source "$REPO_ROOT/clusters/common.sh"
cluster_load
cluster_summary

# Which diagnostic. Fits run in the listed order, most important first, so a
# timeout still leaves the key rows (results.jsonl is written per fit).
DIAG="${DIAG:-sign}"
case "$DIAG" in
    sign)
        SCRIPT="diagnostics/sign_diagnostics_4x4.py"; SUBDIR="sign_4x4"
        DEFAULT_FITS="spin eigen eigen+sign eigen:factored:amp_phase eigen:s:logcosh eigen:s:amp_phase eigen:d:logcosh eigen:d:amp_phase" ;;
    learned_basis)
        SCRIPT="diagnostics/learned_basis_4x4.py"; SUBDIR="learned_basis_4x4"
        DEFAULT_FITS="eigen:frozen eigen:learned spin:frozen spin:learned random:learned" ;;
    *)
        echo "ERROR: DIAG must be 'sign' or 'learned_basis', got '$DIAG'." >&2; exit 1 ;;
esac
[[ -f "$REPO_ROOT/$SCRIPT" ]] || { echo "ERROR: $SCRIPT not found in $REPO_ROOT." >&2; exit 1; }
J2S="${J2S:-0.0 0.5}"
SEEDS="${SEEDS:-0 1}"
STEPS="${STEPS:-2000}"
FITS="${FITS:-$DEFAULT_FITS}"
read -r -a J2_ARR <<<"$J2S"; read -r -a SEED_ARR <<<"$SEEDS"; read -r -a FIT_ARR <<<"$FITS"
N_FITS=$(( ${#J2_ARR[@]} * ${#SEED_ARR[@]} * ${#FIT_ARR[@]} ))
PY_ARGS=(--require-gpu --steps "$STEPS" --j2 "${J2_ARR[@]}" --seeds "${SEED_ARR[@]}" --fits "${FIT_ARR[@]}")
if [[ "$DIAG" == "learned_basis" && -n "${BASIS_LR:-}" ]]; then PY_ARGS+=(--basis-lr "$BASIS_LR"); fi

COMMIT="$(git rev-parse HEAD)"; SHORT="$(git rev-parse --short HEAD)"
DIAG_ROOT="${DIAG_ROOT:-$REPO_ROOT/data/diagnostics}"
OUT="$DIAG_ROOT/$SUBDIR/$(date +%Y-%m-%d_%H-%M-%S)G${SHORT}"
mkdir -p "$OUT"
echo "queued" > "$OUT/STATUS"
echo "${CLUSTER_PYTHON_MODULE:-}" > "$OUT/python_module.txt"
cat > "$OUT/submit_info.txt" <<EOF
submitted:  $(date --iso-8601=seconds 2>/dev/null || date)
commit:     $COMMIT
cluster:    $CLUSTER_NAME  ($CLUSTER_DESCRIPTION)
resources:  ${CLUSTER_SBATCH_ARGS[*]}
diagnostic: $DIAG ($SCRIPT)
fits:       $N_FITS  (J2: $J2S | seeds: $SEEDS | steps: $STEPS)
            ${FITS}
command:    .venv/bin/python -u $SCRIPT --out $OUT ${PY_ARGS[*]}
EOF

echo
echo "Diag:    $DIAG ($SCRIPT)"
echo "Fits:    $N_FITS  (J2: $J2S | seeds: $SEEDS | steps: $STEPS)"
echo "Output:  $OUT"
JOB_ID="$(sbatch --parsable "${CLUSTER_SBATCH_ARGS[@]}" --job-name="diag_${SUBDIR}" \
          --output="$OUT/slurm-%j.out" --error="$OUT/slurm-%j.err" \
          "$REPO_ROOT/submit_diag.sh" --job "$OUT" "$SCRIPT" "${PY_ARGS[@]}")"
echo "job id:     $JOB_ID" >> "$OUT/submit_info.txt"

cat <<EOF

============================================================
SUBMITTED  job $JOB_ID
============================================================
Results folder:  $OUT

  cat  $OUT/STATUS                 # queued / running / done / failed
  tail -f $OUT/slurm-$JOB_ID.out   # live: one table row per finished fit
  cat  $OUT/summary.txt            # final ranked table (written at the end)
EOF
