#!/bin/bash
set -euo pipefail

# ==========================================================================
# submit_more.sh
#
# Submit more runs of a sweep that submit.sh already created:
#
#     ./submit_more.sh data/spinbasis/<sweep-id> 8-11
#
# For clusters with a per-user submit limit (Explorer: 8 queued jobs, and
# every array task counts). Submit the first part with
#     ARRAY=0-7 ./submit.sh config_spinbasis.toml
# and, once queue slots are free, the rest with this script. The runs land in
# the SAME sweep folder, with the same frozen configs (matrix.json) and the
# same cluster profile (cluster.txt) as the first part.
#
# Refuses runs that have already started (run_XXXX/manifest.json exists), so
# nothing is run twice. Warns if the checkout's commit differs from the one
# the sweep was created at: run.sh runs the CURRENT code.
# ==========================================================================

SWEEP_ARG="${1:?usage: ./submit_more.sh <sweep-dir> <runs, e.g. 8-11 or 3,5>}"
ARRAY_SPEC="${2:?usage: ./submit_more.sh <sweep-dir> <runs, e.g. 8-11 or 3,5>}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SWEEP_DIR="$(cd "$SWEEP_ARG" && pwd)"
MATRIX="$SWEEP_DIR/matrix.json"
VENV_PY="$REPO_ROOT/.venv/bin/python"

[[ -f "$MATRIX" ]] || { echo "ERROR: no matrix.json in $SWEEP_DIR" >&2; exit 1; }
[[ -f "$SWEEP_DIR/cluster.txt" ]] || { echo "ERROR: no cluster.txt in $SWEEP_DIR" >&2; exit 1; }
[[ -x "$VENV_PY" ]] || { echo "ERROR: $REPO_ROOT/.venv is missing. Run:  uv sync" >&2; exit 1; }

# Validate the runs and read the basis / commit the sweep was made at.
read -r BASIS SWEEP_COMMIT < <("$VENV_PY" - "$MATRIX" "$ARRAY_SPEC" <<'PY'
import json, re, sys
from pathlib import Path
matrix_path, spec = Path(sys.argv[1]), sys.argv[2]
m = json.load(open(matrix_path))
n = m["num_runs"]
ids = set()
for part in spec.split(","):
    mm = re.fullmatch(r"(\d+)(?:-(\d+))?", part.strip())
    if not mm:
        sys.exit(f"ERROR: runs {spec!r}: use ranges and commas, e.g. 8-11 or 3,5")
    a, b = int(mm.group(1)), int(mm.group(2) or mm.group(1))
    if a > b or b >= n:
        sys.exit(f"ERROR: runs {spec!r}: this sweep has runs 0-{n - 1}")
    ids.update(range(a, b + 1))
started = [i for i in sorted(ids) if (matrix_path.parent / f"run_{i:04d}" / "manifest.json").exists()]
if started:
    sys.exit(f"ERROR: run(s) {started} already started (manifest.json exists); not resubmitting them.")
print(m.get("basis", "eigen"), m.get("git_commit", "unknown"))
PY
)
[[ -n "${BASIS:-}" ]] || exit 1

case "$BASIS" in
    spin)  JOB_NAME="SB2D" ;;
    eigen) JOB_NAME="EB2D" ;;
    *) echo "ERROR: unknown basis '$BASIS' in $MATRIX" >&2; exit 1 ;;
esac

# Same cluster profile as the rest of the sweep.
CLUSTER_RECORDED="$(cat "$SWEEP_DIR/cluster.txt")"
# shellcheck source=clusters/common.sh
source "$REPO_ROOT/clusters/common.sh"
cluster_load "$CLUSTER_RECORDED"

HEAD_COMMIT="$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
echo "Sweep:    $SWEEP_DIR"
echo "Basis:    $BASIS ($JOB_NAME)    cluster: $CLUSTER_NAME"
echo "Runs:     $ARRAY_SPEC"
if [[ "$HEAD_COMMIT" != "$SWEEP_COMMIT" ]]; then
    echo "WARNING: this checkout is at ${HEAD_COMMIT:0:7}, the sweep was created at ${SWEEP_COMMIT:0:7}."
    echo "         The configs are frozen in matrix.json, but run.sh runs the current code."
fi
echo
read -r -p "Submit these runs? [y/N] " ANSWER
case "$ANSWER" in
    y|Y|yes|YES) ;;
    *) echo "Submission cancelled."; exit 0 ;;
esac

# run.sh finds the repo through $SLURM_SUBMIT_DIR, as for submit.sh.
cd "$REPO_ROOT"

if ! sbatch --test-only "${CLUSTER_SBATCH_ARGS[@]}" --job-name="$JOB_NAME" \
        --array="$ARRAY_SPEC" run.sh "$MATRIX" "$SWEEP_DIR"; then
    echo "ERROR: Slurm rejected the test submission. No job was submitted." >&2
    exit 1
fi

JOB_ID="$(sbatch --parsable "${CLUSTER_SBATCH_ARGS[@]}" --job-name="$JOB_NAME" \
        --array="$ARRAY_SPEC" run.sh "$MATRIX" "$SWEEP_DIR")" \
    || { echo "ERROR: sbatch failed; no job was submitted." >&2; exit 1; }

echo
echo "SUBMITTED runs $ARRAY_SPEC of $(basename "$SWEEP_DIR") as Slurm job $JOB_ID"
echo "  squeue --array -j $JOB_ID"
