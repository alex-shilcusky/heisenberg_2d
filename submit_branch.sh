#!/bin/bash
set -euo pipefail

# ==========================================================================
# submit_branch.sh
#
# Run this on the HPC login node:
#
#     ./submit_branch.sh --from <sweep-or-run-dir> --run N --extra 5000
#
# The submit.sh counterpart for CONTINUING an existing run instead of starting
# a sweep. It validates the repository and cluster profile exactly as submit.sh
# does, prepares a new one-run data folder seeded with the parent's checkpoint
# (and pointing back at it), asks for confirmation, tests the Slurm
# submission, and then submits a single array task.
#
# WHY NOT JUST RE-RUN run.sh ON THE OLD FOLDER
# --------------------------------------------
# run.sh already continues a run IN PLACE -- that is how the automatic
# resubmission after a clean wall-clock stop works, and it is the right thing
# when a job simply ran out of window. But it overwrites run.mpack and appends
# to the same logs, so the state you already paid for is gone. Use this script
# when you want to push a finished run somewhere new -- more steps, a different
# learning rate, symmetries switched on, a different J2 -- while keeping the
# original intact.
#
# WHAT GETS SUBMITTED
# -------------------
# The branch folder is an ordinary one-run sweep folder: matrix.json plus
# run_0000/. So the submission below is the same sbatch line submit.sh uses,
# with --array=0, and run.sh needs no changes -- including its self-resubmission
# after a timeout, which passes the same (matrix, sweep_dir) back to sbatch.
# That matters: a few thousand steps will not fit in one window.
#
# OPTIONS
# -------
#   --from DIR        sweep folder (with --run N) or a run_XXXX folder   [required]
#   --run N           run number inside the sweep folder                 [default 0]
#   --extra N         run N MORE steps than the parent reached           [one of
#   --total N         set the total step target to N                      these two]
#   --set PATH=VALUE  patch the config, repeatable, e.g.
#                     --set optimizer.final_learning_rate=0.005
#   --fresh           ignore the parent's checkpoint and start from a random
#                     init, keeping the provenance link. REQUIRED when an
#                     override changes the parameter tree (layers, ratio,
#                     heads, attention_mode, embedding, L, transl_invariant,
#                     symmetry.*), because run.mpack only restores into an
#                     identically shaped model.
#   --no-history      do not copy the parent's logs/diagnostics into the branch
#   --name NAME       branch folder name  [default: <stamp>G<commit>P<pid>_branch]
#   --yes             skip the confirmation prompt
#   --dry-run         prepare and validate everything, but do not submit
#
# The branch folder is created next to its parent sweep (so a spin-basis
# branch lands in data/spinbasis/, an eigenbasis one in data/eigenbasis/).
# Override with RUN_ROOT=/some/existing/path; the cluster with CLUSTER=...,
# exactly as for submit.sh.
# ==========================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_ROOT="${RUN_ROOT:-}"

BRANCH_TOOL="$SCRIPT_DIR/branch_run.py"

# --------------------------------------------------------------------------
# Arguments
#
# Most flags are forwarded verbatim to branch_run.py, which owns the parent
# lookup, the config patching, the parameter-tree safety check and the
# provenance record. Only the ones this script itself acts on are consumed.
# --------------------------------------------------------------------------
FORWARD=()
NAME=""
ASSUME_YES=0
DRY_RUN=0
HAVE_FROM=0
HAVE_STEPS=0

usage() {
    sed -n '/^# OPTIONS/,/^# =====/p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit "${1:-1}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --from)       HAVE_FROM=1; FORWARD+=("$1" "${2:?--from needs a directory}"); shift 2 ;;
        --run)        FORWARD+=("$1" "${2:?--run needs a number}"); shift 2 ;;
        --extra|--total)
                      HAVE_STEPS=1; FORWARD+=("$1" "${2:?$1 needs a number}"); shift 2 ;;
        --set)        FORWARD+=("$1" "${2:?--set needs PATH=VALUE}"); shift 2 ;;
        --fresh|--no-history)
                      FORWARD+=("$1"); shift ;;
        --name)       NAME="${2:?--name needs a value}"; shift 2 ;;
        --yes|-y)     ASSUME_YES=1; shift ;;
        --dry-run)    DRY_RUN=1; shift ;;
        -h|--help)    usage 0 ;;
        *)            echo "ERROR: unknown argument: $1" >&2; echo >&2; usage 1 ;;
    esac
done

if [[ "$HAVE_FROM" -eq 0 ]]; then
    echo "ERROR: --from is required." >&2; echo >&2; usage 1
fi
if [[ "$HAVE_STEPS" -eq 0 ]]; then
    echo "ERROR: one of --extra N or --total N is required." >&2; echo >&2; usage 1
fi

OUT_ROOT_ARGS=()
if [[ -n "$RUN_ROOT" ]]; then
    echo "Run root:       $RUN_ROOT"

    if [[ ! -d "$RUN_ROOT" ]]; then
        echo "ERROR: Run root does not exist: $RUN_ROOT" >&2
        exit 1
    fi

    TEST_FILE="$RUN_ROOT/.write_test_$$"
    if ! touch "$TEST_FILE" 2>/dev/null; then
        echo "ERROR: Cannot write to $RUN_ROOT" >&2
        exit 1
    fi
    rm -f "$TEST_FILE"
    OUT_ROOT_ARGS=(--out-root "$RUN_ROOT")
else
    echo "Run root:       next to the parent sweep"
fi

# --------------------------------------------------------------------------
# Basic checks
# --------------------------------------------------------------------------
if [[ ! -f "$BRANCH_TOOL" ]]; then
    echo "ERROR: branch_run.py not found next to this script: $BRANCH_TOOL" >&2
    exit 1
fi

if ! command -v git >/dev/null 2>&1; then
    echo "ERROR: git not found." >&2
    exit 1
fi

if ! command -v module >/dev/null 2>&1; then
    echo "ERROR: Environment Modules is not available." >&2
    exit 1
fi

# --------------------------------------------------------------------------
# Cluster profile
#
# Resolved before the module load, because which python module exists is
# itself cluster-specific. Autodetects from the Slurm partitions visible on
# this login node; override with CLUSTER=explorer ./submit_branch.sh ...
# --------------------------------------------------------------------------
source "$SCRIPT_DIR/clusters/common.sh"
cluster_load

echo
cluster_summary

module load "$CLUSTER_PYTHON_MODULE"

if ! command -v python >/dev/null 2>&1; then
    echo "ERROR: python not found." >&2
    exit 1
fi

if ! command -v sbatch >/dev/null 2>&1; then
    echo "ERROR: sbatch not found. Are you on the HPC login node?" >&2
    exit 1
fi

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "ERROR: Current directory is not inside a Git repository." >&2
    exit 1
fi

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

# --------------------------------------------------------------------------
# Check that the Git working tree is clean
#
# Same rule as submit.sh: the branch records the submitting commit in its
# manifest and parent.json, and that record is worthless if the tree was dirty.
# --------------------------------------------------------------------------
if [[ -n "$(git status --porcelain)" ]]; then
    echo
    echo "ERROR: Git working directory is dirty."
    echo
    git status --short
    echo
    echo "Commit or stash your changes before submitting a run."
    exit 1
fi

GIT_COMMIT="$(git rev-parse HEAD)"
GIT_SHORT_COMMIT="$(git rev-parse --short HEAD)"

echo "Git repository: $REPO_ROOT"
echo "Git commit:     $GIT_COMMIT"
echo "Python:         $(python --version 2>&1)"

RUN_SCRIPT="$REPO_ROOT/run.sh"
if [[ ! -f "$RUN_SCRIPT" ]]; then
    echo "ERROR: Run script not found: $RUN_SCRIPT" >&2
    exit 1
fi
if [[ ! -x "$RUN_SCRIPT" ]]; then
    echo "ERROR: Run script is not executable: $RUN_SCRIPT" >&2
    echo "       Run: chmod +x \"$RUN_SCRIPT\"" >&2
    exit 1
fi

# --------------------------------------------------------------------------
# Prepare the branch folder
#
# Named the same way submit.sh names a sweep -- timestamp first, so it sorts
# chronologically next to the real sweeps in the plotting notebooks, with the
# submitting commit and pid for traceability -- plus a _branch suffix so it is
# obvious what it is.
#
# branch_run.py does the work and launches nothing on its own. --cluster pins
# the profile resolved above into the branch's cluster.txt, which is where
# run.sh reads it from (including on its self-resubmissions).
# --------------------------------------------------------------------------
TIMESTAMP="$(date +%Y-%m-%d_%H-%M-%S)"
BRANCH_ID="${NAME:-${TIMESTAMP}G${GIT_SHORT_COMMIT}P$$_branch}"

EMIT="$(mktemp)"
trap 'rm -f "$EMIT"' EXIT

echo
echo "Preparing branch folder..."
echo

python "$BRANCH_TOOL" \
    "${FORWARD[@]}" \
    ${OUT_ROOT_ARGS[@]+"${OUT_ROOT_ARGS[@]}"} \
    --name "$BRANCH_ID" \
    --cluster "$CLUSTER_NAME" \
    --emit-dir "$EMIT"

BRANCH_DIR="$(tr -d '[:space:]' < "$EMIT")"
if [[ -z "$BRANCH_DIR" || ! -d "$BRANCH_DIR" ]]; then
    echo "ERROR: branch_run.py did not produce a folder." >&2
    exit 1
fi

MATRIX="$BRANCH_DIR/matrix.json"
if [[ ! -f "$MATRIX" ]]; then
    echo "ERROR: no matrix.json in $BRANCH_DIR" >&2
    exit 1
fi
case "$(python -c 'import json, sys; print(json.load(open(sys.argv[1]))["basis"])' "$MATRIX")" in
    spin)  JOB_NAME="SB2D" ;;
    eigen) JOB_NAME="EB2D" ;;
    *)     echo "ERROR: no basis recorded in $MATRIX" >&2; exit 1 ;;
esac

# --------------------------------------------------------------------------
# Walltime consistency
#
# MAX_RUNTIME_MIN is what stops the run gracefully and writes the checkpoint.
# If it is not safely below the profile's --time, Slurm hard-kills the job
# first and that window's progress is lost. branch_run.py warns about this;
# here it is a hard error, as in submit.sh.
# --------------------------------------------------------------------------
WALL_MIN="$(cluster_walltime_minutes "$CLUSTER_WALLTIME" || true)"
CONFIG_MAX_RUNTIME="$(python - "$MATRIX" <<'PY'
import json, sys
with open(sys.argv[1]) as f:
    matrix = json.load(f)
print(max(r["config"]["run"]["MAX_RUNTIME_MIN"] for r in matrix["runs"]))
PY
)"

if [[ -z "$WALL_MIN" ]]; then
    echo "WARNING: could not parse CLUSTER_WALLTIME='$CLUSTER_WALLTIME';" >&2
    echo "         skipping the MAX_RUNTIME_MIN consistency check." >&2
elif [[ "$CONFIG_MAX_RUNTIME" -ge "$WALL_MIN" ]]; then
    echo >&2
    echo "ERROR: MAX_RUNTIME_MIN ($CONFIG_MAX_RUNTIME min) is not below the" >&2
    echo "       $CLUSTER_NAME walltime $CLUSTER_WALLTIME ($WALL_MIN min)." >&2
    echo "       Slurm would hard-kill the job before the checkpoint is" >&2
    echo "       written, losing that window's progress." >&2
    echo >&2
    echo "       Re-run with --set run.MAX_RUNTIME_MIN=$CLUSTER_MAX_RUNTIME_MIN," >&2
    echo "       or raise CLUSTER_WALLTIME in clusters/$CLUSTER_NAME.sh." >&2
    echo >&2
    echo "       The prepared folder is left in place for inspection:" >&2
    echo "       $BRANCH_DIR" >&2
    exit 1
elif [[ "$CONFIG_MAX_RUNTIME" -gt "$((WALL_MIN - 5))" ]]; then
    echo "WARNING: MAX_RUNTIME_MIN ($CONFIG_MAX_RUNTIME min) leaves under 5 min" >&2
    echo "         of margin below the $CLUSTER_WALLTIME walltime." >&2
elif [[ "$((CONFIG_MAX_RUNTIME * 2))" -lt "$WALL_MIN" ]]; then
    # The opposite mistake to the one above, and a silent one: the job is legal,
    # it just parks a long allocation and stops itself early. Switching to a
    # bigger partition without raising MAX_RUNTIME_MIN lands exactly here.
    echo >&2
    echo "WARNING: MAX_RUNTIME_MIN ($CONFIG_MAX_RUNTIME min) is less than half the" >&2
    echo "         $CLUSTER_NAME walltime $CLUSTER_WALLTIME ($WALL_MIN min)." >&2
    echo "         The job will hold the allocation for $WALL_MIN min and stop" >&2
    echo "         itself after $CONFIG_MAX_RUNTIME, needing more windows than" >&2
    echo "         necessary. This profile recommends $CLUSTER_MAX_RUNTIME_MIN." >&2
    echo "         Re-run with --set run.MAX_RUNTIME_MIN=$CLUSTER_MAX_RUNTIME_MIN." >&2
fi

# --------------------------------------------------------------------------
# What is about to run
# --------------------------------------------------------------------------
echo
python - "$BRANCH_DIR/parent.json" <<'PY'
import json, sys
with open(sys.argv[1]) as f:
    prov = json.load(f)
p, b = prov["parent"], prov["branch"]
print("============================================================")
print("BRANCH")
print("============================================================")
print(f"parent sweep : {p['sweep_id'] or '(none)'}")
print(f"parent run   : {p['run_dir']}")
print(f"parent steps : {p['applied_steps']}")
ck = p["checkpoint"]
print(f"checkpoint   : {'reused' if ck['reused'] else 'NOT reused (fresh init)'}"
      + (f"  sha256 {ck['sha256'][:12]}" if ck.get("sha256") else ""))
print(f"steps        : {b['start_step']} -> {b['target_step']}  "
      f"({b['steps_to_run']} to run)")
print(f"history      : {'copied' if b['history_copied'] else 'not copied'}")
if b["overrides"]:
    print("overrides    :")
    for k, v in b["overrides"].items():
        print(f"               {k} = {v!r}")
else:
    print("overrides    : (none -- same config, more steps)")
if b["physics_changed"]:
    print(f"NOTE         : changes the Hamiltonian ({', '.join(b['physics_changed'])}),")
    print( "               so this solves a different problem than its parent")
PY

echo
echo "Output:         $BRANCH_DIR"

if [[ "$DRY_RUN" -eq 1 ]]; then
    echo
    echo "--dry-run: nothing submitted. To submit it as prepared:"
    echo "  sbatch ${CLUSTER_SBATCH_ARGS[*]} --array=0 $RUN_SCRIPT $MATRIX $BRANCH_DIR"
    exit 0
fi

# --------------------------------------------------------------------------
# Confirmation
# --------------------------------------------------------------------------
echo
echo "This will submit one Slurm job to $CLUSTER_NAME."

if [[ "$ASSUME_YES" -eq 0 ]]; then
    echo
    read -r -p "Submit this run? [y/N] " ANSWER
    case "$ANSWER" in
        y|Y|yes|YES) ;;
        *)
            echo
            echo "Aborted. The prepared folder is left in place:"
            echo "  $BRANCH_DIR"
            echo "To submit it later:"
            echo "  sbatch ${CLUSTER_SBATCH_ARGS[*]} --array=0 $RUN_SCRIPT $MATRIX $BRANCH_DIR"
            exit 0
            ;;
    esac
fi

# --------------------------------------------------------------------------
# Slurm preflight, then submit. One run -> one array task, --array=0.
# --------------------------------------------------------------------------
echo
echo "Running Slurm preflight check..."

if ! sbatch \
    --test-only \
    "${CLUSTER_SBATCH_ARGS[@]}" \
    --job-name="$JOB_NAME" \
    --array=0 \
    "$RUN_SCRIPT" \
    "$MATRIX" \
    "$BRANCH_DIR"
then
    echo
    echo "ERROR: Slurm rejected the test submission."
    echo "No job was submitted. The prepared folder is left in place:"
    echo "  $BRANCH_DIR"
    exit 1
fi

echo
echo "Submitting..."

JOB_ID="$(
    sbatch \
        --parsable \
        "${CLUSTER_SBATCH_ARGS[@]}" \
        --job-name="$JOB_NAME" \
        --array=0 \
        "$RUN_SCRIPT" \
        "$MATRIX" \
        "$BRANCH_DIR"
)"

echo
echo "============================================================"
echo "SUBMITTED"
echo "============================================================"
echo "Slurm job ID: $JOB_ID"
echo "Branch ID:    $BRANCH_ID"
echo "Output:       $BRANCH_DIR"
echo
echo "Monitor with:"
echo "  squeue --array -j $JOB_ID"
echo
echo "Cancel with:"
echo "  scancel $JOB_ID"
echo
echo "Live energy:"
echo "  tail -f $BRANCH_DIR/run_0000/stdout.txt"
