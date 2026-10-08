#!/bin/bash

# Portable Slurm directives only. Everything that differs between clusters --
# partition, account, GPU request, memory, cores, walltime -- comes from the
# cluster profile in clusters/<name>.sh and is passed on the sbatch COMMAND
# LINE by submit.sh (and below, when this script resubmits itself). Slurm
# gives command-line options precedence over these in-script directives.
#
# Switch clusters with:   CLUSTER=aicr-batch ./submit.sh config_<basis>basis.toml
# See clusters/common.sh for how the profile is resolved.
#
# One run.sh serves both bases: it reads [system] basis from the run's
# config.json and runs <basis>basis/run_2D_<basis>basis.py. The job name
# (SB2D | EB2D) is set by submit.sh and re-passed below on resubmission.
#SBATCH --nodes=1
#SBATCH --job-name=J1J2
#SBATCH --ntasks=1
#SBATCH --output=run_hist/run_%A_%a.out
#SBATCH --error=run_hist/run_%A_%a.err
# stdout/stderr are redirected into the per-run directory below,
# after the job starts.

set +e

# ==========================================================================
# run.sh
#
# Arguments:
#
#     run.sh MATRIX.json SWEEP_DIR
#
# Normally called by submit.sh / Slurm.
# ==========================================================================

MATRIX="${1:?Missing matrix.json argument}"
SWEEP_DIR="${2:?Missing sweep directory argument}"

# Normalize to absolute paths. We cd into the repo root below, and both of
# these are re-used after that point.
if [[ ! -f "$MATRIX" ]]; then
    echo "ERROR: matrix.json does not exist: $MATRIX" >&2
    exit 1
fi
if [[ ! -d "$SWEEP_DIR" ]]; then
    echo "ERROR: Sweep directory does not exist: $SWEEP_DIR" >&2
    exit 1
fi
MATRIX="$(cd "$(dirname "$MATRIX")" && pwd)/$(basename "$MATRIX")"
SWEEP_DIR="$(cd "$SWEEP_DIR" && pwd)"

# --------------------------------------------------------------------------
# Check Slurm array environment
# --------------------------------------------------------------------------
echo "$SLURM_ARRAY_TASK_ID"
if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "ERROR: This script is intended to run as a Slurm array task." >&2
    exit 1
fi

RUN_ID="$SLURM_ARRAY_TASK_ID"

# --------------------------------------------------------------------------
# Determine this run's directory
# --------------------------------------------------------------------------

RUN_DIR="$SWEEP_DIR/run_$(printf '%04d' "$RUN_ID")"

if [[ ! -d "$RUN_DIR" ]]; then
    echo "ERROR: Run directory does not exist: $RUN_DIR" >&2
    exit 1
fi

# --------------------------------------------------------------------------
# Redirect stdout/stderr into the run directory.
#
# From this point onward, output from this script and child processes
# goes into these files.
# --------------------------------------------------------------------------

exec >> "$RUN_DIR/stdout.txt"
exec 2>> "$RUN_DIR/stderr.txt"

echo "============================================================"
echo "Heisenberg J1-J2 2D Run"
echo "============================================================"
echo "Started:      $(date --iso-8601=seconds)"
echo "Run ID:       $RUN_ID"
echo "Sweep ID:     ${SLURM_ARRAY_JOB_ID:-unknown}"
echo "Slurm job ID: ${SLURM_JOB_ID:-unknown}"
echo "Node:         ${SLURMD_NODENAME:-unknown}"
echo

# --------------------------------------------------------------------------
# Repo root
#
# Every project-relative path below (.venv, <basis>basis/, clusters/) needs
# the repo root. Slurm copies this script to a spool directory, so
# ${BASH_SOURCE[0]} is NOT the repo -- $SLURM_SUBMIT_DIR is (submit.sh cds to
# the git toplevel before calling sbatch). Try the candidates in order and
# take the first one that looks like this repo (run.sh + both basis folders).
# --------------------------------------------------------------------------

REPO_ROOT=""
for CAND in "${SLURM_SUBMIT_DIR:-}" "$(pwd)" "$(dirname "${BASH_SOURCE[0]}")"; do
    if [[ -n "$CAND" ]] && [[ -f "$CAND/run.sh" && -d "$CAND/spinbasis" && -d "$CAND/eigenbasis" ]]; then
        REPO_ROOT="$(cd "$CAND" && pwd)"
        break
    fi
done

if [[ -z "$REPO_ROOT" ]]; then
    echo "ERROR: could not locate the repo root (no run.sh + spinbasis/ + eigenbasis/ in" >&2
    echo "       SLURM_SUBMIT_DIR='${SLURM_SUBMIT_DIR:-}', cwd='$(pwd)')." >&2
    exit 1
fi

cd "$REPO_ROOT" || { echo "ERROR: cannot cd to $REPO_ROOT" >&2; exit 1; }

# --------------------------------------------------------------------------
# Cluster profile
#
# Resolved BEFORE any module command, because which python module to load is
# itself cluster-specific.
#
# The sweep records the cluster it was submitted on in cluster.txt, and that
# record wins: autodetection runs on a COMPUTE node here, where the hostname
# differs from the login node, and a resumed sweep must keep using the
# partition and account it started on.
# --------------------------------------------------------------------------

source "$REPO_ROOT/clusters/common.sh"

CLUSTER_RECORD="$SWEEP_DIR/cluster.txt"
if [[ -f "$CLUSTER_RECORD" ]]; then
    RECORDED_CLUSTER="$(tr -d '[:space:]' < "$CLUSTER_RECORD")"
else
    RECORDED_CLUSTER=""
fi

if ! cluster_load "$RECORDED_CLUSTER"; then
    echo "ERROR: could not load a cluster profile for this job." >&2
    exit 1
fi

echo "Cluster:      $CLUSTER_NAME (${CLUSTER_SOURCE})"

# --------------------------------------------------------------------------
# Interpreter
#
# Purge modules FIRST, before anything runs python, so every python in this
# script sees one consistent environment.
#
# Guarded so the script can also be dry-run off-cluster, where Environment
# Modules does not exist. (`module` is a shell function, which command -v
# finds.)
# --------------------------------------------------------------------------

if command -v module >/dev/null 2>&1; then
    module purge
fi

# Call the venv interpreter by ABSOLUTE PATH. Do not rely on
# `source .venv/bin/activate` plus a bare `python`:
#
#   - uv bakes an absolute VIRTUAL_ENV path into .venv/bin/activate. Move or
#     rename the repo and activate silently prepends a directory that does
#     not exist to PATH -- no error, no warning -- so bare `python` falls
#     through to whatever else is on PATH. On a compute node that is
#     /usr/bin/python, i.e. Python 3.6.
#   - `module purge` removes the python module inherited from
#     submit.sh, so there is nothing sane left for `python` to resolve to.
#
# That combination silently ran this project under Python 3.6 and produced a
# bare `File "<fstring>", line 1 / (fname=) / SyntaxError` with no filename
# and no traceback -- the f-string '=' specifier needs >= 3.8.

VENV_PY="$REPO_ROOT/.venv/bin/python"

if [[ ! -x "$VENV_PY" ]]; then
    echo "ERROR: venv interpreter not found or not executable:" >&2
    echo "       $VENV_PY" >&2
    echo "       Recreate it in place:" >&2
    echo "         cd $REPO_ROOT && rm -rf .venv && uv sync" >&2
    exit 1
fi

# A dangling symlink still looks executable but cannot run, and the venv's
# base interpreter may have come from a module we just purged.
if ! "$VENV_PY" -c 'pass' >/dev/null 2>&1; then
    echo "WARNING: $VENV_PY is not runnable after 'module purge';" >&2
    echo "         retrying with $CLUSTER_PYTHON_MODULE loaded." >&2
    if command -v module >/dev/null 2>&1; then
        module load "$CLUSTER_PYTHON_MODULE"
    fi
    if ! "$VENV_PY" -c 'pass' >/dev/null 2>&1; then
        echo "ERROR: venv interpreter is broken. It points at:" >&2
        echo "       $(readlink -f "$VENV_PY" 2>/dev/null || echo '<unresolvable>')" >&2
        echo "       Recreate it:  cd $REPO_ROOT && rm -rf .venv && uv sync" >&2
        exit 1
    fi
fi

# Hard version gate: fail here, loudly, rather than at compile time with a
# one-line SyntaxError. pyproject.toml requires >= 3.10.
if ! "$VENV_PY" -c 'import sys; sys.exit(0 if sys.version_info[:2] >= (3, 10) else 1)'; then
    echo "ERROR: Python >= 3.10 is required, got:" >&2
    "$VENV_PY" -V >&2
    echo "       Recreate the venv:  cd $REPO_ROOT && rm -rf .venv && uv sync" >&2
    exit 1
fi

echo "Repo root:    $REPO_ROOT"
echo "Interpreter:  $VENV_PY"
echo "Python:       $("$VENV_PY" -V 2>&1)"
echo

# --------------------------------------------------------------------------
# Get metadata and resolved configuration for this run.
# --------------------------------------------------------------------------

"$VENV_PY" - "$MATRIX" "$RUN_ID" "$RUN_DIR" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone

matrix_path = sys.argv[1]
run_id = int(sys.argv[2])
run_dir = sys.argv[3]

with open(matrix_path) as f:
    matrix = json.load(f)

try:
    run = matrix["runs"][run_id]
except IndexError:
    raise SystemExit(
        f"ERROR: No run {run_id} exists in {matrix_path}"
    )

# ----------------------------------------------------------------------
# Save the exact resolved configuration.
# ----------------------------------------------------------------------

with open(os.path.join(run_dir, "config.json"), "w") as f:
    json.dump(run["config"], f, indent=2)
    f.write("\n")

# ----------------------------------------------------------------------
# Create the manifest.
# ----------------------------------------------------------------------

manifest = {
    "sweep_id": matrix["sweep_id"],
    "run_id": run_id,
    "basis": run["config"]["system"]["basis"],

    "git_commit": matrix["git_commit"],

    "submitted_config": matrix["source_config"],

    "sweep_parameters": matrix["sweep_parameters"],
    "sweep_values": run["sweep_values"],

    "config": run["config"],

    "slurm": {
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "array_task_count": os.environ.get("SLURM_ARRAY_TASK_COUNT"),
        "node": os.environ.get("SLURMD_NODENAME"),
    },

    "started_at": datetime.now(timezone.utc).isoformat(),
}

with open(os.path.join(run_dir, "manifest.json"), "w") as f:
    json.dump(manifest, f, indent=2)
    f.write("\n")
PY

echo "Resolved configuration written to:"
echo "  $RUN_DIR/config.json"

echo
echo "Manifest written to:"
echo "  $RUN_DIR/manifest.json"

# --------------------------------------------------------------------------
# Load the resolved configuration into a temporary file.
#
# Your real simulation can replace this with whatever invocation it needs.
# --------------------------------------------------------------------------

CONFIG_JSON="$RUN_DIR/config.json"

# The basis picks the entry point. Each entry point imports its own
# <basis>basis/helpers: python puts the script's directory first on sys.path,
# so the two `helpers` packages never collide.
BASIS="$("$VENV_PY" -c 'import json, sys; print(json.load(open(sys.argv[1]))["system"]["basis"])' "$CONFIG_JSON")"
case "$BASIS" in
    spin)  JOB_NAME="SB2D" ;;
    eigen) JOB_NAME="EB2D" ;;
    *)     echo "ERROR: unknown [system] basis '$BASIS' in $CONFIG_JSON" >&2; exit 1 ;;
esac
ENTRY="$REPO_ROOT/${BASIS}basis/run_2D_${BASIS}basis.py"
if [[ ! -f "$ENTRY" ]]; then
    echo "ERROR: entry point not found: $ENTRY" >&2
    exit 1
fi
echo "Basis:        $BASIS  ($ENTRY)"

echo
echo "Configuration:"
cat "$CONFIG_JSON"

echo
echo "Starting simulation..."

# --------------------------------------------------------------------------
# YOUR REAL SIMULATION GOES HERE.
#
# `module purge` and interpreter resolution already happened at the top.
# --------------------------------------------------------------------------
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export JAX_TRACEBACK_FILTERING=off

"$VENV_PY" -u "$ENTRY" "$CONFIG_JSON" "$RUN_DIR"
PYEXIT=$?

echo
if [[ "$PYEXIT" -eq 0 ]]; then
    echo "Simulation exited cleanly (code 0)."
else
    echo "ERROR: simulation exited with code $PYEXIT."
    echo "       See $RUN_DIR/stderr.txt"
fi

set -e

# --------------------------------------------------------------------------
# Recover the simulation's file prefix, then classify the outcome.
#
# the entry point writes its prefix to $RUN_DIR/.runinfo_$SLURM_JOB_ID
# once setup completes, so the file's absence means python died before setup.
#
# FNAME must be resolved BEFORE the status check. It used to be read only
# inside the resubmit block further down, so ${FNAME} was empty here and
# ${FNAME}.done / ${FNAME}.timeout could never match -- every run, including
# ones that reached their target and wrote .done, was recorded as "failed".
# --------------------------------------------------------------------------

RUNINFO="$RUN_DIR/.runinfo_${SLURM_JOB_ID}"
FNAME=""

if [[ -f "$RUNINFO" ]]; then
    FNAME="$(cat "$RUNINFO")"
    rm -f "$RUNINFO"
fi

if [[ -z "$FNAME" ]]; then
    STATUS="crashed_before_setup"
elif [[ "$PYEXIT" -ne 0 ]]; then
    STATUS="failed"
elif [[ -f "${FNAME}.done" ]]; then
    STATUS="completed"
elif [[ -f "${FNAME}.timeout" ]]; then
    STATUS="timeout"
elif [[ -f "${FNAME}.frozen" ]]; then
    # The spin-basis freeze guard stopped a collapsed wavefunction on purpose:
    # a verdict, not a crash. Never resubmitted (that needs .timeout).
    STATUS="frozen"
else
    STATUS="failed"
fi

echo
echo "Status: $STATUS"

"$VENV_PY" - "$RUN_DIR/manifest.json" "$STATUS" <<'PY'
import json
import sys
from datetime import datetime, timezone

path = sys.argv[1]

with open(path) as f:
    manifest = json.load(f)

manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
manifest["status"] = sys.argv[2]

with open(path, "w") as f:
    json.dump(manifest, f, indent=2)
    f.write("\n")
PY

echo
echo "Finished: $(date --iso-8601=seconds)"

# ==========================================================================
# Auto-resubmit ONLY on a clean time-stop
#
# Resubmit iff:
#   1. The simulation exited successfully (PYEXIT == 0)
#   2. A .timeout sentinel exists
#   3. A .done sentinel does NOT exist
#
# The same sweep/task is resubmitted, so:
#
#   SLURM_ARRAY_TASK_ID
#   RUN_DIR
#   MATRIX
#
# all remain associated with the same logical simulation.
#
# FNAME / RUNINFO were already resolved and consumed by the status block
# above, so this branches on $STATUS rather than re-reading them.
# ==========================================================================

case "$STATUS" in

    crashed_before_setup)

        echo "WARNING: no runinfo file (python crashed before setup)"
        echo "         — NOT resubmitting."
        echo "         Check: $RUN_DIR/stderr.txt"
        ;;

    completed)

        echo "Target reached — no resubmission."
        ;;

    timeout)

        echo "Clean wall-clock stop — resubmitting to continue from checkpoint."

        # Record the continuation in the manifest before resubmitting.
        "$VENV_PY" - "$RUN_DIR/manifest.json" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone

path = sys.argv[1]

with open(path) as f:
    manifest = json.load(f)

manifest.setdefault("resubmissions", []).append({
    "job_id": os.environ.get("SLURM_JOB_ID"),
    "array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
    "time": datetime.now(timezone.utc).isoformat(),
    "reason": "clean wall-clock timeout",
})

with open(path, "w") as f:
    json.dump(manifest, f, indent=2)
    f.write("\n")
PY

        # Resubmit ONLY this array task.
        #
        # --array="$SLURM_ARRAY_TASK_ID"
        # means that if this is task 17, we submit task 17 again,
        # rather than launching the entire sweep again.
        #
        # Resubmit the repo's copy of this script, NOT "$0": Slurm runs a
        # spool copy of the batch script, and that path disappears when the
        # job's spool directory is cleaned up.
        # The resource flags must be repeated here: they are no longer
        # #SBATCH directives inside run.sh, so a bare resubmission would
        # land on the cluster's default partition with default memory.
        sbatch \
            "${CLUSTER_SBATCH_ARGS[@]}" \
            --job-name="$JOB_NAME" \
            --array="$SLURM_ARRAY_TASK_ID" \
            "$REPO_ROOT/run.sh" \
            "$MATRIX" \
            "$SWEEP_DIR"
        ;;

    *)

        echo "Python exited $PYEXIT with no clean time-stop — NOT resubmitting."
        echo "Check: $RUN_DIR/stderr.txt"
        ;;

esac
