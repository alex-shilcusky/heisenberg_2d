#!/bin/bash
set -euo pipefail

# ==========================================================================
# submit.sh
#
# Run this on the HPC login node, from the repo:
#
#     CLUSTER=aicr-batch ./submit.sh config_spinbasis.toml
#     CLUSTER=aicr-batch ./submit.sh config_eigenbasis.toml
#
# The config's [system] basis ("spin" | "eigen") selects the code
# (<basis>basis/run_2D_<basis>basis.py), the data folder (data/<basis>basis/)
# and the Slurm job name (SB2D | EB2D). The script validates the
# repository/configuration, expands the parameter sweep, asks for
# confirmation, tests the Slurm submission, and then submits one Slurm array
# task per parameter combination.
# ==========================================================================

CONFIG="${1:?usage: ./submit.sh config_spinbasis.toml | config_eigenbasis.toml}"
ORIG_PWD="$(pwd)"

# ${BASH_SOURCE[0]} is this file's path, so this is independent of the cwd the
# script was invoked from.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --------------------------------------------------------------------------
# Basic checks
# --------------------------------------------------------------------------

if [[ ! -f "$CONFIG" ]]; then
    echo "ERROR: Configuration file not found: $CONFIG" >&2
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
# this login node; override with CLUSTER=explorer ./submit.sh config_eigenbasis.toml
# --------------------------------------------------------------------------

source "$SCRIPT_DIR/clusters/common.sh"
cluster_load

echo
cluster_summary
echo

module load "$CLUSTER_PYTHON_MODULE"

if ! command -v python >/dev/null 2>&1; then
    echo "ERROR: python not found." >&2
    exit 1
fi

if ! command -v sbatch >/dev/null 2>&1; then
    echo "ERROR: sbatch not found. Are you on the HPC login node?" >&2
    exit 1
fi
# --------------------------------------------------------------------------
# Check that we are inside a Git repository
# --------------------------------------------------------------------------

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "ERROR: Current directory is not inside a Git repository." >&2
    exit 1
fi

REPO_ROOT="$(git rev-parse --show-toplevel)"

# Make paths relative to the repository root.
cd "$REPO_ROOT"

CONFIG="$(realpath "$CONFIG")"

# --------------------------------------------------------------------------
# Check that the Git working tree is clean
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

# --------------------------------------------------------------------------
# Validate Python / tomllib
# --------------------------------------------------------------------------

if ! python -c '
import sys
if sys.version_info < (3, 11):
    raise SystemExit("Python >= 3.11 is required.")
import tomllib
' >/dev/null 2>&1; then

    echo "ERROR: Python >= 3.11 with tomllib is required." >&2
    echo "       Current Python:" >&2
    python --version >&2
    exit 1
fi

echo "Python:         $(python --version)"

# --------------------------------------------------------------------------
# Basis
#
# [system] basis picks the code, the data folder and the job name. The data
# folder defaults to data/<basis>basis inside this repo (gitignored, created on
# first use). Override with RUN_ROOT=/some/existing/path ./submit.sh ...; a
# relative RUN_ROOT is taken relative to where you ran submit.sh from.
# --------------------------------------------------------------------------

BASIS="$(python - "$CONFIG" <<'PY'
import sys, tomllib
with open(sys.argv[1], "rb") as f:
    b = tomllib.load(f).get("system", {}).get("basis")
if b not in ("spin", "eigen"):
    raise SystemExit(f'ERROR: [system] basis must be "spin" or "eigen", got {b!r}')
print(b)
PY
)"

ENTRY="$REPO_ROOT/${BASIS}basis/run_2D_${BASIS}basis.py"
if [[ ! -f "$ENTRY" ]]; then
    echo "ERROR: entry point for basis '$BASIS' not found: $ENTRY" >&2
    exit 1
fi
case "$BASIS" in
    spin)  JOB_NAME="SB2D" ;;
    eigen) JOB_NAME="EB2D" ;;
esac

DEFAULT_RUN_ROOT="$REPO_ROOT/data/${BASIS}basis"
if [[ -n "${RUN_ROOT:-}" && "$RUN_ROOT" != /* ]]; then
    RUN_ROOT="$ORIG_PWD/$RUN_ROOT"
fi
RUN_ROOT="${RUN_ROOT:-$DEFAULT_RUN_ROOT}"

# Create the default location on first use. An explicit RUN_ROOT override must
# already exist, so a typo in it fails here instead of scattering runs into a
# new directory.
if [[ "$RUN_ROOT" == "$DEFAULT_RUN_ROOT" ]]; then
    mkdir -p "$RUN_ROOT"
fi

echo "Basis:          $BASIS  ($ENTRY, job name $JOB_NAME)"
echo "Run root:       $RUN_ROOT"

if [[ ! -d "$RUN_ROOT" ]]; then
    echo "ERROR: Run root does not exist: $RUN_ROOT" >&2
    exit 1
fi

if [[ ! -w "$RUN_ROOT" ]]; then
    echo "ERROR: Run root is not writable: $RUN_ROOT" >&2
    exit 1
fi

TEST_FILE="$RUN_ROOT/.write_test_$$"

if ! touch "$TEST_FILE"; then
    echo "ERROR: Cannot write to $RUN_ROOT" >&2
    exit 1
fi

rm "$TEST_FILE"

# --------------------------------------------------------------------------
# Configuration/output locations
# --------------------------------------------------------------------------

if ! mkdir -p "$RUN_ROOT"; then
    echo "ERROR: Cannot create output directory: $RUN_ROOT" >&2
    exit 1
fi

if [[ ! -w "$RUN_ROOT" ]]; then
    echo "ERROR: Output directory is not writable: $RUN_ROOT" >&2
    exit 1
fi

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
# Parse config and expand sweep
#
# This Python program:
#   1. Reads the TOML.
#   2. Finds [sweep.*] values.
#   3. Generates the Cartesian product.
#   4. Creates a JSON matrix containing the resolved config for every run.
# --------------------------------------------------------------------------

TIMESTAMP="$(date +"%Y-%m-%d_%H-%M-%S")"

SWEEP_ID="${TIMESTAMP}G${GIT_SHORT_COMMIT}P$$"
SWEEP_DIR="$RUN_ROOT/$SWEEP_ID"
MATRIX="$SWEEP_DIR/matrix.json"

mkdir -p "$SWEEP_DIR"

# Until the array is actually submitted, any exit (a failed check, a refused
# preflight, "N" at the prompt) removes the half-built sweep folder, so
# data/<basis>basis only ever holds sweeps that were submitted.
SUBMITTED=0
trap '[[ "$SUBMITTED" == 1 ]] || rm -rf "$SWEEP_DIR"' EXIT

# run.sh reads this back on the compute node, where autodetection cannot see
# the login node's hostname and where a resumed sweep must stay on the
# cluster it started on.
echo "$CLUSTER_NAME" > "$SWEEP_DIR/cluster.txt"

echo
echo "Parsing configuration and expanding sweep..."

python - "$CONFIG" "$MATRIX" "$SWEEP_ID" "$GIT_COMMIT" <<'PY'
import copy
import itertools
import json
import sys
import tomllib

config_path = sys.argv[1]
matrix_path = sys.argv[2]
sweep_id = sys.argv[3]
git_commit = sys.argv[4]

with open(config_path, "rb") as f:
    config = tomllib.load(f)

sweep = config.pop("sweep", {})
sweep_zip = config.pop("sweep_zip", {})

# ----------------------------------------------------------------------
# Flatten sweep tables into:
#
#   ("physics.viscosity", [0.001, 0.005, 0.01])
#
# ----------------------------------------------------------------------

def flatten_sweep(obj, prefix="", table="sweep"):
    result = []

    for key, value in obj.items():
        path = f"{prefix}.{key}" if prefix else key

        if isinstance(value, dict):
            result.extend(flatten_sweep(value, path, table))
        else:
            if not isinstance(value, list):
                raise ValueError(
                    f"[{table}] parameter '{path}' must be a TOML array."
                )

            if len(value) == 0:
                raise ValueError(
                    f"[{table}] parameter '{path}' has an empty list."
                )

            result.append((path, value))

    return result


sweep_parameters = flatten_sweep(sweep, table="sweep")
zip_parameters = flatten_sweep(sweep_zip, table="sweep_zip")

# One config = one basis: the basis picks the code and the data folder for the
# whole sweep, so it cannot vary between runs.
if any(path == "system.basis" for path, _ in sweep_parameters + zip_parameters):
    raise SystemExit("ERROR: [system] basis cannot be swept; use one config per basis.")
basis = config["system"]["basis"]


def prune_unused_model_tables(cfg, sweep_values):
    """Keep only the [model.<type>] table that [model] type selects.

    A config may carry several model tables (all of them are needed in the
    base config to SWEEP model.type, because set_path only overwrites keys
    that already exist), but each run uses one. Drop the others, and their
    sweep_values, so config.json / manifest.json show only what was run.
    """
    model = cfg.get("model", {})
    kind = model.get("type")
    if kind is None:                     # flat model section (eigenbasis)
        return
    for key in [k for k, v in model.items() if isinstance(v, dict) and k != kind]:
        del model[key]
        for path in [p for p in sweep_values if p.startswith(f"model.{key}.")]:
            del sweep_values[path]

# ----------------------------------------------------------------------
# [sweep_zip.*] parameters are combined column-wise (zipped) rather than
# multiplied, so you can request specific tuples:
#
#   [sweep_zip.optimizer]
#   b1  = [0.8, 0.6]
#   b2  = [0.9, 0.8]
#
# gives exactly two runs -- (0.8, 0.9) and (0.6, 0.8) -- not four. Every
# list in the table must therefore have the same length.
# ----------------------------------------------------------------------

overlap = {p for p, _ in sweep_parameters} & {p for p, _ in zip_parameters}
if overlap:
    raise ValueError(
        "Parameter(s) appear in both [sweep] and [sweep_zip]: "
        + ", ".join(sorted(overlap))
    )

if zip_parameters:
    lengths = {len(values) for _, values in zip_parameters}

    if len(lengths) != 1:
        detail = ", ".join(
            f"{path} ({len(values)})" for path, values in zip_parameters
        )
        raise ValueError(
            "All [sweep_zip] lists must have the same length, got: " + detail
        )

    num_columns = lengths.pop()
    zip_columns = [
        {path: values[i] for path, values in zip_parameters}
        for i in range(num_columns)
    ]
else:
    zip_columns = [{}]

# ----------------------------------------------------------------------
# Set a dotted path in a nested dictionary.
# ----------------------------------------------------------------------

def set_path(obj, path, value):
    parts = path.split(".")
    current = obj

    for part in parts[:-1]:
        if part not in current:
            raise ValueError(
                f"Sweep parameter '{path}' refers to nonexistent "
                f"configuration section '{part}'."
            )

        if not isinstance(current[part], dict):
            raise ValueError(
                f"Cannot descend into '{part}' while setting '{path}'."
            )

        current = current[part]

    final = parts[-1]

    if final not in current:
        raise ValueError(
            f"Sweep parameter '{path}' does not exist in the base configuration."
        )

    current[final] = value


# ----------------------------------------------------------------------
# Generate all combinations.
# ----------------------------------------------------------------------

if not sweep_parameters:
    combinations = [()]
else:
    combinations = list(itertools.product(
        *(values for _, values in sweep_parameters)
    ))

# Zipped tuples are the outer loop, so with [sweep_zip] alone column i maps
# straight onto run_000i. Any [sweep] parameters are multiplied on top.
runs = []

for column in zip_columns:
    for combination in combinations:
        resolved = copy.deepcopy(config)

        sweep_values = {}

        for (path, _), value in zip(sweep_parameters, combination):
            set_path(resolved, path, value)
            sweep_values[path] = value

        for path, value in column.items():
            set_path(resolved, path, value)
            sweep_values[path] = value

        prune_unused_model_tables(resolved, sweep_values)

        runs.append({
            "run_id": len(runs),
            "sweep_values": sweep_values,
            "config": resolved,
        })

matrix = {
    "sweep_id": sweep_id,
    "basis": basis,
    "git_commit": git_commit,
    "source_config": config_path,
    "sweep_parameters": [path for path, _ in sweep_parameters],
    "sweep_zip_parameters": [path for path, _ in zip_parameters],
    "num_runs": len(runs),
    "runs": runs,
}

with open(matrix_path, "w") as f:
    json.dump(matrix, f, indent=2)
    f.write("\n")

print(f"Number of sweep parameters: {len(sweep_parameters)}")
print(f"Number of zipped parameters: {len(zip_parameters)}"
      + (f" ({len(zip_columns)} tuples)" if zip_parameters else ""))
print(f"Number of runs:             {len(runs)}")
PY

# --------------------------------------------------------------------------
# Read number of runs back from matrix.json
# --------------------------------------------------------------------------

NUM_RUNS="$(python - "$MATRIX" <<'PY'
import json
import sys

with open(sys.argv[1]) as f:
    data = json.load(f)

print(data["num_runs"])
PY
)"

if [[ "$NUM_RUNS" -lt 1 ]]; then
    echo "ERROR: Sweep produced zero runs." >&2
    exit 1
fi

# --------------------------------------------------------------------------
# Pre-create every run directory.
#
# This lets us test that the login node can create/write the output tree
# before we submit anything.
# --------------------------------------------------------------------------

for ((i=0; i<NUM_RUNS; i++)); do
    RUN_DIR="$SWEEP_DIR/run_$(printf '%04d' "$i")"

    if ! mkdir -p "$RUN_DIR"; then
        echo "ERROR: Cannot create $RUN_DIR" >&2
        exit 1
    fi

    if [[ ! -w "$RUN_DIR" ]]; then
        echo "ERROR: Run directory is not writable: $RUN_DIR" >&2
        exit 1
    fi
done

# --------------------------------------------------------------------------
# Display the sweep.
# --------------------------------------------------------------------------

echo
echo "============================================================"
echo "SWEEP"
echo "============================================================"
echo "Sweep ID:     $SWEEP_ID"
echo "Git commit:   $GIT_COMMIT"
echo "Config:       $CONFIG"
echo "Runs:         $NUM_RUNS"
echo "Output:       $SWEEP_DIR"
echo

python - "$MATRIX" <<'PY'
import json
import sys

with open(sys.argv[1]) as f:
    data = json.load(f)

parameters = data["sweep_parameters"]
zipped = data.get("sweep_zip_parameters", [])

if not parameters and not zipped:
    print("No parameter sweep: exactly one run.")
else:
    if parameters:
        print("Sweeping (Cartesian product):")
        for parameter in parameters:
            print(f"  - {parameter}")
    if zipped:
        print("Sweeping (zipped tuples):")
        for parameter in zipped:
            print(f"  - {parameter}")

print()

for run in data["runs"]:
    print(f"Run {run['run_id']:04d}:")
    if run["sweep_values"]:
        for key, value in run["sweep_values"].items():
            print(f"    {key} = {value!r}")
    else:
        print("    (base configuration)")
    print()
PY

# --------------------------------------------------------------------------
# Walltime consistency
#
# MAX_RUNTIME_MIN is what stops the run gracefully and writes the checkpoint.
# If it is not safely below the profile's --time, Slurm hard-kills the job
# first and that window's progress is lost. This used to be two numbers kept
# in sync by hand, in two files; now it is checked.
# --------------------------------------------------------------------------

WALL_MIN="$(cluster_walltime_minutes "$CLUSTER_WALLTIME" || true)"
CONFIG_MAX_RUNTIME="$(python - "$MATRIX" <<'PY'
import json, sys
with open(sys.argv[1]) as f:
    matrix = json.load(f)
vals = {r["config"]["run"]["MAX_RUNTIME_MIN"] for r in matrix["runs"]}
print(max(vals))
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
    echo "       Set MAX_RUNTIME_MIN in $CONFIG to $CLUSTER_MAX_RUNTIME_MIN," >&2
    echo "       or raise CLUSTER_WALLTIME in clusters/$CLUSTER_NAME.sh." >&2
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
    echo "         Set MAX_RUNTIME_MIN in $CONFIG to $CLUSTER_MAX_RUNTIME_MIN." >&2
fi

# --------------------------------------------------------------------------
# Confirmation.
# --------------------------------------------------------------------------

if [[ "$NUM_RUNS" -gt 1 ]]; then
    echo "This will submit a Slurm job array with $NUM_RUNS $BASIS-basis tasks ($JOB_NAME) to $CLUSTER_NAME."
else
    echo "This will submit one $BASIS-basis Slurm job ($JOB_NAME) to $CLUSTER_NAME."
fi

echo
read -r -p "Submit this run? [y/N] " ANSWER

case "$ANSWER" in
    y|Y|yes|YES)
        ;;
    *)
        echo "Submission cancelled."
        exit 0
        ;;
esac

# --------------------------------------------------------------------------
# Ask Slurm to validate the submission without actually submitting it.
#
# The actual resource requests live in run.sh.
# --------------------------------------------------------------------------

echo
echo "Running Slurm preflight check..."

if ! sbatch \
    --test-only \
    "${CLUSTER_SBATCH_ARGS[@]}" \
    --job-name="$JOB_NAME" \
    --array="0-$((NUM_RUNS - 1))" \
    run.sh \
    "$MATRIX" \
    "$SWEEP_DIR"
then
    echo
    echo "ERROR: Slurm rejected the test submission."
    echo "No job was submitted."
    exit 1
fi

echo
echo "Slurm preflight passed."

# --------------------------------------------------------------------------
# Submit the actual array.
# --------------------------------------------------------------------------

echo
echo "Submitting job array..."

cp "$CONFIG" "$SWEEP_DIR/config.toml"

JOB_ID="$(
    sbatch \
        --parsable \
        "${CLUSTER_SBATCH_ARGS[@]}" \
        --job-name="$JOB_NAME" \
        --array="0-$((NUM_RUNS - 1))" \
        run.sh \
        "$MATRIX" \
        "$SWEEP_DIR"
)" || { echo "ERROR: sbatch failed; no job was submitted." >&2; exit 1; }
SUBMITTED=1

echo
echo "============================================================"
echo "SUBMITTED"
echo "============================================================"
echo "Slurm job ID: $JOB_ID"
echo "Sweep ID:     $SWEEP_ID"
echo "Basis:        $BASIS"
echo "Runs:         $NUM_RUNS"
echo "Output:       $SWEEP_DIR"
echo
echo "Monitor with:"
echo "  squeue --array -j $JOB_ID"
echo
echo "Cancel with:"
echo "  scancel $JOB_ID"
