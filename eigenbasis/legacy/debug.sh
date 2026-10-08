#!/bin/bash
set -euo pipefail

# ==========================================================================
# submit.sh
#
# Run this on the HPC login node:
#
#     ./submit.sh config.toml
#
# It validates the repository/configuration, expands the parameter sweep,
# asks for confirmation, tests the Slurm submission, and then submits
# one Slurm array task per parameter combination.
# ==========================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RID=$(($(ls -l debugging | grep -c ^d)+1))

CONFIG="config.toml"

RUN_ROOT="debugging"

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

# Which python module exists is cluster-specific, so resolve the profile
# before loading it. Override with: CLUSTER=explorer ./debug.sh
source "$SCRIPT_DIR/clusters/common.sh"
cluster_load

echo "Cluster:        $CLUSTER_NAME (${CLUSTER_SOURCE})"

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
SWEEP_DIR="$RUN_ROOT"
MATRIX="$SWEEP_DIR/matrix.json"

mkdir -p "$SWEEP_DIR"

# run.sh reads the profile back from here, exactly as it does for a real
# sweep, so a debug run exercises the same cluster configuration.
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

# ----------------------------------------------------------------------
# Flatten sweep tables into:
#
#   ("physics.viscosity", [0.001, 0.005, 0.01])
#
# ----------------------------------------------------------------------

def flatten_sweep(obj, prefix=""):
    result = []

    for key, value in obj.items():
        path = f"{prefix}.{key}" if prefix else key

        if isinstance(value, dict):
            result.extend(flatten_sweep(value, path))
        else:
            if not isinstance(value, list):
                raise ValueError(
                    f"Sweep parameter '{path}' must be a TOML array."
                )

            if len(value) == 0:
                raise ValueError(
                    f"Sweep parameter '{path}' has an empty list."
                )

            result.append((path, value))

    return result


sweep_parameters = flatten_sweep(sweep)

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
    combinations = itertools.product(
        *(values for _, values in sweep_parameters)
    )

runs = []

for run_id, combination in enumerate(combinations):
    resolved = copy.deepcopy(config)

    sweep_values = {}

    for (path, _), value in zip(sweep_parameters, combination):
        set_path(resolved, path, value)
        sweep_values[path] = value

    runs.append({
        "run_id": run_id,
        "sweep_values": sweep_values,
        "config": resolved,
    })

matrix = {
    "sweep_id": sweep_id,
    "git_commit": git_commit,
    "source_config": config_path,
    "sweep_parameters": [path for path, _ in sweep_parameters],
    "num_runs": len(runs),
    "runs": runs,
}

with open(matrix_path, "w") as f:
    json.dump(matrix, f, indent=2)
    f.write("\n")

print(f"Number of sweep parameters: {len(sweep_parameters)}")
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


RUN_DIR="$SWEEP_DIR/run_0000"

if ! mkdir -p "$RUN_DIR"; then
    echo "ERROR: Cannot create $RUN_DIR" >&2
    exit 1
fi

if [[ ! -w "$RUN_DIR" ]]; then
    echo "ERROR: Run directory is not writable: $RUN_DIR" >&2
    exit 1
fi


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

if not parameters:
    print("No parameter sweep: exactly one run.")
else:
    print("Sweeping:")
    for parameter in parameters:
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
# Submit the actual array.
# --------------------------------------------------------------------------

echo
echo "Starting job"
echo "$RUN_SCRIPT"

cp "$CONFIG" "$SWEEP_DIR/config.toml"

NEWDIR="$SWEEP_DIR/run_$(printf '%04d' "$RID")"

trap 'mv "$SWEEP_DIR/run_0000" "$NEWDIR"' EXIT

SLURM_ARRAY_TASK_ID=0 /bin/bash "$RUN_SCRIPT" "$MATRIX" "$SWEEP_DIR"

