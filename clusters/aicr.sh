# ==========================================================================
# clusters/aicr.sh -- AICR (docs.aicr.ai), b200-devel partition
#
# Nodes are 128-core EPYC 9745 / 1.125 TB RAM with 8 GPUs, so 16 cores +
# 64 GB is roughly the per-GPU share. AICR's default memory is only 1 GB per
# CPU, so --mem is mandatory. GPUs are requested with --gpus; --gres is not
# used here, and the partition (not a type string) selects the hardware.
#
# THIS PROFILE IS THE DEVEL PARTITION: 4 h cap, max 4 concurrent jobs per
# user. For the 24 h production partition use the sibling profile instead of
# editing this file:
#
#     CLUSTER=aicr-batch ./submit.sh config_<basis>basis.toml
#     CLUSTER=aicr-batch ./submit_branch.sh --from <dir> --run N --extra M
#
# The partition is chosen PER SUBMISSION that way, and it sticks: submit.sh
# records $CLUSTER_NAME in <sweep>/cluster.txt, and run.sh loads that same
# profile on every self-resubmission, so a sweep never drifts between
# partitions mid-flight. An environment variable read inside this file could
# not do that -- only the profile NAME is persisted.
#
# Both profiles are built by aicr_profile below, so the account, memory, core
# count and module live in exactly one place.
#
# MAX_RUNTIME_MIN in config_<basis>basis.toml is the other half of the walltime pairing: it
# is what stops the run gracefully and writes the checkpoint, so if it exceeds
# --time then Slurm hard-kills the job first. submit.sh enforces that, and
# also warns when it is far BELOW the walltime (the "submitted a 24 h job that
# stops itself after 4 h" case).
# ==========================================================================

CLUSTER_NAME="aicr"
CLUSTER_DESCRIPTION="AICR (docs.aicr.ai) -- EPYC 9745 / 1.125 TB, 8x B200 per node; b200-devel, 4 h"

# Fingerprints used by cluster_autodetect in common.sh. Only this profile
# carries them: aicr-batch.sh is an explicit choice, never autodetected, so
# the two cannot both match the same login node and make detection ambiguous.
CLUSTER_DETECT_PARTITION_RE="(^| )b200-"
CLUSTER_DETECT_HOSTNAME_RE="aicr"

# Module providing a python new enough to run submit.sh's tomllib parsing.
CLUSTER_PYTHON_MODULE="miniforge3"

# --------------------------------------------------------------------------
# aicr_profile <partition> <walltime> <max_runtime_min>
#
# Everything that differs between the AICR partitions, in one function, so
# aicr-batch.sh can source this file and re-invoke it rather than duplicating
# the account/memory/cores (which would then drift).
#
# CLUSTER_SBATCH_ARGS must be rebuilt rather than patched: the array literal
# expands "$CLUSTER_WALLTIME" when it is constructed, so reassigning the
# walltime afterwards would leave a stale --time in the array.
# --------------------------------------------------------------------------
aicr_profile() {
    local partition="$1" walltime="$2" max_runtime="$3"

    # One job window. Keep CLUSTER_MAX_RUNTIME_MIN ~10 min under the walltime
    # so the final checkpoint flushes before Slurm's hard kill.
    CLUSTER_WALLTIME="$walltime"
    CLUSTER_MAX_RUNTIME_MIN="$max_runtime"

    CLUSTER_SBATCH_ARGS=(
        --partition="$partition"
        --account=p2026_0109_neu
        --gpus=1
        --mem=64G
        --cpus-per-task=16
        --time="$CLUSTER_WALLTIME"
    )
}

aicr_profile b200-devel 04:00:00 230
