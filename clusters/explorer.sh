# ==========================================================================
# clusters/explorer.sh -- Northeastern Explorer
#
# The pre-AICR configuration. Verified against every commit from 2026-08-27
# (dd9d57c) through 2026-09-17 (86bfd8f): these five values never changed over
# that whole period, so this is the configuration that actually ran, not a
# reconstruction. GPUs are requested with --gres=gpu:<type>:<n> and the generic
# `gpu` partition carries the GPU nodes, which is the structural difference
# from AICR's --gpus plus a hardware-specific partition.
#
# To request a different accelerator, change the type in the --gres string
# (for example gpu:a100:1 or gpu:v100-sxm2:1).
#
# H200 vs B200: the H200 here is NOT the slower option for this workload. At
# matched config the run history has H200 at 0.86-0.89x the B200 step time
# (2 layers d=72: 3.59 s vs 4.19 s; 8 layers d=72: 11.48 s vs 12.89 s). The
# model is small and float64, so it is launch-bound rather than FLOP-bound and
# Blackwell's extra throughput buys nothing. Step-time model fitted on the
# Explorer runs only:
#
#     s/step = 0.96 + 1.32 * layers      (at N_SAMPLES = 8192)
#
# Width is nearly free; depth is what costs wall-clock.
# ==========================================================================

CLUSTER_NAME="explorer"
CLUSTER_DESCRIPTION="Northeastern Explorer -- gpu partition, H200 via --gres"

# Fingerprints used by cluster_autodetect in common.sh.
# Adjust CLUSTER_DETECT_HOSTNAME_RE if your login node does not match.
CLUSTER_DETECT_PARTITION_RE="(^| )(short|express)( |$)"
CLUSTER_DETECT_HOSTNAME_RE="(explorer|discovery|northeastern|neu\.edu)"

# Module providing a python new enough to run submit.sh's tomllib parsing.
CLUSTER_PYTHON_MODULE="python/3.13.5"

# One job window. Keep CLUSTER_MAX_RUNTIME_MIN ~10 min under the walltime so
# the final checkpoint flushes before Slurm's hard kill.
CLUSTER_WALLTIME="04:00:00"
CLUSTER_MAX_RUNTIME_MIN=230

CLUSTER_SBATCH_ARGS=(
    --partition=gpu
    --gres=gpu:h200:1
    --mem=16GB
    --cpus-per-task=4
    --time="$CLUSTER_WALLTIME"
)
