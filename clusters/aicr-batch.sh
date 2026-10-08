# ==========================================================================
# clusters/aicr-batch.sh -- AICR, b200-batch partition (24 h window)
#
# The production counterpart to aicr.sh. Select it per submission:
#
#     CLUSTER=aicr-batch ./submit.sh config_<basis>basis.toml
#     CLUSTER=aicr-batch ./submit_branch.sh --from <dir> --run N --extra M
#
# and set config_<basis>basis.toml's MAX_RUNTIME_MIN to 1380 to match (or, for a branch,
# pass --set run.MAX_RUNTIME_MIN=1380). submit.sh warns if you forget: a 230
# min budget on a 24 h window holds the allocation for 24 h and uses 4 h of it.
#
# WHY A SEPARATE PROFILE RATHER THAN A QUEUE FLAG
#
# submit.sh writes $CLUSTER_NAME to <sweep>/cluster.txt, and run.sh reloads
# that profile on every self-resubmission. Because this file sets a DIFFERENT
# CLUSTER_NAME, the partition choice is persisted with the sweep and every
# continuation window lands on b200-batch too. A QUEUE=batch environment
# variable read inside aicr.sh would apply to the first window only and then
# silently fall back to devel, with the walltime dropping from 24 h to 4 h
# under a run still budgeted for 1380 min -- a hard kill with no .timeout
# sentinel, which also stops the resubmission chain.
#
# This profile is deliberately absent from CLUSTER_PROFILES in common.sh:
# autodetection must keep resolving an AICR login node to plain "aicr", and
# both profiles match the same b200- partition fingerprint. It is listed by
# cluster_list via CLUSTER_VARIANTS instead.
# ==========================================================================

# Inherit the account, memory, cores, module and detection regexes, then
# override the name, description and queue. CLUSTER_DIR is set by common.sh,
# which is always sourced before any profile.
source "$CLUSTER_DIR/aicr.sh"

CLUSTER_NAME="aicr-batch"
CLUSTER_DESCRIPTION="AICR (docs.aicr.ai) -- EPYC 9745 / 1.125 TB, 8x B200 per node; b200-batch, 24 h"

aicr_profile b200-batch 24:00:00 1380
