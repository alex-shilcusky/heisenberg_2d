# ==========================================================================
# clusters/common.sh
#
# Cluster profile loader. Sourced by submit.sh, run.sh and debug.sh:
#
#     source "$REPO_ROOT/clusters/common.sh"
#     cluster_load            # or: cluster_load "$EXPLICIT_NAME"
#
# After cluster_load returns, these are set:
#
#     CLUSTER_NAME            "aicr" | "explorer"
#     CLUSTER_DESCRIPTION     one-line human description
#     CLUSTER_PYTHON_MODULE   module that provides a modern python
#     CLUSTER_WALLTIME        --time value for a single job window
#     CLUSTER_MAX_RUNTIME_MIN recommended MAX_RUNTIME_MIN for that walltime
#     CLUSTER_SBATCH_ARGS     bash array of sbatch resource flags
#
# WHY THE RESOURCE REQUESTS ARE NOT #SBATCH LINES
#
# #SBATCH directives are inert comments: they cannot branch on a cluster.
# Slurm gives command-line options precedence over in-script directives, so
# the portable resources stay as #SBATCH in run.sh and everything that
# differs between machines is passed by submit.sh (and by run.sh when it
# resubmits itself) from CLUSTER_SBATCH_ARGS.
# ==========================================================================

# Directory holding this file, so the profiles can be found regardless of cwd.
CLUSTER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Profiles considered by autodetection, in order.
CLUSTER_PROFILES=(aicr explorer)

# Profiles that are only ever selected EXPLICITLY (CLUSTER=<name>), never
# autodetected. A partition variant of a cluster shares that cluster's
# fingerprints, so letting autodetection consider it would make detection
# ambiguous -- but it still needs to be discoverable, so cluster_list shows it.
CLUSTER_VARIANTS=(aicr-batch)

# --------------------------------------------------------------------------
# cluster_list -- print the available profiles.
# --------------------------------------------------------------------------
cluster_list() {
    local p
    for p in "${CLUSTER_PROFILES[@]}" "${CLUSTER_VARIANTS[@]}"; do
        [[ -f "$CLUSTER_DIR/$p.sh" ]] || continue
        local desc
        desc="$(sed -n 's/^CLUSTER_DESCRIPTION="\(.*\)"$/\1/p' "$CLUSTER_DIR/$p.sh" | head -1)"
        printf '    %-12s %s\n' "$p" "$desc"
    done
}

# --------------------------------------------------------------------------
# cluster_autodetect -- echo the detected profile name, or nothing.
#
# Detection is by Slurm partition fingerprint first, because a partition name
# is a definitional property of the cluster and is cheap to query. Hostname
# patterns are only a fallback: adjust CLUSTER_DETECT_HOSTNAME_RE in a profile
# if your login node does not match.
# --------------------------------------------------------------------------
cluster_autodetect() {
    local partitions="" host p
    # The `|| true` matters: callers run under `set -e -o pipefail`, and a
    # failing sinfo would otherwise abort the function before the hostname
    # fallback below ever gets a chance.
    if command -v sinfo >/dev/null 2>&1; then
        partitions="$(sinfo -h -o '%P' 2>/dev/null | tr -d '*' | tr '\n' ' ' || true)"
    fi
    host="$(hostname -f 2>/dev/null || hostname 2>/dev/null || echo '')"

    # Pass 1: partition fingerprint.
    for p in "${CLUSTER_PROFILES[@]}"; do
        [[ -f "$CLUSTER_DIR/$p.sh" ]] || continue
        local re
        re="$(sed -n 's/^CLUSTER_DETECT_PARTITION_RE="\(.*\)"$/\1/p' "$CLUSTER_DIR/$p.sh" | head -1)"
        if [[ -n "$re" && -n "$partitions" ]] && grep -qE "$re" <<<"$partitions"; then
            echo "$p"
            return 0
        fi
    done

    # Pass 2: hostname.
    for p in "${CLUSTER_PROFILES[@]}"; do
        [[ -f "$CLUSTER_DIR/$p.sh" ]] || continue
        local re
        re="$(sed -n 's/^CLUSTER_DETECT_HOSTNAME_RE="\(.*\)"$/\1/p' "$CLUSTER_DIR/$p.sh" | head -1)"
        if [[ -n "$re" && -n "$host" ]] && grep -qE "$re" <<<"$host"; then
            echo "$p"
            return 0
        fi
    done

    return 1
}

# --------------------------------------------------------------------------
# cluster_load [NAME] -- resolve and source a profile.
#
# Resolution order:
#   1. NAME argument          (submit.sh --cluster / recorded sweep profile)
#   2. $CLUSTER               (env override:  CLUSTER=explorer ./submit.sh ...)
#   3. autodetection
#
# Exits non-zero with a clear message if nothing resolves, rather than
# silently submitting with the wrong partition.
# --------------------------------------------------------------------------
cluster_load() {
    local requested="${1:-${CLUSTER:-}}"
    local source_desc

    if [[ -n "$requested" ]]; then
        source_desc="explicitly selected"
    else
        requested="$(cluster_autodetect || true)"
        source_desc="autodetected"
    fi

    if [[ -z "$requested" ]]; then
        echo "ERROR: Could not determine which cluster this is." >&2
        echo "       Select one explicitly, e.g.:" >&2
        echo "           CLUSTER=aicr ./submit.sh config_<basis>basis.toml" >&2
        echo "       Available profiles:" >&2
        cluster_list >&2
        return 1
    fi

    # Accept the cluster's former name. Northeastern renamed Discovery to
    # Explorer; older commits and README sections still say "Discovery".
    case "$requested" in
        discovery) requested="explorer" ;;
    esac

    local profile="$CLUSTER_DIR/$requested.sh"
    if [[ ! -f "$profile" ]]; then
        echo "ERROR: Unknown cluster profile: '$requested'" >&2
        echo "       Available profiles:" >&2
        cluster_list >&2
        return 1
    fi

    # shellcheck source=/dev/null
    source "$profile"

    if [[ "${CLUSTER_NAME:-}" != "$requested" ]]; then
        echo "ERROR: $profile sets CLUSTER_NAME='${CLUSTER_NAME:-}'," >&2
        echo "       but was loaded as '$requested'." >&2
        return 1
    fi

    CLUSTER_SOURCE="$source_desc"
    export CLUSTER="$CLUSTER_NAME"
    return 0
}

# --------------------------------------------------------------------------
# cluster_walltime_minutes -- convert a Slurm --time value to whole minutes.
#
# Accepts the forms this project actually uses: HH:MM:SS, D-HH:MM:SS, MM.
# Echoes the minutes, or nothing if the format is not recognised.
# --------------------------------------------------------------------------
cluster_walltime_minutes() {
    local t="$1" days=0

    if [[ "$t" == *-* ]]; then
        days="${t%%-*}"
        t="${t#*-}"
    fi

    local h=0 m=0 s=0
    case "$t" in
        *:*:*) IFS=: read -r h m s <<<"$t" ;;
        *:*)   IFS=: read -r m s <<<"$t" ;;
        *)     m="$t" ;;
    esac

    [[ "$days" =~ ^[0-9]+$ && "$h" =~ ^[0-9]+$ && "$m" =~ ^[0-9]+$ ]] || return 1

    echo $(( 10#$days * 1440 + 10#$h * 60 + 10#$m ))
}

# --------------------------------------------------------------------------
# cluster_summary -- print the resolved profile, for submit-time logging.
# --------------------------------------------------------------------------
cluster_summary() {
    echo "Cluster:        $CLUSTER_NAME (${CLUSTER_SOURCE:-resolved})"
    echo "                $CLUSTER_DESCRIPTION"
    echo "Resources:      ${CLUSTER_SBATCH_ARGS[*]}"
}
