#!/bin/bash
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH --job-name=EB2D
#SBATCH --mem=16GB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --array=0
#SBATCH --output=z_save_data_EB_2D/logs/EB2D_%A_%a.out
#SBATCH --error=z_save_data_EB_2D/logs/EB2D_%A_%a.err

# 2D eigenbasis (2x2 plaquette) J1-J2 VMC with the multilayer ViT.
# Auto-resubmits itself with the SAME command until python writes a .done
# sentinel; resume is checkpoint-driven (--continue_run auto). Modeled on
# z_mainbash.sh.
#
# NOTE on provenance: the run *identity* is in the path/filename (physics +
# optimization + seed + Ns + alpha + n_discard). The FULL parameter set,
# including chunk_size and on_the_fly, is written to <fname>.config.json at
# startup — those two are performance/memory knobs and are intentionally kept
# OUT of the filename so that, e.g., lowering CHUNK_SIZE after an OOM does not
# fork the checkpoint path and lose your resume. Flip TAG_CHUNK_SIZE /
# TAG_ON_THE_FLY at the top of the .py if you disagree.

# STALE: this is the pre-sweep Explorer-era launcher. run_2D_eigenbasis.py no
# longer parses command-line flags at all -- it takes a matrix.json and an
# output directory (sys.argv[1], sys.argv[2]) -- so the invocation below will
# not run as written. Kept for reference; use ./submit.sh config.toml instead.

# ---- absolute path to THIS script (for self-resubmission) ----
THIS_SCRIPT="$HOME/master/2d_eigenbasis/run_2D_eigenbasis.sh"


module purge
cd "$HOME/master/2d_eigenbasis"
source .venv/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9   # let JAX grow into the H200
mkdir -p z_save_data_EB_2D/logs

# -------- run parameters (edit these) --------
L=6                    # LxL physical lattice; 4 -> the ED cross-check case (E0=-8.45792335)
J2=0.0
J1=1.0
sign_rule="TF"         # Marshall rule per coupling [J1,J2]: FF/TF/FT/TT. Physics
                       # identity -> tagged in the filename (sr=..). Flip to test
                       # the Marshall rule in the cluster eigenbasis.
N_ITER=50            # TOTAL target steps across all resubmissions

# SAMPLER: exact (ExactSampler+ED) | metro (MetropolisHamiltonian) | auto
SAMPLER=metro
SAVE_SUBDIR="Aug31_alex_test"     # nests output under z_save_data_EB_2D/<SAVE_SUBDIR>/...

N_SAMPLES=8192         # MC samples per step, stated directly (was 4 * 2048)
seed=0
n_chains=16
n_discard_per_chain=100   # metro burn-in per chain; -1 = NetKet default ("auto").
                          # Tagged into the filename (it changes the sampling).

# overdispersed VMC_NG driver
alpha=2.0              # overdispersed sampling exponent (tracked in savename + logs)
use_ntk=true
on_the_fly=false        # pinned false (srt_onthefly is_jac bug); recorded in config.json

num_layers=2
r=3                    # d_model = r * n_heads
n_heads=20
attn_mode="f"
transl_invariant=true

# LINEAR schedules (annealed lr1->lr2 and diag_shift1->diag_shift2)
lr1=5e-2
lr2=1e-3
diag_shift1=1e-3
diag_shift2=1e-4
lr_transition_steps=1200
ds_transition_steps=1200

# leave ~10 min margin below the 4h wall so the final checkpoint flushes
MAX_RUNTIME_MIN=230
SAVE_EVERY=50
PRINT_EVERY=20
DIAG_EVERY=-1            # sampling-health diag (ESS/R-hat/acc) -> .diag.jsonl.
                        # -1 = follow PRINT_EVERY; 0 = off.
CHUNK_SIZE=512           # 0 = unset; recorded in config.json, NOT in the filename
# ---------------------------------------------

echo "Job: L=${L}x${L}  J2=$J2  target $N_ITER steps  sampler=$SAMPLER  alpha=$alpha  save_subdir='$SAVE_SUBDIR'"
echo "n_discard_per_chain=$n_discard_per_chain  chunk_size=$CHUNK_SIZE  on_the_fly=$on_the_fly"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

time python3 -u -Walways run_2D_eigenbasis.py \
  --L $L --J1 $J1 --J2 $J2 --sign_rule $sign_rule --N_ITER $N_ITER \
  --N_SAMPLES $N_SAMPLES --seed $seed --n_chains $n_chains \
  --n_discard_per_chain $n_discard_per_chain \
  --alpha $alpha --use_ntk $use_ntk --on_the_fly $on_the_fly \
  --num_layers $num_layers --r $r --n_heads $n_heads \
  --attn_mode $attn_mode --transl_invariant $transl_invariant \
  --lr1 $lr1 --lr2 $lr2 --diag_shift1 $diag_shift1 --diag_shift2 $diag_shift2 \
  --lr_transition_steps $lr_transition_steps --ds_transition_steps $ds_transition_steps \
  --sampler $SAMPLER --continue_run auto \
  --max_runtime_min $MAX_RUNTIME_MIN --save_every $SAVE_EVERY --print_every $PRINT_EVERY \
  --diag_every $DIAG_EVERY \
  --chunk_size $CHUNK_SIZE --save_subdir "$SAVE_SUBDIR"
PYEXIT=$?     # capture immediately

# -------- auto-resubmit ONLY on a clean time-stop --------
# Resubmit iff: python exited 0 AND wrote a .timeout sentinel AND not .done.
RUNINFO=".runinfo_${SLURM_JOB_ID}"
if [ -f "$RUNINFO" ]; then
    FNAME=$(cat "$RUNINFO")
    rm -f "$RUNINFO"
    if [ -f "${FNAME}.done" ]; then
        echo "Target reached — no resubmission."
    elif [ "$PYEXIT" -eq 0 ] && [ -f "${FNAME}.timeout" ]; then
        echo "Clean wall-clock stop — resubmitting to continue from checkpoint."
        sbatch "$THIS_SCRIPT"
    else
        echo "Python exited $PYEXIT with no clean time-stop — NOT resubmitting (check the .err)."
    fi
else
    echo "WARNING: no runinfo file (python likely crashed before setup) — NOT resubmitting."
fi
