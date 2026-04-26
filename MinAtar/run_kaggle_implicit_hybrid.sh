#!/bin/bash
# Full FAME sweep on GPU for Kaggle Kernels execution.
# Runs implicit and hybrid detector variants sequentially.
#
# NOTE: Separate single-detector scripts also exist:
#   run_kaggle_implicit.sh  -- implicit only (with checkpointing)
#   run_kaggle_hybrid.sh    -- hybrid only   (with checkpointing)
#
# Use those when running each detector in its own Kaggle session so that
# checkpointing can resume a preempted run independently.

set -euo pipefail

cd "$(dirname "$0")"

PY="${PY:-python}"
SEQ="${SEQ:-0}"
SEEDS=(${SEEDS:-1})
GPU="${GPU:-0}"
TSTEPS="${TSTEPS:-3500000}"   # full protocol: 7 tasks x 500k
SWITCH="${SWITCH:-500000}"

# Directories for Kaggle output persistence.
RESULTS_DIR="/kaggle/working/results"
MODELS_DIR="/kaggle/working/models"
LOGS_DIR="/kaggle/working/logs"
CHECKPOINT_DIR="/kaggle/working/checkpoints"

mkdir -p "$RESULTS_DIR" "$MODELS_DIR" "$LOGS_DIR" "$CHECKPOINT_DIR"

for s in "${SEEDS[@]}"; do
    # ----------------------------------------------------------------
    # --- implicit detector ---
    # ----------------------------------------------------------------
    echo "========================================"
    echo "Running implicit detector | seq=${SEQ} seed=${s}"
    echo "Started at $(date)"
    echo "========================================"

    "$PY" FAME.py \
        --detector implicit \
        --swoks_L_D 1200 --swoks_L_W 30 \
        --swoks_alpha 1e-3 --swoks_beta 2.0 \
        --swoks_stable_phase 36000 --swoks_interval 240 \
        --swoks_warmup 5000 \
        --swoks_snapshot 1 --swoks_snapshot_interval 6000 \
        --imp_L_D 1200 --imp_alpha 1e-3 \
        --imp_stable_phase 36000 --imp_interval 240 \
        --imp_warmup 5000 --imp_lr 1e-4 \
        --imp_update_every 16 \
        --hyb_tau_imp_loose 1e-2 \
        --hyb_tau_imp_strict 1e-6 \
        --hyb_tau_stat_strict 1e-3 \
        --hyb_tau_combined 15.0 \
        --hyb_horizon 480 \
        --hyb_persistence 2 \
        --lr1 1e-3 --lr2 1e-5 \
        --size_fast2meta 12000 \
        --detection_step 600 \
        --warmstep 50000 \
        --lambda_reg 1.0 \
        --t-steps "$TSTEPS" --switch "$SWITCH" \
        --seq "$SEQ" --seed "$s" --gpu "$GPU" \
        --save --save-model \
        --results_dir "$RESULTS_DIR" \
        --models_dir "$MODELS_DIR" \
        --checkpoint_dir "$CHECKPOINT_DIR" \
        --checkpoint_interval 100000 \
        2>&1 | tee "$LOGS_DIR/implicit_seq${SEQ}_seed${s}.log"

    echo "Finished implicit | seq=${SEQ} seed=${s} at $(date)"

    # ----------------------------------------------------------------
    # --- hybrid detector ---
    # ----------------------------------------------------------------
    echo "========================================"
    echo "Running hybrid detector | seq=${SEQ} seed=${s}"
    echo "Started at $(date)"
    echo "========================================"

    "$PY" FAME.py \
        --detector hybrid \
        --swoks_L_D 1200 --swoks_L_W 30 \
        --swoks_alpha 1e-3 --swoks_beta 2.0 \
        --swoks_stable_phase 36000 --swoks_interval 240 \
        --swoks_warmup 5000 \
        --swoks_snapshot 1 --swoks_snapshot_interval 6000 \
        --imp_L_D 1200 --imp_alpha 1e-3 \
        --imp_stable_phase 36000 --imp_interval 240 \
        --imp_warmup 5000 --imp_lr 1e-4 \
        --imp_update_every 16 \
        --hyb_tau_imp_loose 1e-2 \
        --hyb_tau_imp_strict 1e-6 \
        --hyb_tau_stat_strict 1e-3 \
        --hyb_tau_combined 15.0 \
        --hyb_horizon 480 \
        --hyb_persistence 2 \
        --lr1 1e-3 --lr2 1e-5 \
        --size_fast2meta 12000 \
        --detection_step 600 \
        --warmstep 50000 \
        --lambda_reg 1.0 \
        --t-steps "$TSTEPS" --switch "$SWITCH" \
        --seq "$SEQ" --seed "$s" --gpu "$GPU" \
        --save --save-model \
        --results_dir "$RESULTS_DIR" \
        --models_dir "$MODELS_DIR" \
        --checkpoint_dir "$CHECKPOINT_DIR" \
        --checkpoint_interval 50000 \
        2>&1 | tee "$LOGS_DIR/hybrid_seq${SEQ}_seed${s}.log"

    echo "Finished hybrid | seq=${SEQ} seed=${s} at $(date)"
done

echo ""
echo "========================================"
echo "All training runs finished at $(date)"
echo "========================================"

"$PY" compare_oracle_vs_swoks.py \
    --results_dir "$RESULTS_DIR" \
    --seq "$SEQ" \
    --seeds "${SEEDS[@]}" \
    --tolerance 60000

