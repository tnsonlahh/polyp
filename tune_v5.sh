#!/bin/bash
# V5 Hyperparameter Tuning — Run fold 1 only for quick comparison
# Key insight: rampup=200 means ICT was basically OFF in the original run
#
# Priority tuning axes (most impactful first):
#   1. consistency_rampup: 200→40 (let consistency kick in by epoch 10)
#   2. consistency: 0.1→0.3 (stronger signal once ramped)
#   3. ict_alpha: 1.0→0.4 (less random mixing, more extreme)
#   4. ict_weight: 1.0 vs 0.5
#
# Usage: bash tune_v5.sh <gpu_id>

GPU=${1:-0}
BASE="python mean_teacher_v5.py --config_yml Configs/kvasir_seg.yml --gpu $GPU --seed 1 --folds 1"

echo "===== V5 Tuning Sweep (fold 1 only) ====="
echo ""

# Exp A: Fix rampup only (strongest single change)
echo "[A] rampup=40, rest default"
$BASE --exp mt_v5_tuneA --consistency 0.1 --consistency_rampup 40.0 --ict_alpha 1.0 --ict_weight 1.0
echo ""

# Exp B: Fix rampup + stronger consistency
echo "[B] rampup=40, consistency=0.3"
$BASE --exp mt_v5_tuneB --consistency 0.3 --consistency_rampup 40.0 --ict_alpha 1.0 --ict_weight 1.0
echo ""

# Exp C: Fix rampup + better mixing
echo "[C] rampup=40, consistency=0.3, ict_alpha=0.4"
$BASE --exp mt_v5_tuneC --consistency 0.3 --consistency_rampup 40.0 --ict_alpha 0.4 --ict_weight 1.0
echo ""

# Exp D: Fix rampup + stronger consistency + lower ICT weight
echo "[D] rampup=40, consistency=0.3, ict_alpha=0.4, ict_weight=0.5"
$BASE --exp mt_v5_tuneD --consistency 0.3 --consistency_rampup 40.0 --ict_alpha 0.4 --ict_weight 0.5
echo ""

# Exp E: Very fast rampup
echo "[E] rampup=20, consistency=0.3, ict_alpha=0.4"
$BASE --exp mt_v5_tuneE --consistency 0.3 --consistency_rampup 20.0 --ict_alpha 0.4 --ict_weight 1.0
echo ""

echo "===== Tuning complete! Check results: ====="
for exp in tuneA tuneB tuneC tuneD tuneE; do
    echo "--- mt_v5_$exp ---"
    cat checkpoints/kvasir/mt_v5_$exp/fold1/test_results.txt 2>/dev/null || echo "  (not found)"
    echo ""
done
