#!/bin/bash
# train_all.sh
# Train fixmatch, bcp, cps, ccvc, gta with 10% label data and early stop

CONFIG="Configs/kvasir_seg.yml"
GPU="0"

echo "Starting sequential training with 10% labeled data and early stopping..."
echo "Configuration used: $CONFIG"

echo "====================================="
echo "1. Training FixMatch..."
python3 fixmatch.py --config_yml $CONFIG --gpu $GPU --exp exp_fixmatch_10pct

echo "====================================="
echo "2. Training BCP..."
python3 BCP.py --config_yml $CONFIG --gpu $GPU --exp exp_bcp_10pct

echo "====================================="
echo "3. Training CPS..."
python3 cps.py --config_yml $CONFIG --gpu $GPU --exp exp_cps_10pct

echo "====================================="
echo "4. Training CCVC..."
python3 ccvc.py --config_yml $CONFIG --gpu $GPU --exp exp_ccvc_10pct

echo "====================================="
echo "5. Training GTA..."
python3 GTA_seg.py --config_yml $CONFIG --gpu $GPU --exp exp_gta_10pct

echo "====================================="
echo "All training processes have been completed successfully!"
