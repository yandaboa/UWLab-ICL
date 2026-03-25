#!/bin/bash
set -euo pipefail

# Forward dynamics sweep knobs.
SEEDS=(1 2)
HIDDEN_DIMS=(256 512 1024)
NUM_LAYERS=(4 6)
BATCH_SIZES=(512)
LEARNING_RATES=(3e-4 1e-4)
WEIGHT_DECAYS=(1e-4 1e-5)

SBATCH_FILE="/gscratch/weirdlab/yanda/lti/UWLab-yanda/train_forward_dynamics.sbatch"
mkdir -p logs

for SEED in "${SEEDS[@]}"; do
  for HIDDEN_DIM in "${HIDDEN_DIMS[@]}"; do
    for NUM_LAYER in "${NUM_LAYERS[@]}"; do
      for BATCH_SIZE in "${BATCH_SIZES[@]}"; do
        for LEARNING_RATE in "${LEARNING_RATES[@]}"; do
          for WEIGHT_DECAY in "${WEIGHT_DECAYS[@]}"; do
            RUN_NAME="fd_s${SEED}_h${HIDDEN_DIM}_L${NUM_LAYER}_b${BATCH_SIZE}_lr${LEARNING_RATE}_wd${WEIGHT_DECAY}"
            sbatch --job-name="$RUN_NAME" "$SBATCH_FILE" \
              "$SEED" "$HIDDEN_DIM" "$NUM_LAYER" "$BATCH_SIZE" "$LEARNING_RATE" "$WEIGHT_DECAY" "$RUN_NAME"
            echo "submitted $RUN_NAME"
          done
        done
      done
    done
  done
done
