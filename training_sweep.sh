#!/bin/bash
set -euo pipefail

# SEEDS=(1)
# HIDDEN_DIMS=(128)
# NUM_LAYERS=(4)
# NUM_HEADS=(4)
SEEDS=(1 2 3)
HIDDEN_DIMS=(128 256 512)
NUM_LAYERS=(4 6 8)
NUM_HEADS=(4 8 12)

SBATCH_FILE="/gscratch/weirdlab/yanda/lti/UWLab-yanda/train.sbatch"
mkdir -p logs

for SEED in "${SEEDS[@]}"; do
  for HIDDEN_DIM in "${HIDDEN_DIMS[@]}"; do
    for NUM_LAYER in "${NUM_LAYERS[@]}"; do
      for NUM_HEAD in "${NUM_HEADS[@]}"; do
        RUN_NAME="seed${SEED}_hd${HIDDEN_DIM}_L${NUM_LAYER}_H${NUM_HEAD}"
        # Keep job name <= ~128 chars (Slurm limit varies)
        sbatch --job-name="$RUN_NAME" "$SBATCH_FILE" \
          "$SEED" "$HIDDEN_DIM" "$NUM_LAYER" "$NUM_HEAD" "$RUN_NAME"
        echo "submitted $RUN_NAME"
      done
    done
  done
done
