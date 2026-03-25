#!/bin/bash
#SBATCH --job-name=fd-train
#SBATCH --account=cse
#SBATCH --partition=gpu-l40s
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --chdir=/gscratch/weirdlab/yanda/lti/UWLab-yanda
#SBATCH --export=ALL
#SBATCH --exclude=g3118,g3092

SEED="$1"
HIDDEN_DIM="$2"
NUM_LAYERS="$3"
BATCH_SIZE="$4"
LEARNING_RATE="$5"
WEIGHT_DECAY="$6"
RUN_NAME="$7"

set -e
set -o pipefail
mkdir -p logs

source /gscratch/weirdlab/yanda/miniforge3/etc/profile.d/conda.sh
conda activate uwlab

cd /gscratch/weirdlab/yanda/lti/UWLab-yanda

echo "RUN_NAME=$RUN_NAME"
echo "seed=$SEED hidden_dim=$HIDDEN_DIM num_layers=$NUM_LAYERS batch_size=$BATCH_SIZE lr=$LEARNING_RATE wd=$WEIGHT_DECAY"

python -u scripts/reinforcement_learning/rsl_rl/train_forward_dynamics.py \
  --seed "$SEED" \
  model.hidden_dim="$HIDDEN_DIM" \
  model.num_layers="$NUM_LAYERS" \
  data.batch_size="$BATCH_SIZE" \
  optim.learning_rate="$LEARNING_RATE" \
  optim.weight_decay="$WEIGHT_DECAY" \
  logging.run_name="$RUN_NAME"
