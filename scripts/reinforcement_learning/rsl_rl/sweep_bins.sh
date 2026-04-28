#!/usr/bin/env bash
# Sweep over action-space bin counts to measure discretization impact on expert performance.
#
# Usage (from repo root, inside the Isaac Sim Docker with lti conda activated):
#   bash scripts/reinforcement_learning/rsl_rl/sweep_bins.sh \
#       [--checkpoint <path>] [--num_envs <N>] [--num_steps <S>] [--object <obj>]
#
# Defaults:
#   checkpoint : logs/rsl_rl/teacher/model_2900.pt
#   num_envs   : 64
#   num_steps  : 1000   (at 10 Hz env-step rate this is ~100 s, usually 50-80 episodes)
#   object     : peg / peghole

set -euo pipefail

# ── parse optional overrides ──────────────────────────────────────────────────
CHECKPOINT="logs/rsl_rl/teacher/model_2900.pt"
NUM_ENVS=1024
NUM_STEPS=200
INSERTIVE="peg"
RECEPTIVE="peghole"
CLIP_VAL=25.0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoint) CHECKPOINT="$2"; shift 2 ;;
        --num_envs)   NUM_ENVS="$2";   shift 2 ;;
        --num_steps)  NUM_STEPS="$2";  shift 2 ;;
        --object)     INSERTIVE="$2";  RECEPTIVE="${3:-peghole}"; shift 3 ;;
        --clip_val)   CLIP_VAL="$2";   shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

TASK="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Augmented-Play-v0"
RESULTS_DIR="logs/discretize_sweep"
mkdir -p "$RESULTS_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$RESULTS_DIR/sweep_${TIMESTAMP}.log"
RESULTS_CSV="$RESULTS_DIR/results_${TIMESTAMP}.csv"

echo "checkpoint : $CHECKPOINT"
echo "task       : $TASK"
echo "num_envs   : $NUM_ENVS"
echo "num_steps  : $NUM_STEPS"
echo "object     : $INSERTIVE / $RECEPTIVE"
echo "clip_val   : $CLIP_VAL"
echo "log        : $LOG_FILE"
echo ""

echo "num_bins,num_episodes,num_successes,success_rate" > "$RESULTS_CSV"

# ── helper: run one condition ─────────────────────────────────────────────────
run_condition() {
    local bins="$1"
    local label

    if [[ "$bins" -eq 0 ]]; then
        label="continuous"
    else
        label="${bins}_bins"
    fi

    echo "================================================================"
    echo "Running: $label"
    echo "================================================================"

    local extra_args=""
    if [[ "$bins" -gt 0 ]]; then
        extra_args="--num_bins $bins --discretize_clip_val $CLIP_VAL"
    fi

    # shellcheck disable=SC2086
    local output
    output=$(python scripts/reinforcement_learning/rsl_rl/play.py \
        --task "$TASK" \
        --num_envs "$NUM_ENVS" \
        --num_steps "$NUM_STEPS" \
        --checkpoint "$CHECKPOINT" \
        --headless \
        $extra_args \
        "env.scene.insertive_object=$INSERTIVE" \
        "env.scene.receptive_object=$RECEPTIVE" \
        2>&1 | tee -a "$LOG_FILE")

    local episodes successes rate
    episodes=$(echo "$output" | grep "^Number of episodes:" | awk '{print $NF}')
    successes=$(echo "$output" | grep "^Number of successes:" | awk '{print $NF}')
    rate=$(echo "$output"     | grep "^Success rate:" | awk '{print $NF}' | tr -d '%')

    episodes=${episodes:-0}
    successes=${successes:-0}
    rate=${rate:-0.00}

    echo "$bins,$episodes,$successes,$rate" >> "$RESULTS_CSV"
    echo "  → episodes=$episodes  successes=$successes  rate=${rate}%"
    echo ""
}

# ── conditions to sweep ───────────────────────────────────────────────────────
# 0 = continuous baseline; then increasing bin counts
BIN_COUNTS=(0 2 3 5 10 20 50 100 256)

for b in "${BIN_COUNTS[@]}"; do
    run_condition "$b"
done

# ── print summary table ───────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "SWEEP RESULTS"
echo "================================================================"
echo "Checkpoint : $CHECKPOINT"
echo "Task       : $TASK"
echo "Object     : $INSERTIVE / $RECEPTIVE"
echo "Clip val   : $CLIP_VAL"
echo ""
column -t -s',' "$RESULTS_CSV"
echo ""
echo "Full log : $LOG_FILE"
echo "CSV      : $RESULTS_CSV"

# ── write markdown results file ───────────────────────────────────────────────
MD_FILE="$RESULTS_DIR/results_${TIMESTAMP}.md"

{
    echo "# Action Discretization Sweep Results"
    echo ""
    echo "**Date:** $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
    echo "## Config"
    echo ""
    echo "| Parameter | Value |"
    echo "|---|---|"
    echo "| Task | \`$TASK\` |"
    echo "| Checkpoint | \`$CHECKPOINT\` |"
    echo "| Object | $INSERTIVE / $RECEPTIVE |"
    echo "| num\_envs | $NUM_ENVS |"
    echo "| num\_steps | $NUM_STEPS |"
    echo "| clip\_val | $CLIP_VAL |"
    echo ""
    echo "## Results"
    echo ""
    echo "| num\_bins | Episodes | Successes | Success Rate |"
    echo "|---|---|---|---|"
    # skip header row
    tail -n +2 "$RESULTS_CSV" | while IFS=',' read -r bins eps succ rate; do
        if [[ "$bins" -eq 0 ]]; then
            label="continuous"
        else
            label="$bins"
        fi
        echo "| $label | $eps | $succ | ${rate}% |"
    done
    echo ""
    echo "## Notes"
    echo ""
    echo "- \`num_bins=0\` is the unmodified continuous expert (baseline)."
    echo "- Continuous arm dims (indices 0–5) are snapped to the nearest of \`num_bins\` uniform"
    echo "  centers in \`[-${CLIP_VAL}, +${CLIP_VAL}]\`."
    echo "- Gripper dim (index 6) is always sign-thresholded to {-1, +1} regardless of \`num_bins\`."
    echo "- Policy: \`act_inference\` (deterministic mean, no tanh; actions are unbounded Gaussian)."
} > "$MD_FILE"

echo "Markdown : $MD_FILE"
