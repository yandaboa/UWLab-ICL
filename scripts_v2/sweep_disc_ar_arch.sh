#!/usr/bin/env bash
# Sweep transformer architecture (hidden_depth, n_head, hidden_dim) for the
# discrete-AR head end-to-end through run_incontext_exploration.py. One run
# per GPU; new runs launched as GPUs free up.
#
# Reference command (single run, reproduces the per-job invocation):
#   python run_incontext_exploration.py \
#     --expert_policy_checkpoint logs/rsl_rl/teacher/exported/policy.pt \
#     --config_name in_context_adaptation_interleave.yaml \
#     --exp_name sweep_disc_ar --wandb_project incontext_adaptation_sweep \
#     --output_dir logs/sweep_disc_ar --num_demos 20000 --num_data_envs 1024 \
#     --num_eval_envs 64 --num_eval_episodes 128 --max_iterations 4 \
#     --insertive_object peg --receptive_object peghole --no_video \
#     --use_inverse_actions --num_bins 100 --discretize_clip_val 50.0 \
#     --config_overrides policy.hidden_dim=<D> policy.hidden_depth=<L> policy.n_head=<H>
#
# Usage:
#   bash scripts_v2/sweep_disc_ar_arch.sh
#
# Override knobs:
#   NUM_GPUS=8 HIDDEN_DEPTHS="2 4 8" N_HEADS="2 4 8" HIDDEN_DIMS="64 128 256 512" \
#     bash scripts_v2/sweep_disc_ar_arch.sh

set -uo pipefail

# ---- knobs ---------------------------------------------------------------
NUM_GPUS=${NUM_GPUS:-8}
HIDDEN_DEPTHS=${HIDDEN_DEPTHS:-"4 8 12"}
N_HEADS=${N_HEADS:-"4 8 12"}
HIDDEN_DIMS=${HIDDEN_DIMS:-"64 128 256 512"}

EXPERT_CKPT=${EXPERT_CKPT:-"logs/rsl_rl/teacher/exported/policy.pt"}
CONFIG_NAME=${CONFIG_NAME:-"in_context_adaptation_interleave.yaml"}
WANDB_PROJECT=${WANDB_PROJECT:-"incontext_adaptation_sweep"}
EXP_NAME_BASE=${EXP_NAME_BASE:-"sweep_disc_ar"}
OUTPUT_DIR_BASE=${OUTPUT_DIR_BASE:-"logs/sweep_disc_ar"}

NUM_DEMOS=${NUM_DEMOS:-20000}
NUM_DATA_ENVS=${NUM_DATA_ENVS:-1024}
NUM_EVAL_ENVS=${NUM_EVAL_ENVS:-64}
NUM_EVAL_EPISODES=${NUM_EVAL_EPISODES:-128}
MAX_ITERATIONS=${MAX_ITERATIONS:-4}
INSERTIVE_OBJECT=${INSERTIVE_OBJECT:-"peg"}
RECEPTIVE_OBJECT=${RECEPTIVE_OBJECT:-"peghole"}
NUM_BINS=${NUM_BINS:-100}
DISCRETIZE_CLIP_VAL=${DISCRETIZE_CLIP_VAL:-50.0}

LOG_DIR="${OUTPUT_DIR_BASE}/sweep_logs/$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOG_DIR}"
echo "[sweep] per-run logs → ${LOG_DIR}"
echo "[sweep] sweeping hidden_depth ∈ {${HIDDEN_DEPTHS}}, n_head ∈ {${N_HEADS}}, hidden_dim ∈ {${HIDDEN_DIMS}} on ${NUM_GPUS} GPUs"
echo

# ---- enumerate combos (skip hidden_dim not divisible by n_head) ----------
combos=()
for L in ${HIDDEN_DEPTHS}; do
    for H in ${N_HEADS}; do
        for D in ${HIDDEN_DIMS}; do
            if (( D % H != 0 )); then
                echo "[sweep] skip L=${L} H=${H} D=${D} (D not divisible by H)"
                continue
            fi
            combos+=("${L} ${H} ${D}")
        done
    done
done
echo "[sweep] ${#combos[@]} combos to run"
echo

# ---- worker pool: GPU index -> running pid ------------------------------
declare -A gpu_pid
declare -A pid_gpu
declare -A pid_tag

reap_finished() {
    # Walk pid_gpu and free any pid that has exited.
    for p in "${!pid_gpu[@]}"; do
        if ! kill -0 "$p" 2>/dev/null; then
            local g="${pid_gpu[$p]}"
            local tag="${pid_tag[$p]:-?}"
            wait "$p" 2>/dev/null
            local rc=$?
            if (( rc == 0 )); then
                echo "[sweep] DONE  ${tag} on GPU${g} (pid ${p})"
            else
                echo "[sweep] FAIL  ${tag} on GPU${g} (pid ${p}, rc=${rc})"
            fi
            unset "gpu_pid[$g]" "pid_gpu[$p]" "pid_tag[$p]"
        fi
    done
}

free_gpu() {
    for g in $(seq 0 $((NUM_GPUS - 1))); do
        if [[ -z "${gpu_pid[$g]:-}" ]]; then
            echo "$g"
            return
        fi
    done
    echo ""
}

launch_one() {
    local L="$1" H="$2" D="$3" gpu="$4"
    local tag="l${L}_h${H}_d${D}"
    local exp="${EXP_NAME_BASE}_${tag}"
    local outdir="${OUTPUT_DIR_BASE}/${tag}"
    local logfile="${LOG_DIR}/${tag}.log"

    echo "[sweep] LAUNCH ${tag} → GPU${gpu} (log: ${logfile})"
    CUDA_VISIBLE_DEVICES=${gpu} \
        python run_incontext_exploration.py \
            --expert_policy_checkpoint "${EXPERT_CKPT}" \
            --config_name "${CONFIG_NAME}" \
            --exp_name "${exp}" \
            --wandb_project "${WANDB_PROJECT}" \
            --output_dir "${outdir}" \
            --num_demos "${NUM_DEMOS}" \
            --num_data_envs "${NUM_DATA_ENVS}" \
            --num_eval_envs "${NUM_EVAL_ENVS}" \
            --num_eval_episodes "${NUM_EVAL_EPISODES}" \
            --max_iterations "${MAX_ITERATIONS}" \
            --insertive_object "${INSERTIVE_OBJECT}" \
            --receptive_object "${RECEPTIVE_OBJECT}" \
            --no_video \
            --use_inverse_actions \
            --num_bins "${NUM_BINS}" \
            --discretize_clip_val "${DISCRETIZE_CLIP_VAL}" \
            --config_overrides \
                "policy.hidden_depth=${L}" \
                "policy.n_head=${H}" \
                "policy.hidden_dim=${D}" \
        > "${logfile}" 2>&1 &
    local pid=$!
    gpu_pid[$gpu]=$pid
    pid_gpu[$pid]=$gpu
    pid_tag[$pid]=$tag
}

# ---- dispatch loop -------------------------------------------------------
for combo in "${combos[@]}"; do
    read -r L H D <<<"${combo}"

    # Wait for a free GPU.
    while :; do
        reap_finished
        gpu=$(free_gpu)
        [[ -n "${gpu}" ]] && break
        wait -n  # any one job finishes
    done

    launch_one "${L}" "${H}" "${D}" "${gpu}"
done

# Drain remaining jobs.
echo "[sweep] all combos dispatched; draining…"
while (( ${#pid_gpu[@]} > 0 )); do
    wait -n
    reap_finished
done

echo "[sweep] all done."
