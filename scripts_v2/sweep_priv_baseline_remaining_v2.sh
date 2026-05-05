#!/usr/bin/env bash
# Second-pass continuation of the privileged-baseline sweep. Dispatches the
# 5 still-needed jobs (Row 3 priv-MLP DAgger is intentionally skipped — its
# iter-0 student is too poor for full-DAgger collection to pass the success
# filter at any width).
#
# Critically different from the prior continuation: GPU "free" is decided
# at every dispatch by a real-time nvidia-smi memory check (>5 GB used =
# busy), so it cannot collide with the original-sweep orchestrators that
# survived the previous mgr-kill.
#
# Pending jobs:
#   r4_mark_disc_bc80k_d512, r4_mark_disc_bc80k_d1024
#   r5_mark_disc_dagger_d1024
#   r1_priv_mlp_bc20k_d1024, r2_priv_mlp_bc80k_d1024

set -uo pipefail

GPU_IDS=${GPU_IDS:-"4 5 6 7"}
BUSY_MEM_MB=${BUSY_MEM_MB:-5000}
HIDDEN_DEPTH=6
N_HEAD=8

EXPERT_CKPT="logs/rsl_rl/teacher/exported/policy.pt"
WANDB_PROJECT="incontext_priv_baseline"
EXP_NAME_BASE="priv_baseline"
OUTPUT_DIR_BASE="logs/priv_baseline"

NUM_DATA_ENVS=1024
NUM_EVAL_ENVS=64
NUM_EVAL_EPISODES=512
INSERTIVE_OBJECT="peg"
RECEPTIVE_OBJECT="peghole"

PRIVILEGED_DATA_TASK="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-PrivilegedKnown-Distillation-DataCollection-v0"
PRIVILEGED_EVAL_TASK="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-PrivilegedKnown-Distillation-StudentEval-v0"
AUGMENTED_DATA_TASK="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Privileged-Augmented-Distillation-DataCollection-v0"
AUGMENTED_EVAL_TASK="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Privileged-Augmented-Distillation-StudentEval-v0"

PRIV_MLP_CONFIG="in_context_privileged_mlp.yaml"
MARK_DISC_CONFIG="in_context_markovian_disc.yaml"

LOG_DIR="${OUTPUT_DIR_BASE}/sweep_logs/$(date +%Y%m%d_%H%M%S)_v2"
mkdir -p "${LOG_DIR}"
echo "[sweep] per-run logs → ${LOG_DIR}"

# tag|config|data_task|eval_task|num_demos|max_iter|num_bins|full_dagger|disc_clip|D
jobs=(
    "r4_mark_disc_bc80k_d1024|${MARK_DISC_CONFIG}|${AUGMENTED_DATA_TASK}|${AUGMENTED_EVAL_TASK}|80000|1|100|0|50.0|1024"
    "r5_mark_disc_dagger_d1024|${MARK_DISC_CONFIG}|${AUGMENTED_DATA_TASK}|${AUGMENTED_EVAL_TASK}|20000|4|100|1|50.0|1024"
    "r1_priv_mlp_bc20k_d1024|${PRIV_MLP_CONFIG}|${PRIVILEGED_DATA_TASK}|${PRIVILEGED_EVAL_TASK}|20000|1|0|0|2.0|1024"
    "r2_priv_mlp_bc80k_d1024|${PRIV_MLP_CONFIG}|${PRIVILEGED_DATA_TASK}|${PRIVILEGED_EVAL_TASK}|80000|1|0|0|2.0|1024"
)
echo "[sweep] queued ${#jobs[@]} pending jobs"

# Track only the jobs we ourselves launched.
declare -A pid_gpu pid_tag

# True iff GPU $1's memory_used > BUSY_MEM_MB right now (live nvidia-smi check).
gpu_busy_now() {
    local idx="$1"
    local mem
    mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$idx" 2>/dev/null | tr -d ' ')
    [[ -z "$mem" ]] && return 0
    (( mem > BUSY_MEM_MB ))
}

# True iff one of MY launched jobs is on GPU $1.
has_my_job_on_gpu() {
    local idx="$1"
    for p in "${!pid_gpu[@]}"; do
        if [[ "${pid_gpu[$p]}" == "$idx" ]] && kill -0 "$p" 2>/dev/null; then
            return 0
        fi
    done
    return 1
}

reap_finished() {
    for p in "${!pid_gpu[@]}"; do
        if ! kill -0 "$p" 2>/dev/null; then
            local g="${pid_gpu[$p]}"
            local tag="${pid_tag[$p]:-?}"
            wait "$p" 2>/dev/null
            local rc=$?
            if (( rc == 0 )); then
                echo "[sweep] DONE  ${tag} on GPU${g}"
            else
                echo "[sweep] FAIL  ${tag} on GPU${g} (rc=${rc}) → ${LOG_DIR}/${tag}.log"
            fi
            unset "pid_gpu[$p]" "pid_tag[$p]"
        fi
    done
}

find_free_gpu() {
    for g in ${GPU_IDS}; do
        if has_my_job_on_gpu "$g"; then continue; fi
        if gpu_busy_now "$g"; then continue; fi
        echo "$g"; return
    done
    echo ""
}

launch_one() {
    local spec="$1" gpu="$2"
    IFS='|' read -r tag config data_task eval_task num_demos max_iter num_bins full_dagger disc_clip D <<<"${spec}"
    local exp="${EXP_NAME_BASE}_${tag}"
    local outdir="${OUTPUT_DIR_BASE}/${tag}"
    local logfile="${LOG_DIR}/${tag}.log"

    local cmd=(python run_incontext_exploration.py
        --expert_policy_checkpoint "${EXPERT_CKPT}"
        --data_task "${data_task}"
        --eval_task "${eval_task}"
        --config_name "${config}"
        --exp_name "${exp}"
        --wandb_project "${WANDB_PROJECT}"
        --output_dir "${outdir}"
        --num_demos "${num_demos}"
        --num_data_envs "${NUM_DATA_ENVS}"
        --num_eval_envs "${NUM_EVAL_ENVS}"
        --num_eval_episodes "${NUM_EVAL_EPISODES}"
        --max_iterations "${max_iter}"
        --insertive_object "${INSERTIVE_OBJECT}"
        --receptive_object "${RECEPTIVE_OBJECT}"
        --no_video
        --use_inverse_actions
        --discretize_clip_val "${disc_clip}"
    )
    if (( num_bins > 0 )); then cmd+=(--num_bins "${num_bins}"); fi
    if (( full_dagger == 1 )); then cmd+=(--full_dagger); fi
    cmd+=(--config_overrides "policy.hidden_dim=${D}" "policy.hidden_depth=${HIDDEN_DEPTH}")
    if [[ "${config}" == "${MARK_DISC_CONFIG}" ]]; then cmd+=("policy.n_head=${N_HEAD}"); fi

    echo "[sweep] LAUNCH ${tag} → GPU${gpu} (log: ${logfile})"
    CUDA_VISIBLE_DEVICES=${gpu} "${cmd[@]}" >"${logfile}" 2>&1 &
    local pid=$!
    pid_gpu[$pid]=$gpu
    pid_tag[$pid]=$tag
    # Wait long enough for Isaac Sim to actually claim GPU memory before
    # re-checking GPU availability for the next dispatch (else two jobs slip
    # onto the same GPU during the boot window).
    sleep 60
}

for spec in "${jobs[@]}"; do
    while :; do
        reap_finished
        gpu=$(find_free_gpu)
        [[ -n "${gpu}" ]] && break
        sleep 30
    done
    launch_one "${spec}" "${gpu}"
done

echo "[sweep] all queued jobs dispatched; draining…"
while (( ${#pid_gpu[@]} > 0 )); do
    reap_finished
    sleep 30
done
echo "[sweep] continuation sweep complete."
