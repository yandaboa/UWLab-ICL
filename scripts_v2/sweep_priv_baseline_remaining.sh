#!/usr/bin/env bash
# Continuation of the privileged-baseline sweep. Dispatches the 6 still-needed
# jobs across the GPU pool as they become free. Excludes Row 3 (priv-MLP full
# DAgger) — its iter-0 BC student has 0% eval success, so iter-1 collection
# under success-filter never makes progress.
#
# Already-done: r1_d256, r2_d256, r4_d256
# Already-running (started by the prior sweep mgr that was killed): r5_d256,
# r1_d512, r2_d512 — DO NOT relaunch these.
# Remaining: r4_d512, r5_d512, r1_d1024, r2_d1024, r4_d1024, r5_d1024
#
# Usage (inside isaac-sim container with conda env active):
#   bash scripts_v2/sweep_priv_baseline_remaining.sh

set -uo pipefail

GPU_IDS=${GPU_IDS:-"4 5 6 7"}
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

LOG_DIR="${OUTPUT_DIR_BASE}/sweep_logs/$(date +%Y%m%d_%H%M%S)_remaining"
mkdir -p "${LOG_DIR}"
echo "[sweep] per-run logs → ${LOG_DIR}"

# row|tag|config|data_task|eval_task|num_demos|max_iter|num_bins|full_dagger|disc_clip|D
jobs=(
    "r4_mark_disc_bc80k_d512|${MARK_DISC_CONFIG}|${AUGMENTED_DATA_TASK}|${AUGMENTED_EVAL_TASK}|80000|1|100|0|50.0|512"
    "r5_mark_disc_dagger_d512|${MARK_DISC_CONFIG}|${AUGMENTED_DATA_TASK}|${AUGMENTED_EVAL_TASK}|20000|4|100|1|50.0|512"
    "r1_priv_mlp_bc20k_d1024|${PRIV_MLP_CONFIG}|${PRIVILEGED_DATA_TASK}|${PRIVILEGED_EVAL_TASK}|20000|1|0|0|2.0|1024"
    "r2_priv_mlp_bc80k_d1024|${PRIV_MLP_CONFIG}|${PRIVILEGED_DATA_TASK}|${PRIVILEGED_EVAL_TASK}|80000|1|0|0|2.0|1024"
    "r4_mark_disc_bc80k_d1024|${MARK_DISC_CONFIG}|${AUGMENTED_DATA_TASK}|${AUGMENTED_EVAL_TASK}|80000|1|100|0|50.0|1024"
    "r5_mark_disc_dagger_d1024|${MARK_DISC_CONFIG}|${AUGMENTED_DATA_TASK}|${AUGMENTED_EVAL_TASK}|20000|4|100|1|50.0|1024"
)
echo "[sweep] queued ${#jobs[@]} remaining jobs"

declare -A gpu_pid pid_gpu pid_tag

# Treat GPU as busy if it already has > 5 GB of memory used (likely a sweep job
# carried over from the prior, killed sweep manager).
mark_busy_from_nvidia() {
    while read -r line; do
        local idx=$(echo "$line" | cut -d, -f1 | xargs)
        local mem=$(echo "$line" | cut -d, -f2 | tr -d -c 0-9)
        for g in ${GPU_IDS}; do
            if [[ "$idx" == "$g" ]] && (( mem > 5000 )); then
                gpu_pid[$g]="external"
                echo "[sweep] GPU${g} marked busy externally (memory ${mem} MiB)"
            fi
        done
    done < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader)
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
                echo "[sweep] FAIL  ${tag} on GPU${g} (rc=${rc})"
            fi
            unset "gpu_pid[$g]" "pid_gpu[$p]" "pid_tag[$p]"
        fi
    done
    # Also free any GPU whose external occupant has gone idle (memory dropped low).
    for g in ${GPU_IDS}; do
        if [[ "${gpu_pid[$g]:-}" == "external" ]]; then
            mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g")
            if (( mem < 5000 )); then
                unset "gpu_pid[$g]"
                echo "[sweep] GPU${g} freed (external job ended; memory ${mem} MiB)"
            fi
        fi
    done
}

free_gpu() {
    for g in ${GPU_IDS}; do
        if [[ -z "${gpu_pid[$g]:-}" ]]; then
            echo "$g"; return
        fi
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
    gpu_pid[$gpu]=$pid
    pid_gpu[$pid]=$gpu
    pid_tag[$pid]=$tag
}

mark_busy_from_nvidia

for spec in "${jobs[@]}"; do
    while :; do
        reap_finished
        gpu=$(free_gpu)
        [[ -n "${gpu}" ]] && break
        sleep 30
    done
    launch_one "${spec}" "${gpu}"
done

echo "[sweep] all queued jobs dispatched; draining…"
while :; do
    reap_finished
    has_internal=0
    for p in "${!pid_gpu[@]}"; do
        if [[ "$p" != "external" ]]; then has_internal=1; fi
    done
    (( has_internal == 0 )) && break
    sleep 30
done

echo "[sweep] remaining-jobs sweep complete."
