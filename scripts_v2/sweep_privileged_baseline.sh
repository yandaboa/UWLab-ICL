#!/usr/bin/env bash
# Sweep the 5-row × 3-width privileged-baseline matrix to disentangle the BC
# distillation gap from the identification gap of the context-conditioned
# student. See in_context_adaptation.md for full design notes.
#
# Rows (each x 3 hidden_dims ∈ {256, 512, 1024} = 15 total runs):
#   1. Privileged MLP, BC small  (1 iter,  20k demos)
#   2. Privileged MLP, BC large  (1 iter,  80k demos)
#   3. Privileged MLP, full DAgger (4 iters x 20k demos = 80k)
#   4. Markovian discrete-AR, BC large (1 iter, 80k demos, no privileged obs)
#   5. Markovian discrete-AR, full DAgger (4 iters x 20k demos = 80k)
#
# Usage (inside the isaac-sim container with conda env active):
#   bash scripts_v2/sweep_privileged_baseline.sh
#
# Override knobs:
#   GPU_IDS="4 5 6 7" HIDDEN_DIMS="256 512 1024" \
#       bash scripts_v2/sweep_privileged_baseline.sh
#
# Smoke mode (1 iter, ~200 demos, ~200 grad steps — sanity-check the pipeline):
#   SMOKE=1 bash scripts_v2/sweep_privileged_baseline.sh

set -uo pipefail

# ---- knobs ---------------------------------------------------------------
GPU_IDS=${GPU_IDS:-"4 5 6 7"}
HIDDEN_DIMS=${HIDDEN_DIMS:-"256 512 1024"}
HIDDEN_DEPTH=${HIDDEN_DEPTH:-6}
N_HEAD=${N_HEAD:-8}

EXPERT_CKPT=${EXPERT_CKPT:-"logs/rsl_rl/teacher/exported/policy.pt"}
WANDB_PROJECT=${WANDB_PROJECT:-"incontext_priv_baseline"}
EXP_NAME_BASE=${EXP_NAME_BASE:-"priv_baseline"}
OUTPUT_DIR_BASE=${OUTPUT_DIR_BASE:-"logs/priv_baseline"}

NUM_DATA_ENVS=${NUM_DATA_ENVS:-1024}
NUM_EVAL_ENVS=${NUM_EVAL_ENVS:-64}
NUM_EVAL_EPISODES=${NUM_EVAL_EPISODES:-512}
INSERTIVE_OBJECT=${INSERTIVE_OBJECT:-"peg"}
RECEPTIVE_OBJECT=${RECEPTIVE_OBJECT:-"peghole"}

PRIVILEGED_DATA_TASK="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-PrivilegedKnown-Distillation-DataCollection-v0"
PRIVILEGED_EVAL_TASK="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-PrivilegedKnown-Distillation-StudentEval-v0"
AUGMENTED_DATA_TASK="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Privileged-Augmented-Distillation-DataCollection-v0"
AUGMENTED_EVAL_TASK="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Privileged-Augmented-Distillation-StudentEval-v0"

PRIV_MLP_CONFIG="in_context_privileged_mlp.yaml"
MARK_DISC_CONFIG="in_context_markovian_disc.yaml"

# Smoke mode: tiny everything so we just verify the pipeline boots end-to-end.
SMOKE=${SMOKE:-0}
if [[ "${SMOKE}" == "1" ]]; then
    NUM_DATA_ENVS=128
    NUM_EVAL_ENVS=8
    NUM_EVAL_EPISODES=8
    SMOKE_DEMOS=200
    SMOKE_OVERRIDES=("training.max_gradient_steps=200" "training.checkpoint_every=200" "training.val_every=200" "training.sample_every=200")
    OUTPUT_DIR_BASE="${OUTPUT_DIR_BASE}_smoke"
    EXP_NAME_BASE="${EXP_NAME_BASE}_smoke"
    echo "[sweep] SMOKE mode — tiny demos / grad steps."
fi

LOG_DIR="${OUTPUT_DIR_BASE}/sweep_logs/$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOG_DIR}"
echo "[sweep] per-run logs → ${LOG_DIR}"

# ---- enumerate jobs ------------------------------------------------------
# Each job is a "spec" string: row|tag|config|data_task|eval_task|num_demos|max_iter|num_bins|full_dagger|disc_clip|D
jobs=()
for D in ${HIDDEN_DIMS}; do
    for row in 1 2 3 4 5; do
        case "${row}" in
            1)  spec="r1_priv_mlp_bc20k_d${D}|${PRIV_MLP_CONFIG}|${PRIVILEGED_DATA_TASK}|${PRIVILEGED_EVAL_TASK}|20000|1|0|0|2.0|${D}";;
            2)  spec="r2_priv_mlp_bc80k_d${D}|${PRIV_MLP_CONFIG}|${PRIVILEGED_DATA_TASK}|${PRIVILEGED_EVAL_TASK}|80000|1|0|0|2.0|${D}";;
            3)  spec="r3_priv_mlp_dagger_d${D}|${PRIV_MLP_CONFIG}|${PRIVILEGED_DATA_TASK}|${PRIVILEGED_EVAL_TASK}|20000|4|0|1|2.0|${D}";;
            4)  spec="r4_mark_disc_bc80k_d${D}|${MARK_DISC_CONFIG}|${AUGMENTED_DATA_TASK}|${AUGMENTED_EVAL_TASK}|80000|1|100|0|50.0|${D}";;
            5)  spec="r5_mark_disc_dagger_d${D}|${MARK_DISC_CONFIG}|${AUGMENTED_DATA_TASK}|${AUGMENTED_EVAL_TASK}|20000|4|100|1|50.0|${D}";;
        esac
        if [[ "${SMOKE}" == "1" ]]; then
            spec=$(echo "${spec}" | awk -F'|' -v d="${SMOKE_DEMOS}" 'BEGIN{OFS="|"} { $5=d; $6=1; print }')
        fi
        jobs+=("${spec}")
    done
done
echo "[sweep] queued ${#jobs[@]} jobs across GPUs {${GPU_IDS}}"

# ---- worker pool ---------------------------------------------------------
declare -A gpu_pid pid_gpu pid_tag

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
            unset "gpu_pid[$g]" "pid_gpu[$p]" "pid_tag[$p]"
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

    local extra_overrides=()
    if [[ "${SMOKE}" == "1" ]]; then
        extra_overrides=("${SMOKE_OVERRIDES[@]}")
    fi

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

    if (( num_bins > 0 )); then
        cmd+=(--num_bins "${num_bins}")
    fi
    if (( full_dagger == 1 )); then
        cmd+=(--full_dagger)
    fi

    cmd+=(--config_overrides
        "policy.hidden_dim=${D}"
        "policy.hidden_depth=${HIDDEN_DEPTH}"
    )
    if [[ "${config}" == "${MARK_DISC_CONFIG}" ]]; then
        cmd+=("policy.n_head=${N_HEAD}")
    fi
    cmd+=("${extra_overrides[@]}")

    echo "[sweep] LAUNCH ${tag} → GPU${gpu} (log: ${logfile})"
    CUDA_VISIBLE_DEVICES=${gpu} "${cmd[@]}" >"${logfile}" 2>&1 &
    local pid=$!
    gpu_pid[$gpu]=$pid
    pid_gpu[$pid]=$gpu
    pid_tag[$pid]=$tag
}

# ---- dispatch ------------------------------------------------------------
for spec in "${jobs[@]}"; do
    while :; do
        reap_finished
        gpu=$(free_gpu)
        [[ -n "${gpu}" ]] && break
        wait -n
    done
    launch_one "${spec}" "${gpu}"
done

echo "[sweep] all jobs dispatched; draining…"
while (( ${#pid_gpu[@]} > 0 )); do
    wait -n
    reap_finished
done

echo "[sweep] all done."
