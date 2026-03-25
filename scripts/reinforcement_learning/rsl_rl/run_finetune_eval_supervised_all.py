#!/usr/bin/env python3
"""Batch finetune + eval runner for supervised MLP checkpoints.

Per task-episode file:
1) finetune a base supervised checkpoint on that dataset
2) evaluate the finetuned checkpoint while forcing env resets from that task file

Supports one active job per GPU slot (queue mode), similar to run_eval_point_mass_supervised_all.py.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
TRAIN_SCRIPT = SCRIPT_DIR / "train_supervised.py"
EVAL_SCRIPT = SCRIPT_DIR / "eval_supervised.py"

DEFAULT_OUTPUT_ROOT = SCRIPT_DIR.parents[2] / "logs" / "rsl_rl" / "supervised_finetune_eval"

# -----------------------------------------------------------------------------
# Defaults: edit these in-code sweep values, like training_sweep.sh.
# Each path format uses "{x}" replacement.
# -----------------------------------------------------------------------------
TASK_EPISODE_PATH_FORMATS: List[str] = [
    "episodes/20260316_044141/individual_tasks/64samples/task_group_{x}.pt",
    "episodes/20260316_044141/individual_tasks/16samples/task_group_{x}.pt",
    "episodes/20260316_044141/individual_tasks/8samples/task_group_{x}.pt",
]
TASK_EPISODE_INDEX_GROUPS: List[str] = [
    "000000 000001 000002 000003 000004 000005 000006 000007 000008 000009 000010 000011 000012 000013 000014 000015 000016",
]
# If True: run baseline eval only on base checkpoint for BASELINE_* task episodes.
# If False: run finetune + eval for TASK_* task episodes.
RUN_BASELINE_EVAL_ONLY = False
BASELINE_TASK_EPISODE_PATH_FORMATS: List[str] = [
    "episodes/20260316_044141/individual_tasks/64samples/task_group_{x}.pt",
]
BASELINE_TASK_EPISODE_INDEX_GROUPS: List[str] = [
    "000000 000001 000002 000003 000004 000005 000006 000007 000008 000009 000010 000011 000012 000013 000014 000015 000016",
]
CUDA_DEVICE_IDS: List[str] = ["0", "1", "2", "3", "4", "5", "6", "7"]
PYTHON_BIN = "python"
EVAL_NUM_STEPS = 500
FINETUNE_EPOCHS = 100
FINETUNE_LR_DIVISOR = 5.0
DEFAULT_EVAL_EXTRA_ARGS = (
    "env.scene.insertive_object=peg "
    "env.scene.receptive_object=peghole "
    "--use_wandb"
)


@dataclass
class JobSpec:
    task_episode_path: Path
    task_episode_path_format: str
    task_episode_path_format_index: int
    task_episode_index_token: str
    task_index: int
    gpu_slot: str | None
    device: str
    job_dir: Path
    run_log: Path
    finetune_dataset: Path | None
    should_finetune: bool


def _parse_index_groups(source_groups: List[str]) -> List[List[str]]:
    parsed: List[List[str]] = []
    for group in source_groups:
        indices = [idx for idx in group.split() if idx]
        if indices:
            parsed.append(indices)
    return parsed


def _task_episode_specs(
    repo_root: Path,
    path_formats: List[str],
    index_groups: List[str],
    path_format_name: str,
) -> List[tuple[int, str, str, Path]]:
    groups = _parse_index_groups(index_groups)
    if len(groups) == 0:
        raise ValueError("No task episode index groups configured.")

    if len(path_formats) == 0:
        raise ValueError(f"No {path_format_name} configured.")

    all_specs: List[tuple[int, str, str, Path]] = []
    for format_idx, pattern in enumerate(path_formats):
        if "{x}" not in pattern:
            raise ValueError(f"{path_format_name}[{format_idx}] must contain '{{x}}': {pattern}")
        for group in groups:
            for idx in group:
                path = Path(pattern.replace("{x}", idx)).expanduser()
                if not path.is_absolute():
                    path = (repo_root / path).resolve()
                else:
                    path = path.resolve()
                all_specs.append((format_idx, pattern, idx, path))
    return all_specs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run batch supervised finetune+eval or baseline-eval jobs.")
    parser.add_argument("--base-checkpoint", type=str, required=True, help="Base supervised checkpoint to finetune.")
    parser.add_argument("--train-config", type=str, required=True, help="Base train_supervised YAML config path.")
    parser.add_argument("--task", type=str, default="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Play-v0", help="Isaac task name for eval_supervised.py")
    parser.add_argument("--eval-num-steps", type=int, default=EVAL_NUM_STEPS, help="Evaluation environment steps.")
    parser.add_argument("--eval-log-interval", type=int, default=100, help="eval_supervised.py --log_interval")
    parser.add_argument("--num-envs", type=int, default=1024, help="Number of environments to simulate for evaluation.")
    parser.add_argument("--checkpoint-kind", choices=["best", "last"], default="best")
    parser.add_argument("--finetune-epochs", type=int, default=FINETUNE_EPOCHS)
    parser.add_argument("--finetune-lr-divisor", type=float, default=FINETUNE_LR_DIVISOR)
    parser.add_argument("--output-root", type=str, default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--python-bin", type=str, default=PYTHON_BIN)
    parser.add_argument(
        "--eval-extra-args",
        type=str,
        default=DEFAULT_EVAL_EXTRA_ARGS,
        help="Optional extra args appended to eval_supervised.py command.",
    )
    parser.add_argument(
        "--train-extra-args",
        type=str,
        default="",
        help="Optional extra args appended to train_supervised.py command.",
    )
    args = parser.parse_args()

    # SCRIPT_DIR points to .../UWLab-ICL/scripts/reinforcement_learning/rsl_rl
    # so repo root is parents[2] => .../UWLab-ICL
    args.repo_root = SCRIPT_DIR.parents[2]
    args.base_checkpoint = str(Path(args.base_checkpoint).expanduser().resolve())
    args.train_config = str(Path(args.train_config).expanduser().resolve())
    args.output_root = str(Path(args.output_root).expanduser().resolve())
    return args


def _build_jobs(args: argparse.Namespace) -> List[JobSpec]:
    base_checkpoint = Path(args.base_checkpoint)
    if not base_checkpoint.is_file():
        raise ValueError(f"--base-checkpoint not found: {base_checkpoint}")
    train_config = Path(args.train_config)
    if not train_config.is_file():
        raise ValueError(f"--train-config not found: {train_config}")
    if args.finetune_epochs <= 0:
        raise ValueError("--finetune-epochs must be > 0")
    if args.eval_num_steps <= 0:
        raise ValueError("--eval-num-steps must be > 0")
    if args.finetune_lr_divisor <= 0:
        raise ValueError("--finetune-lr-divisor must be > 0")

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    should_finetune = not RUN_BASELINE_EVAL_ONLY
    if should_finetune:
        task_specs = _task_episode_specs(
            repo_root=args.repo_root,
            path_formats=TASK_EPISODE_PATH_FORMATS,
            index_groups=TASK_EPISODE_INDEX_GROUPS,
            path_format_name="TASK_EPISODE_PATH_FORMATS",
        )
    else:
        task_specs = _task_episode_specs(
            repo_root=args.repo_root,
            path_formats=BASELINE_TASK_EPISODE_PATH_FORMATS,
            index_groups=BASELINE_TASK_EPISODE_INDEX_GROUPS,
            path_format_name="BASELINE_TASK_EPISODE_PATH_FORMATS",
        )
    cuda_device_ids = list(CUDA_DEVICE_IDS)

    jobs: List[JobSpec] = []
    for task_idx, (format_idx, path_format, index_token, task_path) in enumerate(task_specs):
        if not task_path.is_file():
            print(f"Warning: task_episodes missing, skipping: {task_path}", file=sys.stderr)
            continue
        gpu_slot: str | None = None
        if len(cuda_device_ids) > 0:
            gpu_slot = cuda_device_ids[len(jobs) % len(cuda_device_ids)]
            assigned_device = f"cuda:{gpu_slot}"
        else:
            assigned_device = "cuda"

        mode_tag = "finetune" if should_finetune else "baseline"
        job_dir = output_root / f"task_{task_idx:04d}_fmt{format_idx:02d}_{task_path.stem}_{mode_tag}"
        job_dir.mkdir(parents=True, exist_ok=True)
        finetune_dataset: Path | None = None
        if should_finetune:
            finetune_dataset = task_path

        jobs.append(
            JobSpec(
                task_episode_path=task_path,
                task_episode_path_format=path_format,
                task_episode_path_format_index=format_idx,
                task_episode_index_token=index_token,
                task_index=task_idx,
                gpu_slot=gpu_slot,
                device=assigned_device,
                job_dir=job_dir,
                run_log=job_dir / "run.log",
                finetune_dataset=finetune_dataset,
                should_finetune=should_finetune,
            )
        )
    return jobs


def _job_shell_command(job: JobSpec, args: argparse.Namespace) -> str:
    base_ckpt = shlex.quote(args.base_checkpoint)
    task_episodes = shlex.quote(str(job.task_episode_path))
    train_script = shlex.quote(str(TRAIN_SCRIPT))
    eval_script = shlex.quote(str(EVAL_SCRIPT))
    python_bin = shlex.quote(args.python_bin)
    task = shlex.quote(args.task)
    device = shlex.quote(job.device)
    eval_extra = args.eval_extra_args.strip()
    train_extra = args.train_extra_args.strip()

    eval_num_envs = "" if args.num_envs is None else f"--num_envs {int(args.num_envs)} "
    eval_headless = "--headless "

    train_extra_payload = ""
    if train_extra:
        train_extra_payload = " " + " ".join(shlex.quote(token) for token in shlex.split(train_extra))
    eval_extra_payload = ""
    if eval_extra:
        eval_extra_payload = " " + " ".join(shlex.quote(token) for token in shlex.split(eval_extra))

    if not job.should_finetune:
        return (
            "set -euo pipefail; "
            f"echo \"[START] baseline-eval task={job.task_episode_path} device={job.device}\"; "
            f"CHECKPOINT_PATH={base_ckpt}; "
            f"{python_bin} {eval_script} "
            f"--task {task} "
            f"--num_steps {int(args.eval_num_steps)} "
            f"--log_interval {int(args.eval_log_interval)} "
            f"{eval_num_envs}"
            f"{eval_headless}"
            f"--checkpoint \"$CHECKPOINT_PATH\" "
            f"--task_episodes {task_episodes} "
            f"--device {device}"
            f"{eval_extra_payload}; "
            f"echo \"[DONE] baseline-eval task={job.task_episode_path}\""
        )

    if job.finetune_dataset is None:
        raise ValueError("Internal error: finetune dataset missing for finetune job.")
    finetune_dataset = shlex.quote(str(job.finetune_dataset))
    checkpoint_filename = "best_model.pt" if args.checkpoint_kind == "best" else "last_model.pt"
    checkpoint_filename = shlex.quote(checkpoint_filename)

    return (
        "set -euo pipefail; "
        f"echo \"[START] finetune+eval task={job.task_episode_path} device={job.device}\"; "
        "TMP_TRAIN_LOG=\"$(mktemp)\"; "
        f"{python_bin} {train_script} "
        f"--config {shlex.quote(args.train_config)} "
        f"--finetune_checkpoint {base_ckpt} "
        f"--finetune_dataset {finetune_dataset} "
        f"--finetune_lr_divisor {float(args.finetune_lr_divisor)} "
        f"--epochs {int(args.finetune_epochs)}"
        f"{train_extra_payload} 2>&1 | tee \"$TMP_TRAIN_LOG\"; "
        "FINETUNE_DIR=\"$(python - \"$TMP_TRAIN_LOG\" <<'PY'\n"
        "import sys\n"
        "log_path = sys.argv[1]\n"
        "marker = '[INFO] Saved outputs to:'\n"
        "saved = None\n"
        "with open(log_path, 'r', encoding='utf-8', errors='replace') as f:\n"
        "    for line in f:\n"
        "        if marker in line:\n"
        "            saved = line.split(marker, 1)[1].strip()\n"
        "if not saved:\n"
        "    raise SystemExit('Could not parse finetune output directory from train log.')\n"
        "print(saved)\n"
        "PY\n"
        ")\"; "
        "rm -f \"$TMP_TRAIN_LOG\"; "
        f"CHECKPOINT_PATH=\"$FINETUNE_DIR/{checkpoint_filename}\"; "
        "if [[ ! -f \"$CHECKPOINT_PATH\" ]]; then "
        "  echo \"[WARN] Requested checkpoint missing: $CHECKPOINT_PATH, falling back to last_model.pt\"; "
        "  CHECKPOINT_PATH=\"$FINETUNE_DIR/last_model.pt\"; "
        "fi; "
        "if [[ ! -f \"$CHECKPOINT_PATH\" ]]; then "
        "  echo \"[ERROR] No checkpoint found in $FINETUNE_DIR\"; "
        "  exit 1; "
        "fi; "
        f"{python_bin} {eval_script} "
        f"--task {task} "
        f"--num_steps {int(args.eval_num_steps)} "
        f"--log_interval {int(args.eval_log_interval)} "
        f"{eval_num_envs}"
        f"{eval_headless}"
        f"--checkpoint \"$CHECKPOINT_PATH\" "
        f"--task_episodes {task_episodes} "
        f"--device {device}"
        f"{eval_extra_payload}; "
        f"echo \"[DONE] finetune+eval task={job.task_episode_path}\""
    )


def _launch_job(job: JobSpec, args: argparse.Namespace) -> subprocess.Popen[str]:
    log_fp = open(job.run_log, "w", encoding="utf-8")
    proc = subprocess.Popen(
        ["/bin/bash", "-lc", _job_shell_command(job, args)],
        stdout=log_fp,
        stderr=subprocess.STDOUT,
        text=True,
    )
    log_fp.close()
    mode = "finetune+eval" if job.should_finetune else "baseline-eval"
    print(f"Launched {mode} task={job.task_episode_path.name} on {job.device} (pid={proc.pid}), log: {job.run_log}")
    return proc


def main() -> int:
    args = parse_args()
    try:
        jobs = _build_jobs(args)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if len(jobs) == 0:
        print("No jobs launched.")
        return 1

    queue_mode = len(CUDA_DEVICE_IDS) > 0

    failed = False
    progress = tqdm(total=len(jobs), desc="Completed finetune+eval jobs", unit="job")
    if queue_mode:
        slot_procs: Dict[str, subprocess.Popen[str]] = {}
        slot_names: Dict[str, str] = {}
        for job in jobs:
            assert job.gpu_slot is not None
            slot = job.gpu_slot
            if slot in slot_procs:
                prior_proc = slot_procs[slot]
                prior_name = slot_names[slot]
                if prior_proc.wait() == 0:
                    print(f"[OK] {prior_name}")
                else:
                    print(f"[FAIL] {prior_name}", file=sys.stderr)
                    failed = True
                progress.update(1)
            slot_procs[slot] = _launch_job(job, args)
            slot_names[slot] = f"{job.task_episode_path.name} on cuda:{slot}"

        print("Waiting for final job on each GPU slot...")
        for slot, proc in slot_procs.items():
            name = slot_names[slot]
            if proc.wait() == 0:
                print(f"[OK] {name}")
            else:
                print(f"[FAIL] {name}", file=sys.stderr)
                failed = True
            progress.update(1)
    else:
        procs: List[subprocess.Popen[str]] = []
        names: List[str] = []
        for job in jobs:
            procs.append(_launch_job(job, args))
            names.append(f"{job.task_episode_path.name} on {job.device}")

        print(f"Waiting for {len(procs)} finetune+eval job(s)...")
        for proc, name in zip(procs, names):
            if proc.wait() == 0:
                print(f"[OK] {name}")
            else:
                print(f"[FAIL] {name}", file=sys.stderr)
                failed = True
            progress.update(1)

    progress.close()
    if failed:
        print(f"Completed with failures. Check logs under: {Path(args.output_root).resolve()}", file=sys.stderr)
        return 1

    print("All finetune+eval jobs completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
