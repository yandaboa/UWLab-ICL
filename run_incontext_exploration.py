import argparse
import datetime
import glob
import os
import re
import subprocess

from incontext_eval_log import IncontextEvalLog


def collect_demos(
    task: str,
    dataset_file: str,
    num_envs: int,
    num_demos: int,
    min_exploration_horizon: float,
    max_exploration_horizon: float,
    episode_length_s: float,
    expert_path: str = "logs/policy_peg_final_v4.pt",
    insertive_object: str = "peg",
    receptive_object: str | None = None,
    exploration_checkpoint: str | None = None,
    no_video: bool = False,
    seed: int = 0,
    disable_exploration_ratio_filter: bool = False,
    disable_task_success_filter: bool = False,
    transformer_mini_batch_size: int = 64,
    use_kv_cache: bool = True,
    kv_cache_max_seq_len: int | None = None,
    profile_worker: bool = False,
    use_inverse_actions: bool = False,
    num_bins: int = 0,
    discretize_clip_val: float = 50.0,
    expert_action_scale: list[float] | None = None,
    full_dagger: bool = False,
):
    """Spawn ``collect_demos_asteroid.py`` as a subprocess to collect a zarr dataset."""
    command = [
        "python",
        "scripts_v2/tools/collect_demos_asteroid.py",
        "--task",
        task,
        "--dataset_file",
        dataset_file,
        "--num_envs",
        str(num_envs),
        "--num_demos",
        str(num_demos),
        "--headless",
        "--seed",
        str(seed),
        "--max_exploration_horizon",
        str(max_exploration_horizon),
        "--min_exploration_horizon",
        str(min_exploration_horizon),
        "--episode_length_s",
        str(episode_length_s),
        "--transformer_mini_batch_size",
        str(transformer_mini_batch_size),
        f"env.scene.insertive_object={insertive_object}",
        'agent.algorithm.offline_algorithm_cfg.behavior_cloning_cfg.experts_path=["' + expert_path + '"]',
    ]
    if exploration_checkpoint is not None:
        command += [
            "--exploration_checkpoint",
            exploration_checkpoint,
        ]
    if receptive_object is not None:
        command += [f"env.scene.receptive_object={receptive_object}"]
    if not no_video:
        command += ["--enable_cameras"]
    if disable_exploration_ratio_filter:
        command += ["--disable_exploration_ratio_filter"]
    if disable_task_success_filter:
        assert not disable_exploration_ratio_filter, (
            "--disable_task_success_filter requires the exploration-ratio filter to stay ON"
            " (i.e., do NOT pass --disable_exploration_ratio_filter). Otherwise every episode"
            " would be admitted with no quality gate."
        )
        command += ["--disable_task_success_filter"]
    if not use_kv_cache:
        command += ["--no_kv_cache"]
    if kv_cache_max_seq_len is not None:
        command += ["--kv_cache_max_seq_len", str(kv_cache_max_seq_len)]
    if use_inverse_actions:
        command += ["--use_inverse_actions"]
    if num_bins > 0:
        command += ["--num_bins", str(num_bins), "--discretize_clip_val", str(discretize_clip_val)]
    if expert_action_scale is not None:
        command += ["--expert_action_scale", *(str(s) for s in expert_action_scale)]
    if full_dagger:
        command += ["--full_dagger"]

    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    if profile_worker:
        env["DIFFUSION_POLICY_PROFILE"] = "1"
        env.setdefault("DIFFUSION_POLICY_PROFILE_EVERY", "50")

    demos = subprocess.run(command, env=env)
    if demos.returncode != 0:
        print("Demo collection process failed with return code:", demos.returncode)
        raise SystemExit(1)

    print("Demo collection process finished with return code:", demos.returncode)


def train_policy(
    config_name: str,
    config_dir: str,
    output_dir: str,
    dataset_config: list[dict[str, float | str]],
    wandb_project: str,
    wandb_group: str,
    exp_name: str,
    lr: float,
    pretrained_checkpoint: str | None = None,
    seed: int = 0,
    iteration: int = 0,
    config_overrides: list[str] | None = None,
):
    dataset_str = "task.dataset.dataset_config=["
    for data in dataset_config:
        dataset_str += "{dataset_dir: " + str(data["dataset_dir"]) + ", sampling_ratio: " + str(data["sampling_ratio"]) + "},"
    dataset_str = dataset_str[:-1] + "]"

    command = [
        "python",
        "diffusion_policy/train.py",
        "--config-name",
        config_name,
        "--config-dir",
        config_dir,
        "output_dir=" + output_dir,
        dataset_str,
        "name=" + exp_name,
        "exp_name=" + exp_name,
        "logging.project=" + wandb_project,
        "logging.group=" + wandb_group,
        "optimizer.lr=" + str(lr),
        "seed=" + str(seed),
        "iteration=" + str(iteration),
    ]

    if pretrained_checkpoint is not None:
        command.append("checkpoint.pretrained_ckpt_path=" + pretrained_checkpoint)

    # Forward arbitrary Hydra ``key=value`` overrides (e.g. policy.hidden_dim=128).
    if config_overrides:
        command.extend(config_overrides)

    train_process_return = subprocess.run(command)
    if train_process_return.returncode != 0:
        print("Training process failed with return code:", train_process_return.returncode)
        raise SystemExit(1)
    print("Training process finished with return code:", train_process_return.returncode)


def eval_policy(
    task: str,
    checkpoint: str,
    num_trajectories: int,
    num_envs: int,
    episode_length_s: float = 24.0,
    insertive_object: str = "peg",
    receptive_object: str | None = None,
    no_video: bool = False,
    seed: int = 0,
    transformer_mini_batch_size: int = 64,
    stats_output_path: str | None = None,
    iteration: int | None = None,
):
    """Spawn ``eval_distilled_policy.py`` as a subprocess to evaluate a trained checkpoint."""
    command = [
        "python",
        "scripts_v2/tools/eval_distilled_policy.py",
        "--task",
        task,
        "--seed",
        str(seed),
        "--num_trajectories",
        str(num_trajectories),
        "--num_envs",
        str(num_envs),
        "--transformer_mini_batch_size",
        str(transformer_mini_batch_size),
        "--headless",
        "--checkpoint",
        checkpoint,
        f"env.scene.insertive_object={insertive_object}",
        f"env.episode_length_s={episode_length_s}",
    ]
    if receptive_object is not None:
        command += [f"env.scene.receptive_object={receptive_object}"]
    if not no_video:
        command += ["--save_video", "--enable_cameras"]
    if stats_output_path is not None:
        command += ["--stats_output_path", stats_output_path]
    if iteration is not None:
        command += ["--iteration", str(iteration)]

    eval_process = subprocess.run(command)
    if eval_process.returncode != 0:
        print("Evaluation process failed with return code:", eval_process.returncode)
        raise SystemExit(1)
    print("Evaluation process finished with return code:", eval_process.returncode)


_STEP_CKPT_RE = re.compile(r"step_(\d+)\.ckpt$")


def _expected_train_checkpoint(output_dir: str, step: int = 50_000) -> str:
    """Resolve the checkpoint path produced by a training iteration.

    Selection order (no ``best.ckpt`` lookup — kept simple and predictable):
      1. ``step_{step:07d}.ckpt`` for the requested step (default 50k).
      2. The highest-numbered ``step_*.ckpt`` in the checkpoints dir.
      3. ``latest.ckpt`` — the final-state snapshot from the workspace.
    """
    ckpt_dir = os.path.join(output_dir, "checkpoints")

    preferred = os.path.join(ckpt_dir, f"step_{step:07d}.ckpt")
    if os.path.exists(preferred):
        return preferred

    candidates: list[tuple[int, str]] = []
    for path in glob.glob(os.path.join(ckpt_dir, "step_*.ckpt")):
        m = _STEP_CKPT_RE.search(os.path.basename(path))
        if m is not None:
            candidates.append((int(m.group(1)), path))
    if candidates:
        candidates.sort(key=lambda x: x[0])
        best_step, best_path = candidates[-1]
        print(
            f"[orchestrator] step_{step:07d}.ckpt missing under {ckpt_dir}; "
            f"falling back to {os.path.basename(best_path)} (step {best_step})."
        )
        return best_path

    latest = os.path.join(ckpt_dir, "latest.ckpt")
    if os.path.exists(latest):
        print(f"[orchestrator] no step_*.ckpt under {ckpt_dir}; falling back to latest.ckpt.")
        return latest

    return preferred


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Run in-context exploration and collect demos")
    parser.add_argument(
        "--data_task",
        type=str,
        default="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Privileged-Augmented-Distillation-DataCollection-v0",
        help="Data collection task name",
    )
    parser.add_argument(
        "--eval_task",
        type=str,
        default="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Privileged-Augmented-Distillation-StudentEval-v0",
        help="Evaluation task name",
    )
    parser.add_argument(
        "--expert_policy_checkpoint", default="logs/policy_peg_final_v4.pt", help="Path to expert policy checkpoint"
    )
    parser.add_argument("--num_demos", type=int, default=10, help="Number of demos to collect")
    parser.add_argument("--num_data_envs", type=int, default=2, help="Number of parallel environments for data collection")
    parser.add_argument("--num_eval_envs", type=int, default=2, help="Number of parallel environments for evaluation")
    parser.add_argument("--num_eval_episodes", type=int, default=10, help="Number of evaluation episodes")
    parser.add_argument(
        "--config_dir",
        type=str,
        default="diffusion_policy/diffusion_policy/config",
        help="Path to config directory",
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default="in_context_adaptation.yaml",
        help="Name of the training config file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="logs/in_context_adaptation",
        help="Directory to save output logs and models",
    )
    parser.add_argument(
        "--initial_dataset_path",
        type=str,
        default=None,
        help="Path to initial dataset for training the policy",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default="incontext_adaptation",
        help="Experiment name for logging",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="incontext_adaptation",
        help="Wandb project name",
    )
    parser.add_argument("--no_video", action="store_true", help="If set, do not save videos during evaluation")
    parser.add_argument("--insertive_object", type=str, default="peg", help="Insertive object type")
    parser.add_argument("--receptive_object", type=str, default=None, help="Receptive object type")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for environment")
    parser.add_argument("--start_iteration", type=int, default=None, help="Starting iteration number")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Directory of run to resume from")
    parser.add_argument("--max_iterations", type=int, default=4, help="Maximum number of iterations to run")
    parser.add_argument(
        "--get_dataset",
        action="store_true",
        help="If set, only collect dataset for the specified iteration and exit",
    )
    parser.add_argument(
        "--skip_eval",
        action="store_true",
        help="If set, skip evaluation between iterations (useful for faster iteration).",
    )
    parser.add_argument(
        "--disable_exploration_ratio_filter",
        action="store_true",
        help=(
            "If set, disable the filter that rejects demos where the learner drove >=95%% of the"
            " successful episode. ON by default; pass this flag only for tasks where you want to"
            " keep learner-dominated demos (e.g. pure imitation from the exploration policy)."
        ),
    )
    parser.add_argument(
        "--disable_task_success_filter",
        action="store_true",
        help=(
            "If set, admit every completed episode (success or not) as long as it passes the"
            " exploration-ratio filter. Requires the exploration-ratio filter to remain ON — i.e."
            " you MUST NOT also pass --disable_exploration_ratio_filter (an assert enforces this)."
            " Useful when the exploration policy produces good trajectories that the task's success"
            " termination does not capture."
        ),
    )
    parser.add_argument(
        "--transformer_mini_batch_size",
        type=int,
        default=64,
        help=(
            "Mini-batch size used by DiffusionPolicyWrapper when serializing transformer inference"
            " across envs, forwarded to both the collection and eval subprocesses. Bounds peak"
            " activation memory; too-small values (e.g. 8) dominate wall time for large num_envs."
        ),
    )
    parser.add_argument(
        "--no_kv_cache",
        action="store_true",
        help=(
            "Forwarded to the collection subprocess: disable incremental KV-cached inference in"
            " DiffusionPolicyWrapper and fall back to re-encoding the full trajectory each step."
            " Useful for A/B profiling; normally you want KV caching on."
        ),
    )
    parser.add_argument(
        "--kv_cache_max_seq_len",
        type=int,
        default=None,
        help=(
            "Upper bound on per-env KV cache length forwarded to the collection subprocess."
            " Defaults to the transformer's n_positions (usually 1024)."
        ),
    )
    parser.add_argument(
        "--profile_worker",
        action="store_true",
        help=(
            "Set DIFFUSION_POLICY_PROFILE=1 in the collection subprocess environment so the"
            " DiffusionPolicyWrapper emits per-stage inference timings."
        ),
    )
    parser.add_argument("--checkpoint_num", type=int, default=50000, help="Checkpoint number to resume from / evaluate (matches step_{checkpoint_num:07d}.ckpt; falls back to highest available step_* or latest.ckpt).")
    parser.add_argument(
        "--use_inverse_actions",
        action="store_true",
        help=(
            "Forwarded to the collection subprocess: compute the analytically optimal action"
            " for the augmented MDP via the OSC term's inverse_process_actions, so a non-augmented"
            " expert produces correct demos in the augmented env."
        ),
    )
    parser.add_argument(
        "--num_bins",
        type=int,
        default=0,
        help=(
            "If > 0, forward to the collection subprocess and discretize the 6 continuous arm"
            " dims into this many uniform bins over [-discretize_clip_val, +discretize_clip_val]."
            " Gripper is sign-thresholded. A discretize_spec.json is written alongside the dataset."
        ),
    )
    parser.add_argument(
        "--discretize_clip_val",
        type=float,
        default=2.0,
        help="Symmetric clip range for binning continuous arm action dims (forwarded). Default: 2.0.",
    )
    parser.add_argument(
        "--expert_action_scale",
        type=float,
        nargs=6,
        default=[0.01, 0.01, 0.002, 0.02, 0.02, 0.2],
        metavar=("sx", "sy", "sz", "rx", "ry", "rz"),
        help=(
            "Six-element action scale the expert was trained with (XYZ + axis-angle). Forwarded"
            " to inverse_process_actions when --use_inverse_actions is set."
            " Default: [0.01, 0.01, 0.002, 0.02, 0.02, 0.2]."
        ),
    )
    parser.add_argument(
        "--config_overrides",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Extra Hydra-style overrides forwarded verbatim to ``diffusion_policy/train.py``"
            " (e.g. ``--config_overrides policy.hidden_dim=128 policy.hidden_depth=2 policy.n_head=2``)."
            " Applies to every training subprocess in this run."
        ),
    )
    parser.add_argument(
        "--full_dagger",
        action="store_true",
        help=(
            "Forward to the collection subprocess: in iterations >0, the student drives every"
            " env for the full episode and the recorded action target is the inverse-mapped"
            " expert action (true full DAgger, not the intervention-tail variant)."
            " Iter-0 always runs expert-only (no exploration policy yet)."
        ),
    )
    args = parser.parse_args()

    if args.disable_task_success_filter:
        assert not args.disable_exploration_ratio_filter, (
            "--disable_task_success_filter requires the exploration-ratio filter to stay ON."
            " Remove --disable_exploration_ratio_filter, or drop --disable_task_success_filter."
            " Otherwise no quality filter remains and every episode would be admitted."
        )

    sampling_ratio_curriculum = [
        (1.0,),
        (0.25, 0.75),
        (0.2, 0.3, 0.5),
        (0.1, 0.2, 0.3, 0.4),
    ]
    lrs = [1e-4, 1e-5, 1e-5, 1e-5]
    horizons = [(0.1, 0.3), (0.15, 0.35), (0.2, 0.4), (0.25, 0.5)]
    episode_length_s = [8.0, 9.0, 10.0, 11.0]

    initial_episode_length_s = 6.0
    eval_episode_length_s = 11.0
    exp_name = args.exp_name
    wandb_project = args.wandb_project

    if args.start_iteration is not None:
        assert args.checkpoint_dir is not None, "If start_iteration is provided, checkpoint_dir must also be provided"
        assert args.start_iteration > 0, "start_iteration must be greater than 0"
        assert args.start_iteration < args.max_iterations, "start_iteration must be less than max_iterations"
        assert len(sampling_ratio_curriculum) >= args.max_iterations
        assert len(lrs) >= args.max_iterations
        assert len(horizons) >= args.max_iterations - 1
        assert len(episode_length_s) >= args.max_iterations - 1

        base_output_dir = args.checkpoint_dir
        args.initial_dataset_path = (
            os.path.join(base_output_dir, f"dataset-iteration-{args.start_iteration}")
            if args.initial_dataset_path is None
            else args.initial_dataset_path
        )
        exploration_checkpoint = _expected_train_checkpoint(
            os.path.join(base_output_dir, f"iteration_{args.start_iteration - 1}"), args.checkpoint_num
        )
        dataset_path = os.path.join(base_output_dir, f"dataset-iteration-{args.start_iteration}")
        if args.get_dataset:
            collect_demos(
                task=args.data_task,
                dataset_file=os.path.join(dataset_path, "data.zarr"),
                num_envs=args.num_data_envs,
                num_demos=args.num_demos,
                min_exploration_horizon=horizons[args.start_iteration - 1][0],
                max_exploration_horizon=horizons[args.start_iteration - 1][1],
                episode_length_s=episode_length_s[args.start_iteration - 1],
                expert_path=args.expert_policy_checkpoint,
                exploration_checkpoint=exploration_checkpoint,
                insertive_object=args.insertive_object,
                receptive_object=args.receptive_object,
                no_video=args.no_video,
                seed=args.seed,
                disable_exploration_ratio_filter=args.disable_exploration_ratio_filter,
                disable_task_success_filter=args.disable_task_success_filter,
                transformer_mini_batch_size=args.transformer_mini_batch_size,
                use_kv_cache=not args.no_kv_cache,
                kv_cache_max_seq_len=args.kv_cache_max_seq_len,
                profile_worker=args.profile_worker,
                use_inverse_actions=args.use_inverse_actions,
                num_bins=args.num_bins,
                discretize_clip_val=args.discretize_clip_val,
                expert_action_scale=args.expert_action_scale,
                full_dagger=args.full_dagger,
            )
            raise SystemExit(0)

        dataset_paths = [args.initial_dataset_path]
        for i in range(args.start_iteration):
            dataset_paths.append(os.path.join(base_output_dir, f"dataset-iteration-{i + 1}"))
        iteration_checkpoint = exploration_checkpoint
        start_iteration = args.start_iteration
    else:
        base_output_dir = os.path.join(args.output_dir, exp_name, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        os.makedirs(base_output_dir, exist_ok=True)

        if args.initial_dataset_path is None:
            dataset_path = os.path.join(base_output_dir, "dataset-iteration-0")
            collect_demos(
                task=args.data_task,
                dataset_file=os.path.join(dataset_path, "data.zarr"),
                num_envs=args.num_data_envs,
                num_demos=args.num_demos,
                min_exploration_horizon=0.0,
                max_exploration_horizon=0.0,
                episode_length_s=initial_episode_length_s,
                expert_path=args.expert_policy_checkpoint,
                insertive_object=args.insertive_object,
                receptive_object=args.receptive_object,
                no_video=args.no_video,
                seed=args.seed,
                disable_exploration_ratio_filter=args.disable_exploration_ratio_filter,
                disable_task_success_filter=args.disable_task_success_filter,
                transformer_mini_batch_size=args.transformer_mini_batch_size,
                use_kv_cache=not args.no_kv_cache,
                kv_cache_max_seq_len=args.kv_cache_max_seq_len,
                profile_worker=args.profile_worker,
                use_inverse_actions=args.use_inverse_actions,
                num_bins=args.num_bins,
                discretize_clip_val=args.discretize_clip_val,
                expert_action_scale=args.expert_action_scale,
            )
            args.initial_dataset_path = dataset_path

        dataset_paths = [args.initial_dataset_path]
        iteration_checkpoint = None
        start_iteration = 0

    eval_log_path = os.path.join(base_output_dir, "eval_log.json")
    eval_log = IncontextEvalLog(
        log_path=eval_log_path,
        exp_name=exp_name,
        config={
            "eval_task": args.eval_task,
            "data_task": args.data_task,
            "insertive_object": args.insertive_object,
            "receptive_object": args.receptive_object,
            "num_eval_episodes": args.num_eval_episodes,
            "num_eval_envs": args.num_eval_envs,
            "eval_episode_length_s": eval_episode_length_s,
            "seed": args.seed,
            "expert_policy_checkpoint": args.expert_policy_checkpoint,
            "wandb_project": wandb_project,
            "base_output_dir": base_output_dir,
        },
    )
    print(f"[orchestrator] writing per-iteration eval stats to {eval_log_path}")

    for iteration in range(start_iteration, args.max_iterations):
        print(f"Starting iteration {iteration}...")
        train_output_dir = os.path.join(base_output_dir, f"iteration_{iteration}")
        os.makedirs(train_output_dir, exist_ok=True)

        train_policy(
            config_name=args.config_name,
            config_dir=args.config_dir,
            output_dir=train_output_dir,
            dataset_config=[
                {"dataset_dir": dataset_paths[i], "sampling_ratio": sampling_ratio_curriculum[iteration][i]}
                for i in range(len(sampling_ratio_curriculum[iteration]))
            ],
            wandb_project=wandb_project,
            wandb_group="train",
            exp_name=exp_name,
            pretrained_checkpoint=iteration_checkpoint,
            lr=lrs[iteration],
            seed=args.seed,
            iteration=iteration,
            config_overrides=args.config_overrides,
        )

        iteration_checkpoint = _expected_train_checkpoint(train_output_dir, args.checkpoint_num)
        if not args.skip_eval:
            iter_stats_path = os.path.join(train_output_dir, "eval_stats.json")
            eval_policy(
                task=args.eval_task,
                checkpoint=iteration_checkpoint,
                num_trajectories=args.num_eval_episodes,
                num_envs=args.num_eval_envs,
                episode_length_s=eval_episode_length_s,
                insertive_object=args.insertive_object,
                receptive_object=args.receptive_object,
                no_video=args.no_video,
                seed=args.seed,
                transformer_mini_batch_size=args.transformer_mini_batch_size,
                stats_output_path=iter_stats_path,
                iteration=iteration,
            )
            if os.path.exists(iter_stats_path):
                eval_log.append_from_stats_file(
                    iter_stats_path,
                    iteration=iteration,
                    checkpoint=iteration_checkpoint,
                    task=args.eval_task,
                )
                print(f"[orchestrator] appended iteration {iteration} stats to {eval_log_path}")
            else:
                print(
                    f"[orchestrator] warning: eval stats file {iter_stats_path} was not written;"
                    f" skipping log append for iteration {iteration}."
                )

        if iteration < args.max_iterations - 1:
            dataset_path = os.path.join(base_output_dir, f"dataset-iteration-{iteration + 1}")
            collect_demos(
                task=args.data_task,
                dataset_file=os.path.join(dataset_path, "data.zarr"),
                num_envs=args.num_data_envs,
                num_demos=args.num_demos,
                min_exploration_horizon=horizons[iteration][0],
                max_exploration_horizon=horizons[iteration][1],
                episode_length_s=episode_length_s[iteration],
                expert_path=args.expert_policy_checkpoint,
                exploration_checkpoint=iteration_checkpoint,
                insertive_object=args.insertive_object,
                receptive_object=args.receptive_object,
                no_video=args.no_video,
                seed=args.seed,
                disable_exploration_ratio_filter=args.disable_exploration_ratio_filter,
                disable_task_success_filter=args.disable_task_success_filter,
                transformer_mini_batch_size=args.transformer_mini_batch_size,
                use_kv_cache=not args.no_kv_cache,
                kv_cache_max_seq_len=args.kv_cache_max_seq_len,
                profile_worker=args.profile_worker,
                use_inverse_actions=args.use_inverse_actions,
                num_bins=args.num_bins,
                discretize_clip_val=args.discretize_clip_val,
                expert_action_scale=args.expert_action_scale,
                full_dagger=args.full_dagger,
            )
            dataset_paths.append(dataset_path)
