import argparse
import datetime
import os
import subprocess


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
):
    command = [
        "python",
        "scripts_v2/tools/collect_demos.py",
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

    demos = subprocess.run(command)
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
):
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

    eval_process = subprocess.run(command)
    if eval_process.returncode != 0:
        print("Evaluation process failed with return code:", eval_process.returncode)
        raise SystemExit(1)
    print("Evaluation process finished with return code:", eval_process.returncode)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Run in-context exploration and collect demos")
    parser.add_argument(
        "--data_task",
        type=str,
        default="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-RGB-DataCollection-v0",
        help="Data collection task name",
    )
    parser.add_argument(
        "--eval_task",
        type=str,
        default="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-RGB-Play-v0",
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
        default="baseline_pg_spoc.yaml",
        help="Name of the training config file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="logs/incontext_exploration",
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
        default="incontext_exploration",
        help="Experiment name for logging",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="incontext_exploration",
        help="Wandb project name",
    )
    parser.add_argument("--no_video", action="store_true", help="If set, do not save videos during evaluation")
    parser.add_argument("--insertive_object", type=str, default="peg", help="Insertive object type")
    parser.add_argument("--receptive_object", type=str, default=None, help="Receptive object type")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for environment")
    parser.add_argument("--start_iteration", type=int, default=None, help="Starting iteration number")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Directory of run to resume from")
    parser.add_argument("--max_iterations", type=int, default=3, help="Maximum number of iterations to run")
    parser.add_argument(
        "--get_dataset",
        action="store_true",
        help="If set, only collect dataset for the specified iteration and exit",
    )
    args = parser.parse_args()

    sampling_ratio_curriculum = [
        (1.0,),
        (0.25, 0.75),
        (0.2, 0.3, 0.5),
        (0.1, 0.2, 0.3, 0.4),
    ]
    lrs = [1e-4, 1e-5, 1e-5, 1e-5]
    horizons = [(0.2, 0.5), (0.3, 0.8), (0.4, 0.9)]
    episode_length_s = [20.0, 24.0, 24.0]

    initial_episode_length_s = 20.0
    eval_episode_length_s = 24.0
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
        exploration_checkpoint = os.path.join(
            base_output_dir, f"iteration_{args.start_iteration - 1}", "checkpoints", "step_0040000.ckpt"
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
            )
            args.initial_dataset_path = dataset_path

        dataset_paths = [args.initial_dataset_path]
        iteration_checkpoint = None
        start_iteration = 0

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
        )

        iteration_checkpoint = os.path.join(
            base_output_dir, f"iteration_{iteration}", "checkpoints", "step_0040000.ckpt"
        )
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
            )
            dataset_paths.append(dataset_path)
