"""In-context exploration / DAgger orchestrator with parallel data collection and training.

This is a parallelized rewrite of ``run_incontext_exploration.py``. The key
differences:

* A single Isaac Sim process (``scripts_v2/tools/collect_demos_worker.py``) is booted
  once on the data-collection GPU and reused across iterations via a
  ``multiprocessing.connection`` Unix socket. This eliminates the per-iteration
  Isaac Sim boot cost.

* Training is launched as a non-blocking ``subprocess.Popen`` on a different GPU,
  so the collection process and the training process run truly in parallel.

* We pipeline evaluation with the next iteration's data collection: once
  ``train_i`` finishes we fire off ``eval_i`` on the training GPU and
  ``collect_{i+1}`` on the collection GPU simultaneously.

Timeline (with 3 iterations):

    GPU 0 (data) : collect_0 ───────── collect_1 ──────── collect_2 ──────── ...
                            \\       /         \\       /
    GPU 1 (train):            train_0 → eval_0   train_1 → eval_1   ...

Usage mirrors ``run_incontext_exploration.py``'s CLI; new args are ``--data_gpu``
and ``--train_gpu`` to pin each component to a specific CUDA device.
"""

from __future__ import annotations

import argparse
import datetime
import glob
import os
import re
import secrets
import select
import signal
import subprocess
import sys
import tempfile
import threading
import time
from multiprocessing.connection import Listener
from typing import Any


# ---------------------------------------------------------------------------
# Subprocess log streaming
# ---------------------------------------------------------------------------


class SubprocessLogStreamer:
    """Tees a subprocess's stdout/stderr to a log file and a terminal stream.

    Owns the output log file and a background daemon thread that reads from the
    subprocess pipe line-by-line, writes raw bytes to the log file, and writes
    prefixed bytes to ``term_stream`` so console output is easy to disambiguate
    from the orchestrator's own prints.
    """

    def __init__(self, log_path: str, term_stream=sys.stdout, prefix: str = "") -> None:
        self.log_path = log_path
        self._term_stream = term_stream
        self._prefix_b = prefix.encode("utf-8")
        self._log_f = open(log_path, "wb")
        self._thread: threading.Thread | None = None
        self._pipe = None

    def attach(self, pipe) -> None:
        """Start streaming from ``pipe`` (typically ``proc.stdout``)."""
        self._pipe = pipe
        self._thread = threading.Thread(target=self._run, args=(pipe,), daemon=True)
        self._thread.start()

    def _run(self, pipe) -> None:
        term_bin = getattr(self._term_stream, "buffer", None)
        leading = True
        try:
            while True:
                chunk = pipe.readline()
                if not chunk:
                    break
                out = (self._prefix_b + chunk) if leading else chunk
                leading = chunk.endswith(b"\n")
                try:
                    self._log_f.write(chunk)
                    self._log_f.flush()
                except Exception:
                    pass
                try:
                    if term_bin is not None:
                        term_bin.write(out)
                        term_bin.flush()
                    else:
                        self._term_stream.write(out.decode("utf-8", errors="replace"))
                        self._term_stream.flush()
                except Exception:
                    pass
        finally:
            try:
                pipe.close()
            except Exception:
                pass

    def tail(self, num_bytes: int = 4000) -> str:
        """Return the last ``num_bytes`` of captured output as text."""
        try:
            self._log_f.flush()
        except Exception:
            pass
        try:
            with open(self.log_path, "rb") as f:
                data = f.read()
            return data[-num_bytes:].decode("utf-8", errors="replace")
        except Exception:
            return ""

    def close(self, join_timeout_s: float = 5.0) -> None:
        if self._thread is not None:
            try:
                self._thread.join(timeout=join_timeout_s)
            except Exception:
                pass
        try:
            self._log_f.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Collection worker handle
# ---------------------------------------------------------------------------


class CollectionWorker:
    """Spawns and talks to a long-lived Isaac Sim data-collection worker."""

    def __init__(
        self,
        task: str,
        num_envs: int,
        expert_path: str,
        insertive_object: str,
        receptive_object: str | None,
        max_episode_length_s: float,
        gpu_id: int,
        seed: int,
        no_video: bool,
        enable_exploration_ratio_filter: bool = False,
        python_executable: str | None = None,
        startup_timeout_s: float = 1200.0,
        log_dir: str | None = None,
    ) -> None:
        self.task = task
        self.num_envs = num_envs
        self.gpu_id = gpu_id
        self.seed = seed
        self.no_video = no_video

        # Create a private Unix socket for the worker to connect back on.
        sock_dir = tempfile.mkdtemp(prefix="dagger_worker_")
        self.socket_path = os.path.join(sock_dir, "collect.sock")
        self.auth_key = secrets.token_hex(16)

        self._listener = Listener(self.socket_path, family="AF_UNIX", authkey=self.auth_key.encode("utf-8"))

        # Build the worker command.
        python = python_executable or sys.executable
        cmd: list[str] = [
            python,
            "scripts_v2/tools/collect_demos_worker.py",
            "--task",
            task,
            "--num_envs",
            str(num_envs),
            "--max_episode_length_s",
            str(max_episode_length_s),
            "--socket_path",
            self.socket_path,
            "--auth_key",
            self.auth_key,
            "--seed",
            str(seed),
            "--headless",
            "--device",
            f"cuda:{gpu_id}",
            f"env.scene.insertive_object={insertive_object}",
            'agent.algorithm.offline_algorithm_cfg.behavior_cloning_cfg.experts_path=["' + expert_path + '"]',
        ]
        if receptive_object is not None:
            cmd.append(f"env.scene.receptive_object={receptive_object}")
        if not no_video:
            cmd.append("--enable_cameras")
        if enable_exploration_ratio_filter:
            cmd.append("--enable_exploration_ratio_filter")

        env = os.environ.copy()
        env.pop("CUDA_VISIBLE_DEVICES", None)
        env.setdefault("PYTHONUNBUFFERED", "1")

        if log_dir is None:
            base = os.path.abspath(os.path.join(os.getcwd(), "logs", "dagger_worker"))
            log_dir = os.path.join(base, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_pid{os.getpid()}")
        os.makedirs(log_dir, exist_ok=True)
        worker_log_path = os.path.join(log_dir, "worker.log")
        self.worker_log_path = worker_log_path
        print(f"[orchestrator] launching collection worker on GPU {gpu_id} (log: {worker_log_path}):\n  {' '.join(cmd)}")
        self._log_streamer = SubprocessLogStreamer(
            log_path=worker_log_path, term_stream=sys.stdout, prefix="[worker] "
        )
        self._proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=0,
        )
        self._log_streamer.attach(self._proc.stdout)

        # Accept the worker's callback. We poll the listener socket *and* the child's
        # exit status so we can fail fast if the worker dies during startup.
        listener_sock = self._listener._listener._socket  # type: ignore[attr-defined]
        deadline = time.time() + startup_timeout_s
        self._conn = None
        while self._conn is None and time.time() < deadline:
            if self._proc.poll() is not None:
                raise RuntimeError(
                    f"Collection worker exited during startup with code {self._proc.returncode}."
                )
            ready, _, _ = select.select([listener_sock], [], [], 1.0)
            if ready:
                self._conn = self._listener.accept()
        if self._conn is None:
            self._proc.send_signal(signal.SIGTERM)
            try:
                self._proc.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            self._log_streamer.close()
            raise RuntimeError(
                f"Timed out ({startup_timeout_s}s) waiting for the collection worker to connect.\n"
                f"Worker log tail ({worker_log_path}):\n{self._log_streamer.tail()}"
            )

        hello = self._conn.recv()
        assert hello.get("status") == "ready", f"Unexpected worker hello: {hello}"
        print(f"[orchestrator] collection worker ready: {hello}")
        self._job_counter = 0

    def collect(
        self,
        dataset_file: str,
        num_demos: int,
        min_exploration_horizon: float,
        max_exploration_horizon: float,
        episode_length_s: float,
        exploration_checkpoint: str | None,
        seed: int,
    ) -> dict[str, Any]:
        self._job_counter += 1
        job_id = self._job_counter
        msg = {
            "cmd": "collect",
            "job_id": job_id,
            "dataset_file": dataset_file,
            "num_demos": num_demos,
            "min_exploration_horizon": min_exploration_horizon,
            "max_exploration_horizon": max_exploration_horizon,
            "episode_length_s": episode_length_s,
            "exploration_checkpoint": exploration_checkpoint,
            "seed": seed,
        }
        self._conn.send(msg)
        print(
            f"[orchestrator] sent collect job {job_id} → {dataset_file}"
            f" (demos={num_demos}, horizons=({min_exploration_horizon},{max_exploration_horizon}),"
            f" episode_length_s={episode_length_s}, ckpt={exploration_checkpoint})"
        )
        reply = self._conn.recv()
        if reply.get("status") != "done":
            raise RuntimeError(f"Collection job {job_id} failed: {reply}")
        print(f"[orchestrator] collect job {job_id} done: {reply.get('result')}")
        return reply.get("result", {})

    def close(self, timeout_s: float = 30.0) -> None:
        if self._conn is not None:
            try:
                self._conn.send({"cmd": "shutdown", "job_id": self._job_counter + 1})
                try:
                    if self._conn.poll(timeout=5.0):
                        self._conn.recv()
                except Exception:
                    pass
                self._conn.close()
            except Exception:
                pass
        try:
            self._listener.close()
        except Exception:
            pass
        try:
            self._proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            # SIGTERM first for a clean shutdown, but Isaac Sim's `simulation_app.close()`
            # can deadlock on CUDA; don't wait long before SIGKILLing.
            print("[orchestrator] collection worker did not exit in time; sending SIGTERM.")
            self._proc.send_signal(signal.SIGTERM)
            try:
                self._proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                print("[orchestrator] worker still alive after SIGTERM; sending SIGKILL.")
                self._proc.kill()
                try:
                    self._proc.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    pass
        if getattr(self, "_log_streamer", None) is not None:
            self._log_streamer.close()
        try:
            os.remove(self.socket_path)
        except FileNotFoundError:
            pass
        try:
            os.rmdir(os.path.dirname(self.socket_path))
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Training & evaluation as non-blocking subprocesses
# ---------------------------------------------------------------------------


_STEP_CKPT_RE = re.compile(r"step_(\d+)\.ckpt$")


def _expected_train_checkpoint(output_dir: str, step: int = 40_000) -> str:
    """Resolve the checkpoint path produced by a training iteration.

    Prefers the preferred ``step`` file if present, otherwise falls back to the highest
    numbered ``step_*.ckpt`` in the checkpoints dir, and finally to ``latest.ckpt``. If
    none of these are available, returns the preferred path so callers can surface a
    useful error when they actually try to load it.
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
        print(
            f"[orchestrator] no step_*.ckpt under {ckpt_dir}; falling back to latest.ckpt."
        )
        return latest

    return preferred


def launch_train_policy(
    config_name: str,
    config_dir: str,
    output_dir: str,
    dataset_config: list[dict[str, float | str]],
    wandb_project: str,
    wandb_group: str,
    exp_name: str,
    lr: float,
    gpu_id: int,
    pretrained_checkpoint: str | None = None,
    seed: int = 0,
    iteration: int = 0,
) -> subprocess.Popen:
    dataset_parts = [
        "{dataset_dir: " + str(d["dataset_dir"]) + ", sampling_ratio: " + str(d["sampling_ratio"]) + "}"
        for d in dataset_config
    ]
    dataset_str = "task.dataset.dataset_config=[" + ",".join(dataset_parts) + "]"

    command = [
        sys.executable,
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

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env.setdefault("PYTHONUNBUFFERED", "1")
    print(f"[orchestrator] launching training on GPU {gpu_id}:\n  {' '.join(command)}")
    return subprocess.Popen(command, env=env)


def launch_eval_policy(
    task: str,
    checkpoint: str,
    num_trajectories: int,
    num_envs: int,
    gpu_id: int,
    episode_length_s: float = 24.0,
    insertive_object: str = "peg",
    receptive_object: str | None = None,
    no_video: bool = False,
    seed: int = 0,
) -> subprocess.Popen:
    command = [
        sys.executable,
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
        command.append(f"env.scene.receptive_object={receptive_object}")
    if not no_video:
        command += ["--save_video", "--enable_cameras"]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env.setdefault("PYTHONUNBUFFERED", "1")
    print(f"[orchestrator] launching evaluation on GPU {gpu_id}:\n  {' '.join(command)}")
    return subprocess.Popen(command, env=env)


def _wait_and_check(proc: subprocess.Popen, label: str) -> None:
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"{label} process failed with return code {rc}")
    print(f"[orchestrator] {label} finished with return code 0")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser("Run in-context exploration (DAgger) with parallel data/train GPUs")
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
        "--expert_policy_checkpoint",
        default="logs/policy_peg_final_v4.pt",
        help="Path to expert policy checkpoint",
    )
    parser.add_argument("--num_demos", type=int, default=10, help="Number of demos to collect per iteration")
    parser.add_argument("--num_data_envs", type=int, default=2, help="Number of parallel envs for data collection")
    parser.add_argument("--num_eval_envs", type=int, default=2, help="Number of parallel envs for evaluation")
    parser.add_argument("--num_eval_episodes", type=int, default=10, help="Number of evaluation episodes")
    parser.add_argument("--data_gpu", type=int, default=0, help="GPU id to pin the data-collection worker to")
    parser.add_argument("--train_gpu", type=int, default=1, help="GPU id to pin training & eval subprocesses to")
    parser.add_argument(
        "--config_dir",
        type=str,
        default="diffusion_policy/diffusion_policy/config",
        help="Path to training config directory",
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
    parser.add_argument("--exp_name", type=str, default="incontext_adaptation", help="Experiment name for logging")
    parser.add_argument("--wandb_project", type=str, default="incontext_adaptation", help="Wandb project name")
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
        help="If set, only collect the dataset for the specified iteration and exit",
    )
    parser.add_argument(
        "--skip_eval",
        action="store_true",
        help="If set, skip evaluation between iterations (useful for faster iteration).",
    )
    parser.add_argument(
        "--enable_exploration_ratio_filter",
        action="store_true",
        help=(
            "If set, enable the filter that rejects demos where the learner drove >=95%% of the"
            " successful episode. Off by default; some tasks want this gate, most don't."
        ),
    )
    args = parser.parse_args()

    sampling_ratio_curriculum = [
        (1.0,),
        (0.25, 0.75),
        (0.2, 0.3, 0.5),
        (0.1, 0.2, 0.3, 0.4),
    ]
    lrs = [1e-4, 1e-5, 1e-5, 1e-5]
    horizons = [(0.2, 0.5), (0.3, 0.8), (0.4, 0.9), (0.5, 1.0)]
    episode_length_s_per_iter = [16.0, 16.0, 16.0, 16.0]

    initial_episode_length_s = 16.0
    eval_episode_length_s = 16.0
    # The worker is constructed once with a long enough max; per-job episode lengths
    # are enforced via manual truncation inside the worker.
    worker_max_episode_length_s = max(
        initial_episode_length_s, eval_episode_length_s, *episode_length_s_per_iter
    ) + 1.0

    exp_name = args.exp_name
    wandb_project = args.wandb_project

    # ---- Resolve iteration 0 state (fresh run vs. resume) --------------------
    if args.start_iteration is not None:
        assert args.checkpoint_dir is not None, "If start_iteration is provided, checkpoint_dir must also be provided"
        assert args.start_iteration > 0, "start_iteration must be greater than 0"
        assert args.start_iteration < args.max_iterations, "start_iteration must be less than max_iterations"
        assert len(sampling_ratio_curriculum) >= args.max_iterations
        assert len(lrs) >= args.max_iterations
        assert len(horizons) >= args.max_iterations - 1
        assert len(episode_length_s_per_iter) >= args.max_iterations - 1

        base_output_dir = args.checkpoint_dir
        initial_dataset_path = args.initial_dataset_path or os.path.join(
            base_output_dir, f"dataset-iteration-{args.start_iteration}"
        )
        resume_exploration_checkpoint = _expected_train_checkpoint(
            os.path.join(base_output_dir, f"iteration_{args.start_iteration - 1}")
        )
        dataset_paths = [initial_dataset_path]
        for i in range(args.start_iteration):
            dataset_paths.append(os.path.join(base_output_dir, f"dataset-iteration-{i + 1}"))
        start_iteration = args.start_iteration
        prior_iter_checkpoint: str | None = resume_exploration_checkpoint
    else:
        base_output_dir = os.path.join(
            args.output_dir, exp_name, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        )
        os.makedirs(base_output_dir, exist_ok=True)
        initial_dataset_path = args.initial_dataset_path
        dataset_paths = []
        start_iteration = 0
        prior_iter_checkpoint = None

    # ---- Spin up the long-lived Isaac Sim data-collection worker -------------
    worker = CollectionWorker(
        task=args.data_task,
        num_envs=args.num_data_envs,
        expert_path=args.expert_policy_checkpoint,
        insertive_object=args.insertive_object,
        receptive_object=args.receptive_object,
        max_episode_length_s=worker_max_episode_length_s,
        gpu_id=args.data_gpu,
        seed=args.seed,
        no_video=args.no_video,
        enable_exploration_ratio_filter=args.enable_exploration_ratio_filter,
    )

    try:
        # -- Iteration 0 bootstrap dataset (only for fresh runs) --------------
        if start_iteration == 0 and initial_dataset_path is None:
            dataset_path = os.path.join(base_output_dir, "dataset-iteration-0")
            worker.collect(
                dataset_file=os.path.join(dataset_path, "data.zarr"),
                num_demos=args.num_demos,
                min_exploration_horizon=0.0,
                max_exploration_horizon=0.0,
                episode_length_s=initial_episode_length_s,
                exploration_checkpoint=None,
                seed=args.seed,
            )
            initial_dataset_path = dataset_path

        if start_iteration == 0:
            dataset_paths = [initial_dataset_path]

        # -- "get_dataset" fast path ------------------------------------------
        if args.get_dataset and start_iteration > 0:
            dataset_path = os.path.join(base_output_dir, f"dataset-iteration-{start_iteration}")
            worker.collect(
                dataset_file=os.path.join(dataset_path, "data.zarr"),
                num_demos=args.num_demos,
                min_exploration_horizon=horizons[start_iteration - 1][0],
                max_exploration_horizon=horizons[start_iteration - 1][1],
                episode_length_s=episode_length_s_per_iter[start_iteration - 1],
                exploration_checkpoint=prior_iter_checkpoint,
                seed=args.seed,
            )
            return

        # -- Main iteration loop ----------------------------------------------
        iteration_checkpoint: str | None = prior_iter_checkpoint
        pending_eval: subprocess.Popen | None = None

        for iteration in range(start_iteration, args.max_iterations):
            print(f"\n================== DAgger iteration {iteration} ==================")
            train_output_dir = os.path.join(base_output_dir, f"iteration_{iteration}")
            os.makedirs(train_output_dir, exist_ok=True)

            # Start training. This runs in parallel with anything we want to do on the data GPU.
            train_proc = launch_train_policy(
                config_name=args.config_name,
                config_dir=args.config_dir,
                output_dir=train_output_dir,
                dataset_config=[
                    {
                        "dataset_dir": dataset_paths[i],
                        "sampling_ratio": sampling_ratio_curriculum[iteration][i],
                    }
                    for i in range(len(sampling_ratio_curriculum[iteration]))
                ],
                wandb_project=wandb_project,
                wandb_group="train",
                exp_name=exp_name,
                lr=lrs[iteration],
                gpu_id=args.train_gpu,
                pretrained_checkpoint=iteration_checkpoint,
                seed=args.seed,
                iteration=iteration,
            )

            # While training runs on the train GPU, the data GPU is free. The next iteration's
            # collection needs the checkpoint that training is producing, so we wait here.
            _wait_and_check(train_proc, f"training iter {iteration}")
            iteration_checkpoint = _expected_train_checkpoint(train_output_dir)

            # If an eval from the previous iteration is still running, reap it before
            # launching the new one.
            if pending_eval is not None:
                _wait_and_check(pending_eval, f"evaluation iter {iteration - 1}")
                pending_eval = None

            # Fan-out: eval_i on train GPU, collect_{i+1} on data GPU, in parallel.
            eval_proc: subprocess.Popen | None = None
            if not args.skip_eval:
                eval_proc = launch_eval_policy(
                    task=args.eval_task,
                    checkpoint=iteration_checkpoint,
                    num_trajectories=args.num_eval_episodes,
                    num_envs=args.num_eval_envs,
                    gpu_id=args.train_gpu,
                    episode_length_s=eval_episode_length_s,
                    insertive_object=args.insertive_object,
                    receptive_object=args.receptive_object,
                    no_video=args.no_video,
                    seed=args.seed,
                )

            next_dataset_path: str | None = None
            if iteration < args.max_iterations - 1:
                next_dataset_path = os.path.join(base_output_dir, f"dataset-iteration-{iteration + 1}")
                worker.collect(
                    dataset_file=os.path.join(next_dataset_path, "data.zarr"),
                    num_demos=args.num_demos,
                    min_exploration_horizon=horizons[iteration][0],
                    max_exploration_horizon=horizons[iteration][1],
                    episode_length_s=episode_length_s_per_iter[iteration],
                    exploration_checkpoint=iteration_checkpoint,
                    seed=args.seed,
                )
                dataset_paths.append(next_dataset_path)

            # After collection finishes, wait for any still-running eval before
            # moving to the next iteration (which will launch a new training run).
            if eval_proc is not None:
                _wait_and_check(eval_proc, f"evaluation iter {iteration}")
    finally:
        worker.close()


if __name__ == "__main__":
    main()
