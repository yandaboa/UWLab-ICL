"""Roll out the trained DIAYN policy with random per-env skills, collect
end-effector trajectories from each env's first episode (truncated at the
first success step), and plot the two skills against each other in 3D — one
line per episode, colored by skill. Trajectories are expressed in the
**receptive-object frame** (translation only) so the per-env reset
variance is removed and only skill-conditioned behaviour remains.

Modeled on ``analysis/eval_discriminator.py``: same rollout-and-track-first-
episode pattern, augmented to record EE world pos and receptive-object world
pos per step.

Usage (inside the isaac-sim container, lti env, single GPU; do NOT use GPU 0
on this host — renderer crashes at init):

    CUDA_VISIBLE_DEVICES=1 python analysis/diversity_skill_ee_traj.py \\
        --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Diversity-Play-v0 \\
        --checkpoint logs/rsl_rl/ur5e_robotiq_2f85_omnireset_diversity/2026-05-18_11-02-44/model_4000.pt \\
        --num_envs 32 --num_steps 120 --num_episodes_per_skill 10 \\
        --headless \\
        env.scene.insertive_object=peg env.scene.receptive_object=peghole \\
        env.observations.policy.skill.params.num_skills=2 \\
        agent.algorithm.number_of_skills=2

Outputs (under <ckpt dir>/diversity_skill_ee_traj/):
    raw.pt        — per-env raw buffers (ee_pos_w, receptive_pos_w, skill,
                    first_success_step, first_episode_end_step)
    episodes.pt   — list of dicts, one per kept episode, with
                    ee_pos_local (T_i, 3), skill, succeeded
    plot_3d.png   — 3D plot of EE trajectories in receptive-object frame
"""
import argparse
import pathlib
import sys

# repo-root and rsl_rl scripts dir on sys.path BEFORE local imports
_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
_RSL_SCRIPTS = _REPO_ROOT / "scripts" / "reinforcement_learning" / "rsl_rl"
for p in (_REPO_ROOT, _RSL_SCRIPTS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from isaaclab.app import AppLauncher

import cli_args  # type: ignore  # provided by scripts/reinforcement_learning/rsl_rl

parser = argparse.ArgumentParser(description="Per-skill EE trajectory plot for DIAYN policy.")
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--num_steps", type=int, default=120, help="Max env steps per rollout pass.")
parser.add_argument("--num_episodes_per_skill", type=int, default=10)
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument(
    "--out_subdir",
    type=str,
    default="diversity_skill_ee_traj",
    help="Subdirectory under <ckpt dir> to write outputs into.",
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os

import gymnasium as gym
import matplotlib
import torch
from tqdm import trange

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401, E402

from rsl_rl.runners import DiversityRunner

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
import uwlab_tasks  # noqa: F401
from uwlab_tasks.utils.hydra import hydra_task_config


SKILL_COLORS = {0: "tab:blue", 1: "tab:orange", 2: "tab:green", 3: "tab:red"}


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    agent_cfg = cli_args.sanitize_rsl_rl_cfg(agent_cfg)
    env_cfg.seed = agent_cfg.seed if args_cli.seed is None else args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # random per-env skill on reset
    skill_term = getattr(getattr(env_cfg.observations, "policy", None), "skill", None)
    if skill_term is None:
        raise ValueError("Requires a Diversity-* task with a policy.skill obs term.")
    skill_term.params["force_skill"] = -1

    resume_path = retrieve_file_path(args_cli.checkpoint)
    log_dir = os.path.dirname(resume_path)
    env_cfg.log_dir = log_dir

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    if agent_cfg.class_name != "DiversityRunner":
        raise ValueError(f"Expected DiversityRunner agent_cfg, got class_name={agent_cfg.class_name}")
    runner = DiversityRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)

    loaded = torch.load(resume_path, weights_only=False, map_location=agent_cfg.device)
    actor_only = {
        k: v for k, v in loaded["model_state_dict"].items()
        if not (k.startswith("critic.") or k.startswith("critic_obs_normalizer."))
    }
    runner.alg.policy.load_state_dict(actor_only, strict=False)
    runner.alg.policy.eval()

    policy = runner.get_inference_policy(device=env.unwrapped.device)

    robot = env.unwrapped.scene["robot"]
    ee_body_idx = robot.body_names.index("wrist_3_link")
    receptive = env.unwrapped.scene["receptive_object"]

    num_envs = env.num_envs
    num_skills = int(getattr(env.unwrapped, "diversity_num_skills"))
    progress_term = env.unwrapped.reward_manager.get_term_cfg("progress_context").func

    T = args_cli.num_steps
    num_joints = int(robot.data.joint_pos.shape[-1])

    # Capture per-term discriminator-obs slicing so the linear-probe script
    # can split the concat tensor without re-booting Isaac Sim.
    obs_mgr = env.unwrapped.observation_manager
    disc_term_dim: dict[str, int] = {}

    def _dim_of(shape):
        if hasattr(shape, "__len__"):
            return int(shape[-1])
        return int(shape)

    # isaaclab API: group_obs_term_dim is dict[group] -> list[tuple_shape] and
    # group_obs_term_names is dict[group] -> list[str], in matching order.
    term_dims_list = None
    term_names_list = None
    for attr in ("group_obs_term_dim", "_group_obs_term_dim"):
        v = getattr(obs_mgr, attr, None)
        if isinstance(v, dict) and "discriminator_obs" in v:
            term_dims_list = v["discriminator_obs"]
            break
    for attr in ("group_obs_term_names", "_group_obs_term_names"):
        v = getattr(obs_mgr, attr, None)
        if isinstance(v, dict) and "discriminator_obs" in v:
            term_names_list = v["discriminator_obs"]
            break
    if term_dims_list and term_names_list:
        for n, d in zip(term_names_list, term_dims_list):
            disc_term_dim[n] = _dim_of(d)
    elif isinstance(term_dims_list, dict):
        # alt API: dict per term
        for n, d in term_dims_list.items():
            disc_term_dim[n] = _dim_of(d)
    if not disc_term_dim:
        print("[WARN] could not discover per-term dims for discriminator_obs.")

    # Probe disc_obs dim by computing once
    _probe = obs_mgr.compute_group("discriminator_obs", update_history=False)
    if isinstance(_probe, dict):
        _probe = torch.cat([_probe[k] for k in _probe.keys()], dim=-1)
    disc_obs_dim = int(_probe.shape[-1])
    print(f"[INFO] disc_obs_dim={disc_obs_dim}, per-term dims={disc_term_dim}")

    ee_pos_buf = torch.zeros((T, num_envs, 3), dtype=torch.float32, device="cpu")
    rec_pos_buf = torch.zeros((T, num_envs, 3), dtype=torch.float32, device="cpu")
    joint_pos_buf = torch.zeros((T, num_envs, num_joints), dtype=torch.float32, device="cpu")
    disc_obs_buf = torch.zeros((T, num_envs, disc_obs_dim), dtype=torch.float32, device="cpu")
    success_buf = torch.zeros((T, num_envs), dtype=torch.bool, device="cpu")
    done_buf = torch.zeros((T, num_envs), dtype=torch.bool, device="cpu")
    active_buf = torch.zeros((T, num_envs), dtype=torch.bool, device="cpu")

    # latched per-env state — we only care about each env's FIRST episode
    first_success_step = torch.full((num_envs,), -1, dtype=torch.long, device="cpu")
    first_episode_end_step = torch.full((num_envs,), -1, dtype=torch.long, device="cpu")
    first_episode_active = torch.ones(num_envs, dtype=torch.bool, device="cpu")

    obs = env.get_observations()
    initial_skills = env.unwrapped.diversity_skill_idx.long().cpu().clone()

    print(f"[INFO] Rolling out {num_envs} envs for up to {T} steps; num_skills={num_skills}")
    for t in trange(T, desc="rollout"):
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            successes = progress_term.success.bool()

        ee_pos_buf[t] = robot.data.body_pos_w[:, ee_body_idx, :].detach().cpu()
        rec_pos_buf[t] = receptive.data.root_pos_w.detach().cpu()
        joint_pos_buf[t] = robot.data.joint_pos.detach().cpu()
        with torch.inference_mode():
            d_obs_t = obs_mgr.compute_group("discriminator_obs", update_history=False)
            if isinstance(d_obs_t, dict):
                d_obs_t = torch.cat([d_obs_t[k] for k in d_obs_t.keys()], dim=-1)
        disc_obs_buf[t] = d_obs_t.detach().cpu()
        success_buf[t] = successes.cpu()
        done_buf[t] = dones.cpu().bool()
        active_buf[t] = first_episode_active.clone()

        # latch first-success step within the first episode
        new_success = success_buf[t] & first_episode_active & (first_success_step < 0)
        first_success_step[new_success] = t

        # latch end-of-first-episode (first done that fires while still active)
        just_ended = done_buf[t] & first_episode_active
        first_episode_end_step[just_ended] = t

        # mark envs whose first episode just ended
        first_episode_active &= ~done_buf[t]

        if not first_episode_active.any():
            print(f"[INFO] All envs finished their first episode at step {t}; stopping early.")
            break

    horizon = int(t + 1)
    ee_pos_buf = ee_pos_buf[:horizon]
    rec_pos_buf = rec_pos_buf[:horizon]
    joint_pos_buf = joint_pos_buf[:horizon]
    disc_obs_buf = disc_obs_buf[:horizon]
    success_buf = success_buf[:horizon]
    done_buf = done_buf[:horizon]
    active_buf = active_buf[:horizon]

    # for envs that never ended their first episode, treat horizon as the cutoff
    no_end = first_episode_end_step < 0
    first_episode_end_step[no_end] = horizon - 1

    # ============================================================
    # Per-episode extraction
    # ============================================================
    # For each env: trajectory is steps [0, cutoff) where
    # cutoff = first_success_step if it fired, else first_episode_end_step + 1.
    # EE pos is expressed in receptive-object frame (translation only).
    episodes = []
    per_skill_count: dict[int, int] = {s: 0 for s in range(num_skills)}
    for env_id in range(num_envs):
        skill = int(initial_skills[env_id].item())
        if per_skill_count.get(skill, 0) >= args_cli.num_episodes_per_skill:
            continue
        succeeded = bool(first_success_step[env_id] >= 0)
        if succeeded:
            cutoff = int(first_success_step[env_id].item())  # exclusive of success step
        else:
            cutoff = int(first_episode_end_step[env_id].item()) + 1  # full episode
        if cutoff < 2:
            # too short to plot a path; skip
            continue
        ee_w = ee_pos_buf[:cutoff, env_id, :]  # (cutoff, 3)
        rec_w = rec_pos_buf[:cutoff, env_id, :]  # (cutoff, 3)
        ee_local = ee_w - rec_w  # translation-only alignment
        episodes.append(
            {
                "env_id": env_id,
                "skill": skill,
                "succeeded": succeeded,
                "cutoff": cutoff,
                "ee_pos_local": ee_local.clone(),
                "ee_pos_w": ee_w.clone(),
                "rec_pos_w": rec_w.clone(),
            }
        )
        per_skill_count[skill] = per_skill_count.get(skill, 0) + 1

    # ============================================================
    # Save raw + episodes
    # ============================================================
    out_dir = pathlib.Path(log_dir) / args_cli.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "raw.pt"
    episodes_path = out_dir / "episodes.pt"
    plot_path = out_dir / "plot_3d.png"

    torch.save(
        {
            "ee_pos_w": ee_pos_buf,
            "rec_pos_w": rec_pos_buf,
            "joint_pos": joint_pos_buf,
            "joint_names": list(robot.joint_names),
            "disc_obs": disc_obs_buf,
            "disc_term_dim": disc_term_dim,
            "success": success_buf,
            "done": done_buf,
            "active": active_buf,
            "first_success_step": first_success_step,
            "first_episode_end_step": first_episode_end_step,
            "initial_skills": initial_skills,
            "checkpoint": str(resume_path),
            "task": args_cli.task,
            "num_skills": num_skills,
            "horizon": horizon,
        },
        raw_path,
    )
    torch.save({"episodes": episodes, "num_skills": num_skills}, episodes_path)
    print(f"[INFO] Wrote raw rollout to:  {raw_path}")
    print(f"[INFO] Wrote episodes to:    {episodes_path}")

    print(
        "[INFO] Episode counts per skill: "
        + ", ".join(f"skill {s}: {per_skill_count.get(s, 0)}" for s in range(num_skills))
    )

    # ============================================================
    # 3D plot
    # ============================================================
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    for ep in episodes:
        traj = ep["ee_pos_local"].numpy()
        color = SKILL_COLORS.get(ep["skill"], "tab:gray")
        # solid line for successful episodes, dashed for non-success
        ls = "-" if ep["succeeded"] else "--"
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], color=color, alpha=0.7, linewidth=1.2, linestyle=ls)
        # mark starting point
        ax.scatter(traj[0, 0], traj[0, 1], traj[0, 2], color=color, s=15, marker="o")
        # mark final point
        ax.scatter(traj[-1, 0], traj[-1, 1], traj[-1, 2], color=color, s=25, marker="x")

    # legend
    from matplotlib.lines import Line2D
    legend_handles = []
    for s in range(num_skills):
        if per_skill_count.get(s, 0) > 0:
            legend_handles.append(
                Line2D([0], [0], color=SKILL_COLORS.get(s, "tab:gray"), lw=2, label=f"skill {s} ({per_skill_count[s]} eps)")
            )
    legend_handles.append(Line2D([0], [0], color="black", lw=1.5, linestyle="-", label="succeeded"))
    legend_handles.append(Line2D([0], [0], color="black", lw=1.5, linestyle="--", label="no success in 1st ep"))
    ax.legend(handles=legend_handles, loc="upper left", fontsize=9)

    ax.set_xlabel("x (m, receptive-frame)")
    ax.set_ylabel("y (m, receptive-frame)")
    ax.set_zlabel("z (m, receptive-frame)")
    n_succ = sum(1 for ep in episodes if ep["succeeded"])
    ax.set_title(
        f"DIAYN EE trajectories (pre-success), receptive-object frame\n"
        f"task={args_cli.task}\n"
        f"ckpt={pathlib.Path(resume_path).name}  ({len(episodes)} eps, {n_succ} succeeded)",
        fontsize=9,
    )

    fig.tight_layout()
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)
    print(f"[INFO] Wrote 3D plot to:     {plot_path}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
