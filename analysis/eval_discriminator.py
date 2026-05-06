"""Evaluate the DIAYN discriminator's per-step skill-classification accuracy
on rollouts of the trained policy with random skills, split by whether each
step is *before* or *after* the first time the success criterion fires.

Usage (inside the isaac-sim container, lti env, single GPU):

    python analysis/eval_discriminator.py \
        --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Diversity-Play-v0 \
        --num_envs 256 --num_steps 250 --headless \
        --checkpoint logs/rsl_rl/.../model_5400.pt \
        env.scene.insertive_object=peg env.scene.receptive_object=peghole \
        'agent.policy.actor_hidden_dims=[1024,512,256,128]'

Saves a .pt with raw per-step tensors and a JSON summary into the checkpoint dir.
"""

import argparse
import json
import os
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

parser = argparse.ArgumentParser(description="Evaluate DIAYN discriminator accuracy on policy rollouts.")
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--num_steps", type=int, default=250, help="Max env steps; should exceed natural episode length.")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--out_tag", type=str, default="discriminator_eval")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from tqdm import trange

from rsl_rl.runners import DiversityRunner

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
import uwlab_tasks  # noqa: F401
from uwlab_tasks.utils.hydra import hydra_task_config


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    agent_cfg = cli_args.sanitize_rsl_rl_cfg(agent_cfg)
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # leave force_skill at -1 so resets sample uniformly at random per env
    skill_term = getattr(getattr(env_cfg.observations, "policy", None), "skill", None)
    if skill_term is None:
        raise ValueError("This script requires a Diversity-* task with a policy.skill obs term.")
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
    if "discriminator_state_dict" not in loaded:
        raise RuntimeError(f"Checkpoint {resume_path} has no discriminator_state_dict.")
    runner.alg.discriminator.load_state_dict(loaded["discriminator_state_dict"])
    runner.alg.attach_env(env)
    runner.alg.discriminator.eval()
    runner.alg.policy.eval()

    policy = runner.get_inference_policy(device=env.unwrapped.device)
    discriminator = runner.alg.discriminator

    num_envs = env.num_envs
    device = env.unwrapped.device
    num_skills = int(getattr(env.unwrapped, "diversity_num_skills"))
    progress_term = env.unwrapped.reward_manager.get_term_cfg("progress_context").func

    T = args_cli.num_steps
    true_skill_buf = torch.zeros((T, num_envs), dtype=torch.long, device="cpu")
    pred_skill_buf = torch.zeros((T, num_envs), dtype=torch.long, device="cpu")
    success_buf = torch.zeros((T, num_envs), dtype=torch.bool, device="cpu")
    active_buf = torch.zeros((T, num_envs), dtype=torch.bool, device="cpu")  # still in first episode

    first_success_step = torch.full((num_envs,), -1, dtype=torch.long, device="cpu")
    first_episode_active = torch.ones(num_envs, dtype=torch.bool, device="cpu")

    obs = env.get_observations()

    print(f"[INFO] Rolling out {num_envs} envs for up to {T} steps; num_skills={num_skills}")
    for t in trange(T, desc="rollout"):
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)

            # discriminator obs group
            obs_dict = env.unwrapped.observation_manager.compute()
            disc_obs = obs_dict["discriminator_obs"]
            logits = discriminator(disc_obs)
            preds = logits.argmax(dim=-1)

            true_skills = env.unwrapped.diversity_skill_idx.long()
            successes = progress_term.success.bool()

        true_skill_buf[t] = true_skills.cpu()
        pred_skill_buf[t] = preds.cpu()
        success_buf[t] = successes.cpu()
        active_buf[t] = first_episode_active.clone()

        # latch first-success step (only within the first episode)
        new_success = successes.cpu() & first_episode_active & (first_success_step < 0)
        first_success_step[new_success] = t

        # mark envs whose first episode just ended
        first_episode_active &= ~dones.cpu().bool()

        if not first_episode_active.any():
            print(f"[INFO] All envs finished their first episode at step {t}; stopping early.")
            break

    # truncate buffers to actual horizon used
    horizon = (t + 1) if 't' in dir() else T
    true_skill_buf = true_skill_buf[:horizon]
    pred_skill_buf = pred_skill_buf[:horizon]
    success_buf = success_buf[:horizon]
    active_buf = active_buf[:horizon]

    correct = (pred_skill_buf == true_skill_buf)

    # per-step "ever-succeeded by step t" flag, computed within the first episode only
    ever_success = torch.zeros_like(success_buf)
    cum = torch.zeros(num_envs, dtype=torch.bool)
    for t in range(horizon):
        cum |= (success_buf[t] & active_buf[t])
        ever_success[t] = cum

    pre_mask = active_buf & ~ever_success
    post_mask = active_buf & ever_success

    pre_n = int(pre_mask.sum().item())
    post_n = int(post_mask.sum().item())
    pre_correct = int((correct & pre_mask).sum().item())
    post_correct = int((correct & post_mask).sum().item())
    overall_correct = int((correct & active_buf).sum().item())
    overall_n = int(active_buf.sum().item())

    chance = 1.0 / num_skills
    summary = {
        "checkpoint": str(resume_path),
        "task": args_cli.task,
        "num_envs": num_envs,
        "num_skills": num_skills,
        "horizon": horizon,
        "num_episodes_with_success": int((first_success_step >= 0).sum().item()),
        "num_episodes_total": num_envs,
        "first_success_step_mean": float(first_success_step[first_success_step >= 0].float().mean().item())
        if (first_success_step >= 0).any() else None,
        "first_success_step_median": float(first_success_step[first_success_step >= 0].float().median().item())
        if (first_success_step >= 0).any() else None,
        "pre_success_steps": pre_n,
        "post_success_steps": post_n,
        "overall_steps": overall_n,
        "pre_success_accuracy": pre_correct / max(pre_n, 1),
        "post_success_accuracy": post_correct / max(post_n, 1),
        "overall_accuracy": overall_correct / max(overall_n, 1),
        "chance_accuracy": chance,
    }

    print("\n=== Discriminator-on-rollout summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    out_dir = pathlib.Path(log_dir) / "discriminator_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args_cli.out_tag
    raw_path = out_dir / f"{tag}_raw.pt"
    summary_path = out_dir / f"{tag}_summary.json"
    torch.save(
        {
            "true_skill": true_skill_buf,
            "pred_skill": pred_skill_buf,
            "success": success_buf,
            "active": active_buf,
            "first_success_step": first_success_step,
            "checkpoint": str(resume_path),
            "task": args_cli.task,
            "num_skills": num_skills,
        },
        raw_path,
    )
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[INFO] Wrote raw rollout to: {raw_path}")
    print(f"[INFO] Wrote summary to:     {summary_path}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
