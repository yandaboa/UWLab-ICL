# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Workspace layout

This directory is **not a git repo** — it's a workspace containing several sibling repos for fast in-context adaptation research, plus a couple of shared shell scripts. Each subdir is its own git checkout. Pick the right one before doing anything:

| Subdir | Owner | What it's for |
|---|---|---|
| `UWLab-ICL/` | the user | **Main workhouse repo for in-context adaptation (ASTEROID / DAgger).** Most code edits land here. Has its own detailed `CLAUDE.md`. |
| `UWLab-patrick-private/` | collaborator Patrick | Depth training and multi-task experiments. Branch `pat/distillation` adds teacher→student DAgger for vision tasks. Has its own `CLAUDE.md` (`DEPTH.MD` is the runbook). |
| `UWLab/` | collaborator Sriyash | POMDP / exploration code. No CLAUDE.md — read `README.md` if needed. |
| `OctiLab/` | (legacy) | Outdated version of the codebase the user previously worked on, with other Algorithm Distillation style papers. Reference only — don't make changes here unless explicitly asked. |
| `diffusion_policy/` | (vendored, the user's working clone) | Original Chi et al. Diffusion Policy repo (model training code etc.). The user uses this clone specifically for the **real-world setup and teleop code** (UR5e robot, real-robot eval, demo collection), not the per-repo bundled `diffusion_policy/` forks inside the Isaac Sim repos. See `README_ur5e.md` and `*_real_robot.py` entrypoints. |
| `dm_control/` | (vendored) | DeepMind Control Suite, used for trivial few-shot / point-mass experiments. Read-only fork. |

When the user says "UWLab" without a qualifier, they mean `UWLab-ICL/` (their main repo). "Patrick's repo" → `UWLab-patrick-private/`. "Sriyash's repo" or POMDP/exploration work → `UWLab/`.

**Always start work by `cd`ing into the correct subdir and reading its `CLAUDE.md` (if present).** The per-repo CLAUDE.md files cover commands, env setup, architecture, and gotchas — don't duplicate or override their guidance from this file.

## Shared workspace scripts

Three helper scripts live at the workspace root and are shared by the Isaac-Sim-based repos (`UWLab-ICL`, `UWLab-patrick-private`, `UWLab`, `OctiLab`):

- `isaac-start.sh` — launches/joins the `isaac-sim:5.1.0` Docker container. Mounts `/home/ubuntu` and `/mnt/storage`, forwards `WANDB_API_KEY` from `~/.netrc`, lands you in `/mnt/storage/lti/UWLab-ICL` by default. Run this **before** any Isaac Sim work.
- `activate_conda.sh [env]` — inside the container, activates the `lti` conda env (default) and re-prepends the env's site-packages to `PYTHONPATH` so Isaac Sim's pip_prebundle doesn't shadow our numpy/numba pins.
- `activate_patlab.sh [env]` — same, but defaults to the `patlab` conda env (used for Patrick's repo).

Canonical entry sequence:
```bash
bash isaac-start.sh                        # docker exec into isaac-sim
source activate_conda.sh                   # for UWLab-ICL, UWLab, or OctiLab
# OR
source activate_patlab.sh                  # for UWLab-patrick-private
```

The `dm_control` repo does **not** need Isaac Sim — it runs on host or in a separate env.

## Other workspace artifacts

- `model_2800.pt`, `rsl_rl/`, `videos/` at the workspace root are loose artifacts from past experiments — not authoritative source. Don't treat them as part of any repo.
- `dm_control.code-workspace` is a VS Code multi-root workspace pointing at `dm_control/` and `UWLab-ICL/`.

## Cross-repo conventions

- The Isaac-Sim repos share a similar layout (`source/uwlab*`, `scripts/`, `scripts_v2/`, `rsl_rl/`, `diffusion_policy/`, `pyproject.toml`, `uwlab.sh` wrapper). Patterns and commands often port across, but **task IDs, agent configs, and pipeline scripts diverge** — always confirm against the active repo's CLAUDE.md before reusing a command from a different repo.
- `logs/`, `outputs/`, `wandb/`, `videos/` are gitignored in every repo; checkpoints written there are local-only.

## Note on this file

The canonical copy of this workspace-level guide lives in `UWLab-ICL/WORKSPACE_CLAUDE.md` (so it's tracked by git). The `/mnt/storage/lti/CLAUDE.md` that Claude Code reads is a symlink to that file.
