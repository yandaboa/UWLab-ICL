from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import imageio
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _clone_obs_tree(value: Any) -> Any:
    """Recursively clone observation tensors into plain Python containers."""
    if isinstance(value, torch.Tensor):
        return value.clone()
    if hasattr(value, "items"):
        return {key: _clone_obs_tree(item) for key, item in value.items()}
    return value


def _clone_hidden_state(hidden_state: Any) -> Any:
    if hidden_state is None:
        return None
    if isinstance(hidden_state, tuple):
        return tuple(_clone_hidden_state(value) for value in hidden_state)
    if isinstance(hidden_state, torch.Tensor):
        return hidden_state.clone()
    return hidden_state


class PrivilegedPolicyDebugger:
    """Sweeps privileged observation values and plots the resulting action distributions."""

    def __init__(
        self,
        policy_module: Any,
        output_dir: str | Path,
        sweep_key: str = "friction",
        sweep_range: tuple[float, float] | None = None,
        num_sweep_points: int = 9,
        include_joint_insertive_receptive: bool = False,
    ) -> None:
        self._policy_module = policy_module
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._sweep_key = sweep_key.strip().lower()
        if self._sweep_key not in {"friction", "restitution"}:
            raise ValueError(
                f"Unsupported privileged debug sweep key '{sweep_key}'. Supported values are 'friction' and 'restitution'."
            )
        self._sweep_range = sweep_range
        self._num_sweep_points = num_sweep_points
        self._include_joint_insertive_receptive = include_joint_insertive_receptive
        self._sweep_values_cache: dict[str, np.ndarray] = {}
        self._time_series_history: dict[str, dict[str, Any]] = {}
        print(f"[INFO] Saving privileged debug plots to: {self._output_dir}")
        print(f"[INFO] Sweeping privileged material property: {self._sweep_key}")
        if self._include_joint_insertive_receptive:
            print("[INFO] Adding joint insertive/receptive privileged sweep.")

    def plot_step(self, obs: Any, step_idx: int) -> None:
        policy_obs = self._get_policy_obs(obs)
        sweep_targets = self._resolve_sweep_targets(policy_obs)
        for sweep_target in sweep_targets:
            stats = self._compute_action_sweep(obs, sweep_target)
            self._save_plot(stats=stats, step_idx=step_idx)
            self._record_time_series(stats=stats, step_idx=step_idx)

    def _get_policy_obs(self, obs: Any) -> Any:
        if "policy" not in obs:
            raise ValueError("Privileged debug plotting expects a 'policy' observation group.")
        policy_obs = obs["policy"]
        if not hasattr(policy_obs, "items"):
            raise ValueError("Privileged debug plotting expects obs['policy'] to be a keyed observation mapping.")
        return policy_obs

    def _resolve_sweep_targets(self, policy_obs: Any) -> list[dict[str, Any]]:
        candidate_keys = [key for key in policy_obs.keys() if key.endswith("_material_properties")]
        if not candidate_keys:
            raise ValueError("No material property observation terms found under obs['policy'].")

        targets = []
        # targets += [self._target_for_term(term_key, policy_obs[term_key]) for term_key in candidate_keys]
        if self._include_joint_insertive_receptive:
            insertive_key = "insertive_object_material_properties"
            receptive_key = "receptive_object_material_properties"
            if insertive_key not in policy_obs or receptive_key not in policy_obs:
                raise KeyError(
                    "Joint insertive/receptive sweep requested, but required material property terms are missing "
                    f"from obs['policy']. Available keys: {list(policy_obs.keys())}"
                )
            targets.append(self._joint_target_for_terms([insertive_key, receptive_key], policy_obs))
        return targets

    def _get_num_components(self, obs_value: torch.Tensor) -> int:
        if not isinstance(obs_value, torch.Tensor):
            raise TypeError(f"Observation term must be a tensor, got: {type(obs_value)}")
        if obs_value.shape[0] != 1:
            raise ValueError(
                f"Privileged debug plotting requires a single environment, got tensor shape {obs_value.shape}."
            )
        return int(obs_value[0].numel())

    def _normalize_component_index(self, component_idx: int, num_components: int) -> int:
        normalized_idx = component_idx if component_idx >= 0 else num_components + component_idx
        if normalized_idx < 0 or normalized_idx >= num_components:
            raise IndexError(
                f"Component index {component_idx} is out of range for observation with {num_components} values."
            )
        return normalized_idx

    def _target_for_term(self, term_key: str, obs_value: torch.Tensor) -> dict[str, Any]:
        num_components = self._get_num_components(obs_value)
        if num_components < 3:
            raise ValueError(
                f"Observation term '{term_key}' must have material layout "
                "[static_friction, dynamic_friction, restitution], got {num_components} values."
            )
        component_indices = [0, 1] if self._sweep_key == "friction" else [2]
        return {
            "target_name": term_key,
            "term_keys": [term_key],
            "component_indices": component_indices,
            "component_label": self._sweep_key,
        }

    def _joint_target_for_terms(self, term_keys: list[str], policy_obs: Any) -> dict[str, Any]:
        for term_key in term_keys:
            obs_value = policy_obs[term_key]
            num_components = self._get_num_components(obs_value)
            if num_components < 3:
                raise ValueError(
                    f"Observation term '{term_key}' must have material layout "
                    "[static_friction, dynamic_friction, restitution]."
                )
        return {
            "target_name": "insertive_and_receptive_object_material_properties",
            "term_keys": term_keys,
            "component_indices": [0, 1] if self._sweep_key == "friction" else [2],
            "component_label": self._sweep_key,
        }

    def _target_history_key(self, target_name: str, component_label: str) -> str:
        return f"{component_label}:{target_name}"

    def _record_time_series(self, stats: dict[str, Any], step_idx: int) -> None:
        history_key = self._target_history_key(stats["target_name"], stats["component_label"])
        history = self._time_series_history.setdefault(
            history_key,
            {
                "target_name": stats["target_name"],
                "term_keys": list(stats["term_keys"]),
                "component_label": stats["component_label"],
                "component_indices": list(stats["component_indices"]),
                "sweep_values": stats["sweep_values"].clone(),
                "times": [],
                "action_mean": [],
            },
        )
        history["times"].append(step_idx)
        history["action_mean"].append(stats["action_mean"].clone())

    def _build_sweep_values(self, current_value: torch.Tensor, history_key: str) -> torch.Tensor:
        if history_key not in self._sweep_values_cache:
            if self._sweep_range is not None:
                sweep_min, sweep_max = self._sweep_range
            else:
                center = float(current_value.detach().cpu().item())
                scale = max(abs(center), 1.0)
                sweep_min = center - 0.5 * scale
                sweep_max = center + 0.5 * scale
            self._sweep_values_cache[history_key] = np.linspace(sweep_min, sweep_max, num=self._num_sweep_points, dtype=np.float32)
        return torch.as_tensor(self._sweep_values_cache[history_key], dtype=current_value.dtype, device=current_value.device)

    def _compute_action_sweep(self, obs: Any, sweep_target: dict[str, Any]) -> dict[str, Any]:
        base_obs = _clone_obs_tree(obs)
        policy_obs = self._get_policy_obs(base_obs)
        term_keys = sweep_target["term_keys"]
        target_name = sweep_target["target_name"]
        current_values = []
        normalized_component_indices = None
        for term_key in term_keys:
            if term_key not in policy_obs:
                raise KeyError(f"Observation term '{term_key}' not found in obs['policy'].")
            obs_value = policy_obs[term_key]
            if not isinstance(obs_value, torch.Tensor):
                raise TypeError(f"Observation term '{term_key}' must be a tensor, got: {type(obs_value)}")
            flattened_obs_value = obs_value.view(obs_value.shape[0], -1)
            if flattened_obs_value.shape[-1] < 1:
                raise ValueError(f"Observation term '{term_key}' must contain at least one scalar value.")
            component_indices = [
                self._normalize_component_index(component_idx, flattened_obs_value.shape[-1])
                for component_idx in sweep_target["component_indices"]
            ]
            if normalized_component_indices is None:
                normalized_component_indices = component_indices
            current_values.append(flattened_obs_value[0, component_indices])
        if normalized_component_indices is None:
            raise ValueError("Expected at least one privileged debug target term.")
        current_values_tensor = torch.cat(current_values, dim=0)
        history_key = self._target_history_key(target_name, sweep_target["component_label"])
        sweep_values = self._build_sweep_values(current_values_tensor.mean(), history_key)

        actor_state_snapshot = self._snapshot_actor_hidden_state()
        action_means = []
        action_stds = []
        for sweep_value in sweep_values:
            probe_obs = _clone_obs_tree(base_obs)
            for term_key in term_keys:
                probe_obs["policy"][term_key].view(probe_obs["policy"][term_key].shape[0], -1)[
                    0, normalized_component_indices
                ] = sweep_value
            self._restore_actor_hidden_state(actor_state_snapshot)
            self._policy_module.act(probe_obs)
            action_means.append(self._policy_module.action_mean[0].detach().cpu())
            action_stds.append(self._policy_module.action_std[0].detach().cpu())
        self._restore_actor_hidden_state(actor_state_snapshot)

        return {
            "target_name": target_name,
            "term_keys": term_keys,
            "component_indices": normalized_component_indices,
            "component_label": sweep_target["component_label"],
            "actual_values": current_values_tensor.detach().cpu(),
            "sweep_values": sweep_values.detach().cpu(),
            "action_mean": torch.stack(action_means, dim=0),
            "action_std": torch.stack(action_stds, dim=0),
        }

    def _snapshot_actor_hidden_state(self) -> tuple[str, Any] | None:
        if hasattr(self._policy_module, "memory_a"):
            return ("memory_a", _clone_hidden_state(self._policy_module.memory_a.hidden_state))
        if hasattr(self._policy_module, "memory_s"):
            return ("memory_s", _clone_hidden_state(self._policy_module.memory_s.hidden_state))
        return None

    def _restore_actor_hidden_state(self, snapshot: tuple[str, Any] | None) -> None:
        if snapshot is None:
            return
        attr_name, hidden_state = snapshot
        getattr(self._policy_module, attr_name).hidden_state = _clone_hidden_state(hidden_state)

    def _save_plot(self, stats: dict[str, Any], step_idx: int) -> None:
        action_mean = stats["action_mean"].numpy()
        action_std = np.clip(stats["action_std"].numpy(), a_min=1.0e-4, a_max=None)
        sweep_values = stats["sweep_values"].numpy()
        actual_values = stats["actual_values"].numpy()
        target_name = stats["target_name"]
        component_label = stats["component_label"]

        action_dim = action_mean.shape[-1]
        ncols = 2
        nrows = int(np.ceil(action_dim / ncols))
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 4 * nrows), squeeze=False)
        colors = matplotlib.colormaps["viridis"](np.linspace(0.0, 1.0, len(sweep_values)))
        legend_handles = []

        for action_idx in range(action_dim):
            axis = axes[action_idx // ncols][action_idx % ncols]
            mu = action_mean[:, action_idx]
            sigma = action_std[:, action_idx]
            x_min = float(np.min(mu - 4.0 * sigma))
            x_max = float(np.max(mu + 4.0 * sigma))
            if np.isclose(x_min, x_max):
                x_min -= 1.0
                x_max += 1.0
            x_grid = np.linspace(x_min, x_max, 400)
            for curve_idx, restitution_value in enumerate(sweep_values):
                pdf = np.exp(-0.5 * ((x_grid - mu[curve_idx]) / sigma[curve_idx]) ** 2)
                pdf /= sigma[curve_idx] * np.sqrt(2.0 * np.pi)
                (line,) = axis.plot(
                    x_grid,
                    pdf,
                    color=colors[curve_idx],
                    alpha=0.85,
                    linewidth=1.5,
                    label=f"v={restitution_value:.3f}",
                )
                if action_idx == 0:
                    legend_handles.append(line)
            axis.set_title(f"Action {action_idx}")
            axis.set_xlabel("Action value")
            axis.set_ylabel("Density")
            axis.grid(True, alpha=0.3)

        for action_idx in range(action_dim, nrows * ncols):
            axes[action_idx // ncols][action_idx % ncols].axis("off")

        fig.suptitle(
            f"{target_name}[{component_label}] sweep at step {step_idx}\n"
            f"actual values={np.array2string(actual_values, precision=3, separator=', ')}",
            fontsize=14,
        )
        fig.legend(
            handles=legend_handles,
            loc="center right",
            bbox_to_anchor=(0.995, 0.5),
            title="Counterfactual value",
        )
        fig.tight_layout(rect=(0.0, 0.0, 0.9, 0.95))

        save_dir = self._output_dir / component_label / target_name
        save_dir.mkdir(parents=True, exist_ok=True)
        figure_path = save_dir / f"step_{step_idx:06d}.png"
        fig.savefig(figure_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    def save_time_series_videos(self, rollout_step_dt: float, target_num_steps: int | None = None) -> None:
        if not self._time_series_history:
            return
        for history_key, history in self._time_series_history.items():
            self._save_time_series_video(
                history_key=history_key,
                history=history,
                rollout_step_dt=rollout_step_dt,
                target_num_steps=target_num_steps,
            )

    def _save_time_series_video(
        self,
        history_key: str,
        history: dict[str, Any],
        rollout_step_dt: float,
        target_num_steps: int | None = None,
    ) -> None:
        action_mean = torch.stack(history["action_mean"], dim=0).numpy()
        time_values = np.asarray(history["times"], dtype=np.float32) * rollout_step_dt
        sweep_values = history["sweep_values"].numpy()
        target_name = history["target_name"]
        component_label = history["component_label"]

        num_frames = action_mean.shape[0]
        if num_frames == 0:
            return
        total_steps = target_num_steps if target_num_steps is not None else num_frames
        total_duration = max(total_steps * rollout_step_dt, rollout_step_dt)
        fps = num_frames / total_duration

        action_dim = action_mean.shape[-1]
        ncols = 2
        nrows = int(np.ceil(action_dim / ncols))
        colors = matplotlib.colormaps["viridis"](np.linspace(0.0, 1.0, len(sweep_values)))

        y_min = np.min(action_mean, axis=(0, 1))
        y_max = np.max(action_mean, axis=(0, 1))
        y_pad = np.maximum(0.05 * (y_max - y_min), 1.0e-3)

        frames = []
        save_dir = self._output_dir / component_label / target_name
        save_dir.mkdir(parents=True, exist_ok=True)
        video_path = save_dir / "action_mean_over_time.mp4"

        for frame_idx in range(num_frames):
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 4 * nrows), squeeze=False)
            legend_handles = []
            for action_idx in range(action_dim):
                axis = axes[action_idx // ncols][action_idx % ncols]
                for sweep_idx, sweep_value in enumerate(sweep_values):
                    (line,) = axis.plot(
                        time_values,
                        action_mean[:, sweep_idx, action_idx],
                        color=colors[sweep_idx],
                        alpha=0.85,
                        linewidth=1.5,
                        label=f"v={sweep_value:.3f}",
                    )
                    axis.scatter(
                        time_values[frame_idx],
                        action_mean[frame_idx, sweep_idx, action_idx],
                        color=colors[sweep_idx],
                        edgecolors="black",
                        linewidths=0.5,
                        s=50,
                        zorder=3,
                    )
                    if action_idx == 0:
                        legend_handles.append(line)
                axis.set_title(f"Action {action_idx}")
                axis.set_xlabel("Time (s)")
                axis.set_ylabel("Action mean")
                axis.set_xlim(0.0, total_duration)
                axis.set_ylim(y_min[action_idx] - y_pad[action_idx], y_max[action_idx] + y_pad[action_idx])
                axis.grid(True, alpha=0.3)

            for action_idx in range(action_dim, nrows * ncols):
                axes[action_idx // ncols][action_idx % ncols].axis("off")

            fig.suptitle(
                f"{history_key} action means over time\n"
                f"t={time_values[frame_idx]:.3f}s / {total_duration:.3f}s",
                fontsize=14,
            )
            fig.legend(
                handles=legend_handles,
                loc="center right",
                bbox_to_anchor=(0.995, 0.5),
                title="Counterfactual value",
            )
            fig.tight_layout(rect=(0.0, 0.0, 0.9, 0.95))
            buffer = io.BytesIO()
            fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
            buffer.seek(0)
            frame = (plt.imread(buffer)[..., :3] * 255).astype(np.uint8)
            frames.append(frame)
            plt.close(fig)

        imageio.mimsave(str(video_path), frames, fps=fps, codec="libx264")
        print(f"[INFO] Saved privileged debug time-series video to: {video_path}")
