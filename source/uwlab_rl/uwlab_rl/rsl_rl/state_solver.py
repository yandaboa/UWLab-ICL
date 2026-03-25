from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class NormalizationStats:
    """State/action normalization utilities used by CCIL solving."""

    state_mean: torch.Tensor
    state_std: torch.Tensor
    action_mean: torch.Tensor
    action_std: torch.Tensor

    def to(self, device: torch.device | str) -> "NormalizationStats":
        dev = torch.device(device)
        return NormalizationStats(
            state_mean=self.state_mean.to(dev),
            state_std=self.state_std.to(dev),
            action_mean=self.action_mean.to(dev),
            action_std=self.action_std.to(dev),
        )

    def encode_state(self, state: torch.Tensor) -> torch.Tensor:
        return (state - self.state_mean) / self.state_std

    def decode_state(self, state_n: torch.Tensor) -> torch.Tensor:
        return state_n * self.state_std + self.state_mean

    def encode_action(self, action: torch.Tensor) -> torch.Tensor:
        return (action - self.action_mean) / self.action_std

    def decode_action(self, action_n: torch.Tensor) -> torch.Tensor:
        return action_n * self.action_std + self.action_mean


def residual_fn(
    s_var_n: torch.Tensor,
    a_g: torch.Tensor,
    s_next_star: torch.Tensor,
    norms: NormalizationStats,
    dyn_model: torch.nn.Module,
    clamp_action_norm: tuple[float, float] | None = (-3.0, 3.0),
) -> torch.Tensor:
    """Compute normalized CCIL residual r_n = (s_n + delta_n) - s_next_n."""
    a_g_n = norms.encode_action(a_g)
    if clamp_action_norm is not None:
        lo, hi = clamp_action_norm
        a_g_n = a_g_n.clamp(min=float(lo), max=float(hi))
    a_g_raw = norms.decode_action(a_g_n)
    s_raw = norms.decode_state(s_var_n)
    delta_raw = dyn_model(s_raw, a_g_raw)
    delta_n = delta_raw / norms.state_std
    s_next_n = norms.encode_state(s_next_star)
    return (s_var_n + delta_n) - s_next_n


def build_keep_mask(
    s_g: torch.Tensor,
    s_star: torch.Tensor,
    r_n: torch.Tensor,
    residual_mse: torch.Tensor,
    norms: NormalizationStats,
    r_max: float,
    max_delta_s: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build boolean keep mask and predecessor-locality distance in normalized space."""
    finite_mask = torch.isfinite(s_g).all(dim=-1) & torch.isfinite(r_n).all(dim=-1)
    s_g_n = norms.encode_state(s_g)
    s_star_n = norms.encode_state(s_star)
    predecessor_dist = torch.linalg.norm(s_g_n - s_star_n, dim=-1)
    keep_mask = (
        finite_mask
        & (residual_mse < float(r_max))
        & (predecessor_dist < float(max_delta_s))
    )
    return keep_mask, predecessor_dist


def solve_predecessor_states(
    s_star: torch.Tensor,
    a_g: torch.Tensor,
    s_next_star: torch.Tensor,
    norms: NormalizationStats,
    dyn_model: torch.nn.Module,
    K: int = 10,
    lr_s: float = 0.1,
    eps_opt: float = 1.0e-3,
    r_max: float = 1.0e-2,
    max_delta_s: float = 0.5,
    grad_clip_norm: float = 1.0,
    clamp_action_norm: tuple[float, float] | None = (-3.0, 3.0),
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor | float | int]]:
    """Solve for predecessor states s_G and return keep mask + diagnostics."""
    assert K > 0, "K must be positive."
    device = s_next_star.device
    dtype = torch.float32
    s_star_f = s_star.to(device=device, dtype=dtype)
    a_g_f = a_g.to(device=device, dtype=dtype)
    s_next_f = s_next_star.to(device=device, dtype=dtype)
    norms = norms.to(device)

    old_requires_grad = [param.requires_grad for param in dyn_model.parameters()]
    dyn_model.eval()
    for param in dyn_model.parameters():
        param.requires_grad_(False)
    try:
        with torch.no_grad():
            s_next_n = norms.encode_state(s_next_f)
            a_n = norms.encode_action(a_g_f)
            if clamp_action_norm is not None:
                lo, hi = clamp_action_norm
                a_n = a_n.clamp(min=float(lo), max=float(hi))
            a_raw = norms.decode_action(a_n)
            delta_guess_raw = dyn_model(s_next_f, a_raw)
            delta_guess_n = delta_guess_raw / norms.state_std
            s_var_n = s_next_n - delta_guess_n

        steps_used = 0
        for k in range(int(K)):
            s_var_n = s_var_n.detach().requires_grad_(True)
            r_n = residual_fn(
                s_var_n=s_var_n,
                a_g=a_g_f,
                s_next_star=s_next_f,
                norms=norms,
                dyn_model=dyn_model,
                clamp_action_norm=clamp_action_norm,
            )
            per_sample_loss = (r_n * r_n).sum(dim=-1)
            loss = per_sample_loss.mean()
            loss.backward()
            grad = s_var_n.grad
            assert grad is not None
            if grad_clip_norm > 0.0:
                grad_norm = torch.linalg.norm(grad, dim=-1, keepdim=True).clamp_min(1.0e-12)
                scale = (float(grad_clip_norm) / grad_norm).clamp(max=1.0)
                grad = grad * scale
            with torch.no_grad():
                s_var_n = s_var_n - float(lr_s) * grad
            steps_used = k + 1
            if float(torch.sqrt(per_sample_loss.mean()).item()) < float(eps_opt):
                break

        with torch.no_grad():
            final_r_n = residual_fn(
                s_var_n=s_var_n,
                a_g=a_g_f,
                s_next_star=s_next_f,
                norms=norms,
                dyn_model=dyn_model,
                clamp_action_norm=clamp_action_norm,
            )
            residual_norm = torch.linalg.norm(final_r_n, dim=-1)
            residual_mse = (final_r_n * final_r_n).mean(dim=-1)
            s_g = norms.decode_state(s_var_n)
            keep_mask, predecessor_dist = build_keep_mask(
                s_g=s_g,
                s_star=s_star_f,
                r_n=final_r_n,
                residual_mse=residual_mse,
                norms=norms,
                r_max=r_max,
                max_delta_s=max_delta_s,
            )
            stats: dict[str, torch.Tensor | float | int] = {
                "steps_used": steps_used,
                "residual_norm": residual_norm,
                "residual_mse": residual_mse,
                "predecessor_distance": predecessor_dist,
                "acceptance_rate": float(keep_mask.float().mean().item()),
            }
    finally:
        for param, requires_grad in zip(dyn_model.parameters(), old_requires_grad):
            param.requires_grad_(requires_grad)
    return s_g, keep_mask, stats
