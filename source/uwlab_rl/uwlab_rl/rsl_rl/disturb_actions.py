from __future__ import annotations

import torch


def sample_disturbed_action(
    a_star: torch.Tensor,
    sigma_action: float,
    clamp_spec: tuple[float, float] | None = None,
) -> torch.Tensor:
    """Sample disturbed actions with optional clamping."""
    if sigma_action < 0.0:
        raise ValueError("sigma_action must be non-negative.")
    noise = torch.randn_like(a_star) * float(sigma_action)
    a_g = a_star + noise
    if clamp_spec is not None:
        lo, hi = clamp_spec
        a_g = a_g.clamp(min=float(lo), max=float(hi))
    return a_g
