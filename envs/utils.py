"""Shared utilities for POLARIS environment implementations."""
from __future__ import annotations

import torch

# Minimum center-to-center clearance for randomly-placed objects on the table.
# Slightly below the 0.06 m grid spacing used in PushOWallObstaclesEnv so that
# wall cubes remain the tightest packing while freely-placed objects stay separable.
MIN_OBSTACLE_CLEARANCE: float = 0.055


def sample_nonoverlapping_xy(
    b: int,
    half_extent: float,
    forbidden: list[torch.Tensor],
    min_clearance: float,
    device,
    max_attempts: int = 100,
) -> torch.Tensor:
    """Sample (b, 2) XY positions inside [-half_extent, half_extent]^2.

    Each sampled position is at least min_clearance away (Euclidean) from every
    tensor in forbidden.  forbidden is a list of (b, 2) tensors giving the XY
    centres of already-placed objects.

    Falls back to the negation of the last forbidden centre (clamped to the
    sampling box) if no valid position is found within max_attempts tries.
    """
    xy = torch.zeros((b, 2), device=device)
    for env_i in range(b):
        for _ in range(max_attempts):
            candidate = (torch.rand(2, device=device) * 2 - 1) * half_extent
            if all(
                torch.norm(candidate - f[env_i]) >= min_clearance
                for f in forbidden
            ):
                xy[env_i] = candidate
                break
        else:
            ref = forbidden[-1][env_i] if forbidden else torch.zeros(2, device=device)
            xy[env_i] = (-ref).clamp(-half_extent, half_extent)
    return xy
