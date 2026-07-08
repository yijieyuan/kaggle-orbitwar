"""Geometry primitives. Pure tensor functions with no game-state imports."""

from __future__ import annotations

import torch
from torch import Tensor

from .constants import MAX_SHIP_SPEED

# Pre-compute log(1000) once as a plain Python float for efficiency.
_LOG_1000: float = float(torch.log(torch.tensor(1000.0)).item())
_FLEET_SPEED_LUT_MAX: int = 400


def _fleet_speed_formula(ships: Tensor) -> Tensor:
    """Exact engine-matching speed formula."""
    ratio = (torch.log(ships) / _LOG_1000).clamp(max=1.0)
    return 1.0 + (MAX_SHIP_SPEED - 1.0) * ratio.pow(1.5)


def _build_fleet_speed_lut(max_ships: int) -> Tensor:
    # Index 0 is unused but keeps indexing branch-free for ships >= 1.
    idx = torch.arange(max_ships + 1, dtype=torch.float32).clamp(min=1.0)
    return _fleet_speed_formula(idx)


_FLEET_SPEED_LUT: Tensor = _build_fleet_speed_lut(_FLEET_SPEED_LUT_MAX)
# Per-(device, dtype) cache of the LUT so a CUDA stream isn't synced by an
# H→D copy on every fleet_speed call. Module-level dict, populated lazily.
_FLEET_SPEED_LUT_CACHE: dict[tuple, Tensor] = {}


def _fleet_speed_lut_on(device: torch.device, dtype: torch.dtype) -> Tensor:
    key = (device, dtype)
    cached = _FLEET_SPEED_LUT_CACHE.get(key)
    if cached is None:
        cached = _FLEET_SPEED_LUT.to(device=device, dtype=dtype)
        _FLEET_SPEED_LUT_CACHE[key] = cached
    return cached


# ---------------------------------------------------------------------------
# Pairwise operations  [N] × [M]  →  [N, M]
# ---------------------------------------------------------------------------







# ---------------------------------------------------------------------------
# Fleet physics
# ---------------------------------------------------------------------------

def fleet_speed(ships: Tensor) -> Tensor:
    """Travel speed for a fleet of ``ships`` ships.

    The engine ship-speed formula::

        speed = 1 + (MAX_SHIP_SPEED - 1) * (log(ships) / log(1000)) ** 1.5

    Args:
        ships: ship count, any shape; values are clamped to ≥ 1.

    Returns:
        speed in ``[1, MAX_SHIP_SPEED]``, same shape as ``ships``.
    """
    s = ships.clamp(min=1.0)
    s_lut = s.clamp(max=float(_FLEET_SPEED_LUT_MAX))
    lo = torch.floor(s_lut).long()
    hi = torch.ceil(s_lut).long()
    frac = s_lut - lo.to(dtype=s.dtype)

    lut = _fleet_speed_lut_on(s.device, s.dtype)
    speed = lut[lo] + (lut[hi] - lut[lo]) * frac

    # Over-range fleets (>``_FLEET_SPEED_LUT_MAX`` ships) use the exact
    # formula. We unconditionally compute it and select via ``torch.where``
    # rather than a ``bool(over.any())`` branch — the latter triggers a
    # host/device sync per call on CUDA which dominated the wall-clock
    # of every kernel that batches fleet_speed inside its inner loop.
    over = s > float(_FLEET_SPEED_LUT_MAX)
    speed_formula = _fleet_speed_formula(s)
    return torch.where(over, speed_formula, speed)






# ---------------------------------------------------------------------------
# Segment–circle intersection (sun / planet collision geometry)
# ---------------------------------------------------------------------------





