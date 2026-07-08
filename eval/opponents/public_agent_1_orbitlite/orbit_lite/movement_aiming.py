"""Aiming helpers backed by :class:`PlanetMovement`.

This module is intentionally small: it solves angle/ETA for concrete
``(source_slot, target_slot, fleet_size)`` candidates using cached future planet
positions, then masks candidates whose straight path crosses the sun or another
planet.
"""

from __future__ import annotations

import torch
from torch import Tensor

from .constants import SUN_RADIUS

LAUNCH_SURFACE_OFFSET: float = 0.1
"""Fleet launch offset from source surface.

Matches Kaggle Orbit Wars engine launch placement:
``start = source + unit(angle) * (source_radius + 0.1)``.
"""

TARGET_HIT_SURFACE_OFFSET: float = 0.0
"""Extra target-surface margin for hit ETA estimation.

Kaggle/local engines register fleet-vs-planet contact when the fleet center
enters the target planet radius. ``0.0`` keeps ETA aligned with that rule.
"""

KAGGLE_SUN_RADIUS: float = SUN_RADIUS
"""Sun collision radius used by Kaggle/local engines."""











def _swept_pair_hit_mask(
    ax: Tensor,
    ay: Tensor,
    bx: Tensor,
    by: Tensor,
    p0x: Tensor,
    p0y: Tensor,
    p1x: Tensor,
    p1y: Tensor,
    r: Tensor,
) -> Tensor:
    d0x = ax - p0x
    d0y = ay - p0y
    dvx = (bx - ax) - (p1x - p0x)
    dvy = (by - ay) - (p1y - p0y)
    a = dvx * dvx + dvy * dvy
    b = 2.0 * (d0x * dvx + d0y * dvy)
    c = d0x * d0x + d0y * d0y - r * r
    near_static = a < 1e-12
    c_hit = c <= 0.0
    disc = b * b - 4.0 * a * c
    has_root = disc >= 0.0
    safe_a = torch.where(near_static, torch.ones_like(a), a)
    sq = torch.sqrt(torch.clamp(disc, min=0.0))
    t1 = (-b - sq) / (2.0 * safe_a)
    t2 = (-b + sq) / (2.0 * safe_a)
    quad_hit = has_root & (t2 >= 0.0) & (t1 <= 1.0)
    return torch.where(near_static, c_hit, quad_hit)
