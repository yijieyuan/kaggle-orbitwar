"""Orbit-phase helper used by the movement forecaster."""
from __future__ import annotations

from torch import Tensor


def orbit_phase_index_from_obs_step(obs_step: Tensor) -> Tensor:
    """Convert the observation ``step`` counter into the engine orbit phase index.

    Orbiting planets update with ``theta = orb_a0 + angvel * g_step`` *before*
    ``g_step`` is incremented for the next observation. The public observation
    carries ``step == g_step`` after that increment, so the implied phase index
    is ``max(0, step - 1)`` (and ``0`` at game start when ``step == 0``).
    """
    s = obs_step.float()
    return (s - (s > 0).to(s.dtype)).clamp(min=0.0)
