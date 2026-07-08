"""Cross-k distance cache for the movement-backed planner.

Entry ``cross_dist[k, s, t]`` is the Euclidean distance from planet ``s`` at step
0 to planet ``t`` at step ``k`` — the *cross-time* distance a fleet must travel if
it launches now from ``s`` to intercept ``t`` at time ``k``. For static planets
this equals same-step pairwise distance; for orbiting sources the cross-time form
is the geometrically correct quantity for fleet-intercept feasibility. A
precomputed ``[K+1, P, P]`` window gives exact per-step lookups for free.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .movement import PlanetMovement


@dataclass
class DistanceCache:
    """Per-turn cross-k distance window.

    Tensor shapes:
    - ``cross_dist``: ``[K+1, P, P]`` -- ``[k, s, t] = dist(s@0, t@k)``.
    - ``alive_by_step``: ``[K+1, P]`` -- view sliced from
      ``movement.alive_by_step``.
    """

    cross_dist: Tensor
    alive_by_step: Tensor
    K: int

    @property
    def P(self) -> int:
        return int(self.cross_dist.shape[-1])

    @property
    def device(self) -> torch.device:
        return self.cross_dist.device

    @property
    def dtype(self) -> torch.dtype:
        return self.cross_dist.dtype



def build_distance_cache(
    movement: PlanetMovement,
    *,
    max_k: int,
) -> DistanceCache:
    """Build a fresh cross-k distance cache from the rolling movement cache.

    ``max_k`` is clamped to ``movement.movement_horizon``. Caller is
    expected to clamp its own k queries the same way.
    """
    K = max(0, min(int(max_k), int(movement.movement_horizon)))
    P = int(movement.P)
    src_x0 = movement.x[0]                         # [P]
    src_y0 = movement.y[0]
    tgt_x = movement.x[: K + 1]                    # [K+1, P]
    tgt_y = movement.y[: K + 1]
    # cross[k, s, t] = dist(s@0, t@k)
    dx = src_x0.view(1, P, 1) - tgt_x.unsqueeze(1)
    dy = src_y0.view(1, P, 1) - tgt_y.unsqueeze(1)
    cross_dist = torch.sqrt((dx * dx + dy * dy).clamp(min=0.0))
    alive_by_step = movement.alive_by_step[: K + 1]
    return DistanceCache(
        cross_dist=cross_dist,
        alive_by_step=alive_by_step,
        K=K,
    )


# ---------------------------------------------------------------------------
# Min-distance helper (replaces movement_min_distance_to_targets)
# ---------------------------------------------------------------------------


def min_distance_to_targets(
    cache: DistanceCache,
    source_mask: Tensor,
    target_mask: Tensor,
    *,
    max_k: int,
) -> Tensor:
    """Return per-target nearest-source distance using cross-k lookups.

    For each target ``t``, return the smallest
    ``dist(s@0, t@k)`` over alive valid sources ``s`` and steps
    ``k in [1, min(max_k, cache.K)]``. This is the exact analogue of
    ``movement_min_distance_to_targets`` with sampled steps replaced by the
    full integer range.
    """
    if source_mask.shape[-1] != cache.P or target_mask.shape[-1] != cache.P:
        raise ValueError("source_mask and target_mask must have shape [P]")
    K = max(0, min(int(max_k), int(cache.K)))
    if K <= 0:
        return torch.zeros(cache.P, dtype=cache.dtype, device=cache.device)
    # Clone the cross-k slice so we can ``masked_fill_`` invalid entries to +inf
    # without touching the cache's storage. The union of the three masks is
    # equivalent to ``~valid_pair = ~src_mask | ~tgt_mask | ~alive_at_k``.
    cross = cache.cross_dist[1 : K + 1].clone()    # [K, P_src, P_tgt]
    alive_steps = cache.alive_by_step[1 : K + 1]   # [K, P]
    src_mask = source_mask.to(device=cache.device, dtype=torch.bool)
    tgt_mask = target_mask.to(device=cache.device, dtype=torch.bool)
    inf_v = float("inf")
    cross.masked_fill_(~alive_steps.unsqueeze(1), inf_v)
    cross.masked_fill_(~src_mask.view(1, cache.P, 1), inf_v)
    cross.masked_fill_(~tgt_mask.view(1, 1, cache.P), inf_v)
    best_per_target = cross.amin(dim=(0, 1))       # over K and source axis
    return torch.where(torch.isfinite(best_per_target), best_per_target, torch.zeros_like(best_per_target))


# ---------------------------------------------------------------------------
# Compact candidate pairs (replaces compact_candidate_pairs for regroup)
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# Aiming reachability mask (precheck augmentation for movement_pairwise_grid)
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------








