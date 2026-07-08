from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor

from .movement import (
    MovementConfig,
    PlanetMovement,
    _estimate_new_fleet_arrivals,
)


@dataclass(frozen=True)
class PlannedLaunches:
    source_slots: Tensor
    angle: Tensor
    ships: Tensor
    target_slots: Tensor
    eta_turns: Tensor
    valid: Tensor
    fleet_ids: Tensor



@dataclass(frozen=True)
class LaunchEntries:
    """Multi-launch table for one planning step.

    Each ``[L]`` entry encodes a single launch:

        ``source_slots[b, l]`` -> ``target_slots[b, l]`` with ``ships[b, l]``
        ships at heading ``angle[b, l]`` (rad), ETA ``eta[b, l]`` turns.

    Multiple entries may share the same ``source_slots`` value to encode
    multi-launch fan-out from a single planet. The per-source sum of
    ``ships`` over ``valid`` entries must respect that source's ship budget;
    the engine debits sources sequentially in entry order, so callers should
    plan against running residuals rather than the original budget.

    Entry order also defines the launch dispatch order — fleet IDs assigned
    via :func:`infer_planned_launches_from_entries` increase in cumulative
    order over valid entries, matching the engine's ``cumsum`` rule for
    sparse launch payloads.
    """

    source_slots: Tensor  # [L] long
    target_slots: Tensor  # [L] long
    ships: Tensor  # [L] float
    angle: Tensor  # [L] float
    eta: Tensor  # [L] float
    valid: Tensor  # [L] bool

    @property
    def width(self) -> int:
        return int(self.source_slots.shape[0])





def concat_launch_entries(entries: Sequence[LaunchEntries]) -> LaunchEntries:
    """Concatenate launch-entry tables along the L axis.

    All inputs must share the same ``B`` and per-tensor dtype/device.
    """
    if not entries:
        raise ValueError("concat_launch_entries requires at least one entry table")
    if len(entries) == 1:
        return entries[0]
    return LaunchEntries(
        source_slots=torch.cat([e.source_slots for e in entries], dim=0),
        target_slots=torch.cat([e.target_slots for e in entries], dim=0),
        ships=torch.cat([e.ships for e in entries], dim=0),
        angle=torch.cat([e.angle for e in entries], dim=0),
        eta=torch.cat([e.eta for e in entries], dim=0),
        valid=torch.cat([e.valid for e in entries], dim=0),
    )


def disambiguate_duplicate_launches(
    entries: LaunchEntries,
    *,
    epsilon: float = 1.0e-5,
) -> LaunchEntries:
    """Perturb angle on duplicate launches so they're tracker-distinguishable.

    The engine's slot-order fleet-id assignment plus the agent's
    reconciliation by ``(owner, source, ships, angle)`` cannot disambiguate
    two pending entries that share the full tuple, even though the engine
    creates two distinct fleets. ``PlanetMovement._reconcile_pending_own_launches``
    hard-fails on such collisions ("multiple pending entries resolved to the
    same engine fleet id …").

    This helper finds entries that share ``(source, angle, ships)`` with an
    earlier valid entry in the same lane and adds ``k * epsilon`` to the
    angle of the k-th duplicate. ``epsilon = 1e-5`` rad is well above
    float32's ULP at angle magnitude ~1 (≈6e-8) and well below any
    behaviorally-meaningful aim error (5e-4 unit displacement at 50-unit
    fleet range — sub-planet-radius).

    Both the engine action (``_entries_to_sparse_payload``) and the stash
    (``infer_planned_launches_from_entries``) read ``entries.angle``, so
    applying the perturbation here keeps both branches consistent — the
    engine creates fleets with the perturbed angle, the obs reports the
    perturbed angle, and the stash matches.
    """
    src = entries.source_slots                                                 # [L]
    ang = entries.angle                                                         # [L]
    ships = entries.ships                                                       # [L]
    valid = entries.valid                                                       # [L]
    L = src.shape[0]
    if L < 2 or not bool(valid.any()):
        return entries
    device = src.device
    src_i = src.unsqueeze(1)                                                    # [L, 1]
    src_j = src.unsqueeze(0)                                                    # [1, L]
    ang_i = ang.unsqueeze(1)
    ang_j = ang.unsqueeze(0)
    ships_i = ships.unsqueeze(1)
    ships_j = ships.unsqueeze(0)
    valid_i = valid.unsqueeze(1)
    valid_j = valid.unsqueeze(0)
    j_indices = torch.arange(L, device=device).view(1, L)
    i_indices = torch.arange(L, device=device).view(L, 1)
    earlier = j_indices < i_indices                                             # [L, L]
    match = (
        valid_i & valid_j
        & (src_i == src_j)
        & (ang_i == ang_j)
        & (ships_i == ships_j)
        & earlier
    )                                                                           # [L, L]
    if not bool(match.any()):
        return entries
    dup_count = match.sum(dim=1).to(ang.dtype)                                  # [L]
    new_angle = ang + dup_count * float(epsilon)
    return LaunchEntries(
        source_slots=entries.source_slots,
        target_slots=entries.target_slots,
        ships=entries.ships,
        angle=new_angle,
        eta=entries.eta,
        valid=entries.valid,
    )






def ensure_planet_movement(
    *,
    obs_tensors: dict,
    expected_cfg: MovementConfig,
    cached_movement: PlanetMovement | None,
) -> PlanetMovement:
    """Reuse the cached movement (rolled forward) if its config matches, else
    rebuild from the observation. Returns the live movement cache."""
    if cached_movement is not None and cached_movement.config == expected_cfg:
        cached_movement.update(obs_tensors)
        return cached_movement
    return PlanetMovement.from_obs_tensors(obs_tensors, config=expected_cfg)


def _resolve_player_next_fleet_id(
    obs_tensors: dict,
    *,
    device: torch.device,
) -> Tensor:
    next_fleet_id = obs_tensors.get("player_next_fleet_id", obs_tensors.get("next_fleet_id"))
    if next_fleet_id is None:
        return torch.zeros((), dtype=torch.long, device=device)
    return next_fleet_id.to(device=device, dtype=torch.long)


def infer_planned_launches_from_entries(
    *,
    obs_tensors: dict,
    movement: PlanetMovement,
    entries: LaunchEntries,
    player_id: int,
) -> PlannedLaunches:
    """Resolve fleet IDs and target/ETA arrivals for a launch table.

    Fleet IDs increase in entry order over valid launches via
    ``cumsum(valid) - valid``. This matches the engine's sparse rule and
    cleanly handles multi-launch from the same source slot (each entry receives
    a distinct fleet ID). Target/ETA are recomputed via the swept-pair physics
    in :func:`_estimate_new_fleet_arrivals`. Result is shaped ``[L]``.
    """
    source_slots = entries.source_slots
    angle = entries.angle
    ships = entries.ships
    launch_valid = entries.valid
    L = source_slots.shape[0]
    device = source_slots.device
    P = max(int(movement.P), 1)

    next_fleet_id = _resolve_player_next_fleet_id(obs_tensors, device=device)
    # ``cumsum(valid) - valid`` mirrors the engine's launch_rank formula and is
    # independent of source ordering, so it supports multi-launch per source.
    launch_long = launch_valid.to(torch.long)
    launch_rank = launch_long.cumsum(0) - launch_long
    fleet_ids = next_fleet_id + launch_rank

    src_safe = source_slots.clamp(min=0, max=P - 1)
    launch_x, launch_y = movement.position_at_slots(src_safe, 0)
    source_r = movement.radii[src_safe]
    start_x = launch_x + torch.cos(angle) * (source_r + 0.1)
    start_y = launch_y + torch.sin(angle) * (source_r + 0.1)
    source_planet_ids = movement.planet_ids[src_safe]

    rows = torch.full((L, 7), -1.0, dtype=movement.dtype, device=device)
    rows[..., 0] = fleet_ids.to(dtype=movement.dtype)
    rows[..., 1] = float(player_id)
    rows[..., 2] = start_x.to(dtype=movement.dtype)
    rows[..., 3] = start_y.to(dtype=movement.dtype)
    rows[..., 4] = angle.to(dtype=movement.dtype)
    rows[..., 5] = source_planet_ids.to(dtype=movement.dtype)
    rows[..., 6] = ships.to(dtype=movement.dtype)
    rows[..., 0] = torch.where(
        launch_valid, rows[..., 0], torch.full_like(rows[..., 0], -1.0)
    )

    target_slots = torch.zeros(L, dtype=torch.long, device=device)
    eta_turns = torch.zeros(L, dtype=torch.float32, device=device)
    intent_valid = torch.zeros(L, dtype=torch.bool, device=device)
    fleet_slot = torch.where(launch_valid)[0]
    if int(fleet_slot.numel()) > 0:
        estimate = _estimate_new_fleet_arrivals(
            movement=movement,
            obs_fleets=rows,
            fleet_slot=fleet_slot,
        )
        valid_hit = estimate["has_hit"]
        if bool(valid_hit.any()):
            src = fleet_slot[valid_hit]
            target_slots[src] = estimate["target_slot"][valid_hit]
            eta_turns[src] = estimate["eta_index"][valid_hit].to(dtype=torch.float32) + 1.0
            intent_valid[src] = True

    return PlannedLaunches(
        source_slots=source_slots,
        angle=angle,
        ships=ships,
        target_slots=target_slots,
        eta_turns=eta_turns,
        valid=intent_valid,
        fleet_ids=fleet_ids,
    )




def apply_private_planned_launches(
    *,
    movement: PlanetMovement,
    launches: PlannedLaunches,
    owner_id: int,
    obs_tensors: dict,
) -> None:
    """Record an agent's just-decided launches into its movement cache.

    Seeds the arrival buckets with the source-derived prediction but does *not*
    seed the ``tracked_fleet_ids`` ledger directly: ``launches.fleet_ids`` come
    from the global ``next_fleet_id`` plus a cumsum, which collides with other
    slots' IDs because the engine processes player actions in slot order.
    Instead the launches are stashed and paired against the next observation's
    fleets (which carry the engine's authoritative IDs) via
    ``_reconcile_pending_own_launches``.

    ``obs_tensors`` is required (we snapshot ``next_fleet_id`` for reconciliation).
    """
    if not movement.track_fleets:
        return
    movement.record_fleet_arrivals(
        target_slots=launches.target_slots,
        owner_ids=int(owner_id),
        ships=launches.ships,
        eta=launches.eta_turns,
        valid=launches.valid,
    )
    nfid = obs_tensors.get("next_fleet_id")
    if nfid is None:
        raise ValueError("obs_tensors is missing 'next_fleet_id'")
    movement.stash_pending_own_launches(
        owner_id=int(owner_id),
        source_slots=launches.source_slots,
        ships=launches.ships,
        angle=launches.angle,
        target_slots=launches.target_slots,
        eta=launches.eta_turns,
        valid=launches.valid,
        prev_next_fleet_id=nfid,
    )
