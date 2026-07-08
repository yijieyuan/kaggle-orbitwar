"""What-if-I-launch flow projection over a :class:`PlanetGarrisonStatus`.

``PlanetGarrisonStatus`` is a per-planet ledger of projected owner / ships over a
future horizon, computed from the fleets we currently know about, assuming we do
nothing. :func:`sparse_launch_flow_delta` answers the forward-looking question an
agent faces — *"if I launch these ships, how does each player's net ship flow
(production minus combat losses) change?"* — by recomputing the production→combat
recurrence only for the planets a launch touches and diffing against the baseline.

A launch is two-sided: it debits the source planet's garrison (ships leave now,
before that turn's production) and credits the target's arrival at step ``k``.

Two leading axes are supported (single game):

- ``C`` — candidates: the different launches / launch-sets being scored.
- ``L`` — launches within a candidate: a candidate *is* a set of launches; ``L``
  is summed away during aggregation and is not an output axis.

Pass launches as ``[L]`` (no candidate axis) or ``[C, L]``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .movement import PlanetGarrisonStatus


@dataclass(frozen=True)
class LaunchSet:
    """A batched set of hypothetical launches issued on the current turn.

    All tensors share a leading prefix (empty or ``[C]``) followed by a
    trailing launch axis ``L`` (use ``L=1`` for a single launch). ``eta`` is in
    steps from the current frame (arrival lands at garrison step ``k = eta``;
    ``eta`` must be ``>= 1``). ``owner`` defaults to the acting player but is
    per-launch so opponent what-ifs are expressible.
    """

    source_slots: Tensor  # [*prefix, L] long  (planet slot to launch FROM)
    target_slots: Tensor  # [*prefix, L] long  (planet slot to launch TO)
    ships: Tensor         # [*prefix, L] float
    eta: Tensor           # [*prefix, L] float/long (steps to arrival, >= 1)
    owner: Tensor         # [*prefix, L] long
    valid: Tensor         # [*prefix, L] bool


    @property
    def has_candidate_axis(self) -> bool:
        return self.source_slots.dim() >= 2





def _per_step_survivor(arrivals: Tensor) -> tuple[Tensor, Tensor]:
    """Engine survivor over the owner axis for every step.

    ``arrivals`` is ``[..., A]``; returns ``(survivor_owner, survivor_ships)``
    over the trailing axis, applying the engine rule: survivor ships = top1 -
    top2, ties annihilate (ships 0). Owner is meaningful only where ships > 0.
    """
    A = int(arrivals.shape[-1])
    if A >= 2:
        top2 = arrivals.topk(k=2, dim=-1)
        top_ships = top2.values[..., 0]
        second_ships = top2.values[..., 1]
        top_owner = top2.indices[..., 0].to(dtype=torch.long)
    else:
        top_ships, top_owner = arrivals.max(dim=-1)
        second_ships = torch.zeros_like(top_ships)
        top_owner = top_owner.to(dtype=torch.long)
    tied = top_ships == second_ships
    survivor_ships = torch.where(
        tied, torch.zeros_like(top_ships), (top_ships - second_ships).clamp(min=0.0)
    )
    return top_owner, survivor_ships


def _run_exact_recurrence(
    *,
    init_owner: Tensor,   # [N, P] long
    init_ships: Tensor,   # [N, P] float (already source-debited)
    prod: Tensor,         # [N, P] float
    alive: Tensor,        # [N, P, H+1] bool
    arrivals: Tensor,     # [N, P, H, A] float (steps 1..H, baseline + delta)
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Walk the engine production→combat recurrence over ``k = 1..H``.

    Mirrors ``PlanetMovement._fill_garrison_trajectory`` Half B exactly, but for
    all planets without the simple/complex fast-path split (clarity over the few
    saved kernels — this is the reference path). Returns
    ``(owner, ships, pre_owner, pre_ships)`` each ``[N, P, H+1]`` with step 0 set
    to the recurrence's starting state.
    """
    N, P = init_owner.shape
    H = int(arrivals.shape[2])
    device = init_ships.device

    owner_out = torch.empty(N, P, H + 1, dtype=init_owner.dtype, device=device)
    ships_out = torch.empty(N, P, H + 1, dtype=init_ships.dtype, device=device)
    pre_owner_out = torch.empty_like(owner_out)
    pre_ships_out = torch.empty_like(ships_out)
    owner_out[..., 0] = init_owner
    ships_out[..., 0] = init_ships
    pre_owner_out[..., 0] = init_owner
    pre_ships_out[..., 0] = init_ships

    survivor_owner, survivor_ships = _per_step_survivor(arrivals)  # [N, P, H]

    state_owner = init_owner.clone()
    state_ships = init_ships.clone()
    zero_ships = torch.zeros((), dtype=state_ships.dtype, device=device)
    neg_one = torch.full((), -1, dtype=state_owner.dtype, device=device)
    zero_prod = torch.zeros((), dtype=prod.dtype, device=device)

    for k in range(1, H + 1):
        a_before = alive[..., k - 1]
        a_now = alive[..., k]
        s_owner = survivor_owner[..., k - 1]
        s_ships = survivor_ships[..., k - 1]

        # Production: owned planets alive at the start of this step.
        produces = a_before & (state_owner >= 0)
        state_ships = state_ships + torch.where(produces, prod, zero_prod)

        # Pre-combat snapshot (after production, before same-step combat).
        pre_owner_out[..., k] = torch.where(a_now, state_owner, neg_one)
        pre_ships_out[..., k] = torch.where(a_now, state_ships, zero_ships)

        # Survivor vs the prior garrison.
        has_combat = (s_ships > 0.0) & a_now
        same = state_owner == s_owner
        diff = state_ships - s_ships
        attacker_wins = (~same) & (diff < 0.0)
        combat_ships = torch.where(same, state_ships + s_ships, diff.abs())
        combat_owner = torch.where(attacker_wins, s_owner, state_owner)
        state_ships = torch.where(has_combat, combat_ships, state_ships)
        state_owner = torch.where(has_combat, combat_owner, state_owner)

        # End-of-step death reset.
        state_owner = torch.where(a_now, state_owner, neg_one)
        state_ships = torch.where(a_now, state_ships, zero_ships)

        owner_out[..., k] = state_owner
        ships_out[..., k] = state_ships

    return owner_out, ships_out, pre_owner_out, pre_ships_out


def _validate_inputs(
    status: PlanetGarrisonStatus,
    prod: Tensor,
    alive_by_step: Tensor,
    player_count: int,
) -> tuple[int, int, int, int]:
    """Check shapes and return ``(B, P, H, A)``."""
    if status.arrivals_by_owner is None:
        raise ValueError(
            "garrison status must carry arrivals_by_owner (build it from a "
            "PlanetMovement with track_fleets=True)"
        )
    if status.pre_combat_owner is None or status.pre_combat_ships is None:
        raise ValueError("garrison status must carry pre_combat_owner/ships")
    if status.owner.dim() != 2:
        raise ValueError(
            "expected a full-board status with owner shaped [P, H+1]; got "
            f"{tuple(status.owner.shape)}"
        )
    P, H1 = status.owner.shape
    H = H1 - 1
    A = int(status.arrivals_by_owner.shape[-1])
    if int(player_count) != A:
        raise ValueError(
            f"player_count={player_count} disagrees with arrivals owner axis A={A}"
        )
    if tuple(prod.shape) != (P,):
        raise ValueError(f"prod must be [P]=({P},); got {tuple(prod.shape)}")
    if tuple(alive_by_step.shape) != (H1, P):
        raise ValueError(
            f"alive_by_step must be [H+1, P]=({H1}, {P}); got "
            f"{tuple(alive_by_step.shape)}"
        )
    return P, H, A












# ---------------------------------------------------------------------------
# Per-player flow accounting: diff two garrison statuses
# ---------------------------------------------------------------------------




@dataclass(frozen=True)
class GarrisonFlowDiff:
    """Difference in per-player flow between a current and a hypothetical status.

    Each field is ``[*prefix, A]`` (per player). ``*_delta`` is
    ``hypothetical - current``. ``net_ship_delta`` is the change in net ships
    gained (``produced - lost_to_combat``) — i.e. how much better/worse off each
    player ends up under the hypothetical, ignoring ships in transit.
    """

    player_id: int
    ships_produced_current: Tensor
    ships_produced_hypothetical: Tensor
    ships_produced_delta: Tensor
    ships_lost_combat_current: Tensor
    ships_lost_combat_hypothetical: Tensor
    ships_lost_combat_delta: Tensor
    net_ship_delta: Tensor

    @property
    def player_count(self) -> int:
        return int(self.ships_produced_delta.shape[-1])





def _flow_terms_per_planet(
    *,
    owner: Tensor,        # [.., P, H+1]
    pre_owner: Tensor,    # [.., P, H+1]
    pre_ships: Tensor,    # [.., P, H+1]
    arr_full: Tensor,     # [.., P, H+1, A]
    prod: Tensor,         # [.., P] (broadcastable)
    alive_pmajor: Tensor, # [.., P, H+1] (broadcastable, planet-major)
) -> tuple[Tensor, Tensor]:
    """Per-planet production and combat losses, summed over the horizon only.

    Returns ``(produced, combat_lost)`` each ``[.., P, A]``. Combat follows
    the engine combat rule; production credits ``prod`` to the owner
    holding the planet entering each step (from ``prod``, not ship deltas, so a
    source launch debit is not mistaken for negative production).
    """
    A = int(arr_full.shape[-1])
    H = int(owner.shape[-1]) - 1
    fdtype = pre_ships.dtype
    a_idx = torch.arange(A, device=owner.device)

    # Production credited to owner at the start of each step (= owner[k-1]).
    producing_owner = owner[..., :H]                                 # [.., P, H]
    amount = prod.unsqueeze(-1) * alive_pmajor[..., :H].to(fdtype)   # [.., P, H]
    prod_owner_oh = producing_owner.unsqueeze(-1) == a_idx           # [.., P, H, A]
    produced = (amount.unsqueeze(-1) * prod_owner_oh.to(fdtype)).sum(dim=-2)  # [.., P, A]

    # Combat per step (engine top1 - top2 survivor, then survivor vs garrison).
    arr_k = arr_full[..., 1:, :]
    survivor_owner, survivor_ships = _per_step_survivor(arr_k)       # [.., P, H]
    survived = torch.where(
        a_idx == survivor_owner.unsqueeze(-1),
        survivor_ships.unsqueeze(-1),
        torch.zeros_like(survivor_ships).unsqueeze(-1),
    )
    attacker_lost = (arr_k - survived).clamp(min=0.0)                # [.., P, H, A]
    prior_owner = pre_owner[..., 1:]
    prior_ships = pre_ships[..., 1:]
    fights_garrison = (survivor_ships > 0.0) & (survivor_owner != prior_owner) & (survivor_owner >= 0)
    garrison_loss = torch.where(
        fights_garrison, torch.minimum(prior_ships, survivor_ships), torch.zeros_like(prior_ships)
    )
    is_survivor = (a_idx == survivor_owner.unsqueeze(-1)) & fights_garrison.unsqueeze(-1)
    is_prior = (
        (a_idx == prior_owner.unsqueeze(-1))
        & fights_garrison.unsqueeze(-1)
        & (prior_owner >= 0).unsqueeze(-1)
    )
    garrison_lost = garrison_loss.unsqueeze(-1) * (is_survivor.to(fdtype) + is_prior.to(fdtype))
    combat_lost = (attacker_lost + garrison_lost).sum(dim=-2)        # [.., P, A]
    return produced, combat_lost










# ---------------------------------------------------------------------------
# Sparse prototype: per-candidate flow deltas without dense [C, P, H, A]
# ---------------------------------------------------------------------------


def _normalize_launches_bcl(launches: LaunchSet) -> tuple[Tensor, ...]:
    """Return ``(src, tgt, ships, eta, owner, valid)`` shaped ``[C, L]``."""
    fields = (
        launches.source_slots, launches.target_slots, launches.ships,
        launches.eta, launches.owner, launches.valid,
    )
    if launches.has_candidate_axis:
        return fields
    return tuple(f.unsqueeze(0) for f in fields)  # [L] -> [1, L]


def sparse_launch_flow_delta(
    status: PlanetGarrisonStatus,
    *,
    prod: Tensor,
    alive_by_step: Tensor,
    player_count: int,
    launches: LaunchSet,
    player_id: int = 0,
) -> GarrisonFlowDiff:
    """Sparse equivalent of ``diff_garrison_flow(status, apply_launches_exact(...))``.

    Returns the **same** exact per-candidate, per-player flow diff as the dense
    pipeline, but never materializes the dense ``[C, P, H, A]`` arrivals or a
    ``[C, P, H+1]`` trajectory. It exploits two facts:

    - the garrison projection is per-planet independent given the arrival
      buckets, so a launch only changes the trajectory of the planets it touches
      (its source, via the debit, and its target, via the credit);
    - untouched planets contribute zero to the flow *delta*.

    So it recomputes the recurrence only for the affected ``(candidate, planet)``
    cells (~2 per candidate for single launches, vs all ``P``) and scatter-adds
    their per-planet flow deltas into ``[C, A]``. Cost and memory scale with
    the number of affected cells, not ``B·C·P``.
    """
    P, H, A = _validate_inputs(status, prod, alive_by_step, player_count)
    device = status.owner.device
    fdtype = status.ships.dtype
    assert status.pre_combat_owner is not None and status.pre_combat_ships is not None
    assert status.arrivals_by_owner is not None

    src, tgt, ships, eta, owner, valid = _normalize_launches_bcl(launches)
    C = int(src.shape[0])
    L = int(src.shape[-1])
    src = src.to(device=device, dtype=torch.long)
    tgt = tgt.to(device=device, dtype=torch.long)
    ships = ships.to(device=device, dtype=fdtype)
    owner = owner.to(device=device, dtype=torch.long)
    valid = valid.to(device=device, dtype=torch.bool)
    h_idx = torch.ceil(eta.to(device=device, dtype=fdtype)).to(torch.long) - 1

    valid_t = valid & (ships > 0) & (tgt >= 0) & (tgt < P) & (owner >= 0) & (owner < A) & (h_idx >= 0) & (h_idx < H)
    valid_s = valid & (ships > 0) & (src >= 0) & (src < P)
    src_safe = src.clamp(0, max(P - 1, 0))
    tgt_safe = tgt.clamp(0, max(P - 1, 0))

    # Affected planets per candidate (source debit OR target credit). [C, P]
    affected = torch.zeros(C, P, dtype=fdtype, device=device)
    affected.scatter_add_(1, src_safe, valid_s.to(fdtype))
    affected.scatter_add_(1, tgt_safe, valid_t.to(fdtype))
    affected_mask = affected > 0

    # Baseline per-planet flow (shared across candidates).
    base_prod_pp, base_combat_pp = _flow_terms_per_planet(
        owner=status.owner,
        pre_owner=status.pre_combat_owner,
        pre_ships=status.pre_combat_ships,
        arr_full=status.arrivals_by_owner,
        prod=prod,
        alive_pmajor=alive_by_step.permute(1, 0),
    )                                                        # [P, A]
    base_prod = base_prod_pp.sum(dim=0)                      # [A]
    base_combat = base_combat_pp.sum(dim=0)

    produced_delta = torch.zeros(C, A, dtype=fdtype, device=device)
    combat_delta = torch.zeros(C, A, dtype=fdtype, device=device)

    if bool(affected_mask.any()):
        c_aff, p_aff = affected_mask.nonzero(as_tuple=True)         # [N]
        N = int(c_aff.numel())
        cell_id = torch.full((C, P), -1, dtype=torch.long, device=device)
        cell_id[c_aff, p_aff] = torch.arange(N, device=device)

        # Source debit per affected cell.
        debit_cp = torch.zeros(C, P, dtype=fdtype, device=device)
        debit_cp.scatter_add_(1, src_safe, torch.where(valid_s, ships, torch.zeros_like(ships)))
        debit_aff = debit_cp[c_aff, p_aff]                          # [N]

        # Target credits scattered onto the affected cells: [N, H, A].
        arr_aff = torch.zeros(N, H, A, dtype=fdtype, device=device)
        launch_cell = cell_id.gather(1, tgt_safe)                   # [C, L]
        m = valid_t
        cells, hh, oo, ss = launch_cell[m], h_idx[m], owner[m], ships[m]
        ok = cells >= 0
        arr_aff.index_put_((cells[ok], hh[ok], oo[ok]), ss[ok], accumulate=True)

        base_arr_k = status.arrivals_by_owner[..., 1:, :]           # [P, H, A]
        arrivals_cell = base_arr_k[p_aff] + arr_aff                 # [N, H, A]

        init_owner = status.owner[p_aff, 0]                         # [N]
        init_ships = (status.ships[p_aff, 0] - debit_aff).clamp(min=0.0)
        prod_aff = prod[p_aff]                                      # [N]
        alive_aff = alive_by_step[:, p_aff].transpose(0, 1)         # [N, H+1]

        # One-planet recurrence per affected cell (P=1 lane).
        o_t, _s_t, po_t, ps_t = _run_exact_recurrence(
            init_owner=init_owner.unsqueeze(1),
            init_ships=init_ships.unsqueeze(1),
            prod=prod_aff.unsqueeze(1),
            alive=alive_aff.unsqueeze(1),
            arrivals=arrivals_cell.unsqueeze(1),
        )
        zero_frame = torch.zeros(N, 1, 1, A, dtype=fdtype, device=device)
        arr_full_cell = torch.cat([zero_frame, arrivals_cell.unsqueeze(1)], dim=-2)
        hyp_prod_pp, hyp_combat_pp = _flow_terms_per_planet(
            owner=o_t, pre_owner=po_t, pre_ships=ps_t, arr_full=arr_full_cell,
            prod=prod_aff.unsqueeze(1), alive_pmajor=alive_aff.unsqueeze(1),
        )
        dprod = hyp_prod_pp.squeeze(1) - base_prod_pp[p_aff]            # [N, A]
        dcombat = hyp_combat_pp.squeeze(1) - base_combat_pp[p_aff]
        produced_delta.index_put_((c_aff,), dprod, accumulate=True)
        combat_delta.index_put_((c_aff,), dcombat, accumulate=True)

    produced_current = base_prod.unsqueeze(0)                      # [1, A]
    combat_current = base_combat.unsqueeze(0)
    diff = GarrisonFlowDiff(
        player_id=int(player_id),
        ships_produced_current=produced_current,
        ships_produced_hypothetical=produced_current + produced_delta,
        ships_produced_delta=produced_delta,
        ships_lost_combat_current=combat_current,
        ships_lost_combat_hypothetical=combat_current + combat_delta,
        ships_lost_combat_delta=combat_delta,
        net_ship_delta=produced_delta - combat_delta,
    )
    # Squeeze the candidate axis back out for [L] launches (C == 1, no axis).
    if not launches.has_candidate_axis:
        def _sq(t: Tensor) -> Tensor:
            return t.squeeze(0)
        diff = GarrisonFlowDiff(
            player_id=diff.player_id,
            ships_produced_current=base_prod,
            ships_produced_hypothetical=_sq(diff.ships_produced_hypothetical),
            ships_produced_delta=_sq(diff.ships_produced_delta),
            ships_lost_combat_current=base_combat,
            ships_lost_combat_hypothetical=_sq(diff.ships_lost_combat_hypothetical),
            ships_lost_combat_delta=_sq(diff.ships_lost_combat_delta),
            net_ship_delta=_sq(diff.net_ship_delta),
        )
    return diff
