"""Flow-diff scored planner core: candidate scoring, shortlists, aim, selection.

Pure, tensor-only planning helpers for one game: the competitive net-ship-delta
scorer, target/source shortlists, capture-floor sizing, the strict-superset
reachability gate, the device-stable greedy selector, the hold-reserve cap
``safe_drain``, and the pressure-gradient regrouper.
"""

from __future__ import annotations

import torch
from torch import Tensor

from .garrison_launch import GarrisonFlowDiff, LaunchSet, sparse_launch_flow_delta
from .movement import PlanetGarrisonStatus, PlanetMovement
from .geometry import fleet_speed
from .intercept_aim import intercept_angle
from .movement_aiming import LAUNCH_SURFACE_OFFSET, TARGET_HIT_SURFACE_OFFSET
from .movement_step import LaunchEntries
from .distance_cache import min_distance_to_targets




def largest_initial_player_count(obs_tensors: dict) -> int:
    """Player count for the match: metadata if present, else distinct initial owners.

    """
    metadata_count = obs_tensors.get("player_count")
    if metadata_count is not None:
        count = (
            int(metadata_count.flatten()[0].item())
            if isinstance(metadata_count, Tensor)
            else int(metadata_count)
        )
        if count in (2, 4):
            return count
    initial = obs_tensors["initial_planets"]      # [P, 7]
    pid = initial[:, 0]
    owner = initial[:, 1]
    mask = (pid >= 0) & (owner >= 0)
    owners = owner[mask]
    n_max = 2
    if owners.numel() > 0:
        n_max = max(n_max, int(torch.unique(owners.long()).numel()))
    return n_max


# ---------------------------------------------------------------------------
# Scoring (P2): candidate launches -> competitive net-ship-delta
# ---------------------------------------------------------------------------


def make_launch_set(
    *,
    source_slots: Tensor,   # [C, L] long
    target_slots: Tensor,   # [C, L] long
    ships: Tensor,          # [C, L] float
    eta: Tensor,            # [C, L] float (steps to arrival, >= 1)
    valid: Tensor,          # [C, L] bool
    player_id: int,
) -> LaunchSet:
    """Build a candidate-axis ``LaunchSet`` owned by ``player_id``."""
    owner = torch.full_like(source_slots, int(player_id), dtype=torch.long)
    return LaunchSet(
        source_slots=source_slots.to(torch.long),
        target_slots=target_slots.to(torch.long),
        ships=ships,
        eta=eta,
        owner=owner,
        valid=valid.to(torch.bool),
    )


def competitive_score(diff: GarrisonFlowDiff, *, player_id: int) -> Tensor:
    """Competitive score: ``Δnet_me − Σ_opp Δnet_opp``.

    ``diff.net_ship_delta`` is ``[*prefix, A]`` (per-player change in net ships
    gained = produced − lost-to-combat); returns ``[*prefix]``. The opponent term
    is the equal-weight sum over rivals, so a launch is worth my net gain minus
    the opponents' net gain.
    """
    net = diff.net_ship_delta                       # [*prefix, A]
    me = net[..., int(player_id)]
    opp = net.sum(dim=-1) - me
    return me - opp


def score_candidates(
    status: PlanetGarrisonStatus,
    *,
    prod: Tensor,
    alive_by_step: Tensor,
    player_count: int,
    launches: LaunchSet,
    player_id: int,
) -> Tensor:
    """Competitive score per candidate. ``[C]`` (or scalar if no candidate axis).

    Uses the sparse exact flow projector.
    """
    diff = sparse_launch_flow_delta(
        status,
        prod=prod,
        alive_by_step=alive_by_step,
        player_count=int(player_count),
        launches=launches,
        player_id=int(player_id),
    )
    return competitive_score(diff, player_id=int(player_id))


# ---------------------------------------------------------------------------
# Candidate generation + greedy selection (P3: single-source, single-k, attack)
# ---------------------------------------------------------------------------



# Selection on CPU and CUDA must agree exactly: `torch.topk` / `torch.argmax`
# break ties differently across devices, and this planner ranks by integer ship
# counts / proximity that tie constantly — so device-stable selection is what
# keeps batch-CUDA play identical to CPU. We break ties by ascending slot index
# on both devices via a stable sort / lowest-index argmax.


def _stable_topk_indices(ranked: Tensor, k: int) -> Tensor:
    """Indices of the top-``k`` along the last dim, ties broken by ascending index
    identically on CPU and CUDA (stable descending sort)."""
    order = torch.argsort(ranked, dim=-1, descending=True, stable=True)
    return order[..., :max(1, int(k))]


def _stable_argmax(scores: Tensor) -> Tensor:
    """Lowest-index argmax along the last dim, device-deterministic on ties."""
    C = int(scores.shape[-1])
    is_max = scores == scores.max(dim=-1, keepdim=True).values
    idx = torch.arange(C, device=scores.device).expand_as(scores)
    return torch.where(is_max, idx, torch.full_like(idx, C)).argmin(dim=-1)


def _candidate_indices(values: Tensor, mask: Tensor, cap: int) -> tuple[Tensor, Tensor]:
    """Top-``cap`` slot indices of ``values`` under ``mask``. ``([K] long, [K] bool)``.

    Device-stable (ascending-index tie-break) — see note above.
    """
    p_count = values.shape[0]
    k = p_count if cap <= 0 else min(int(cap), p_count)
    neg_inf = torch.full_like(values, float("-inf"))
    ranked = torch.where(mask, values, neg_inf)
    top_idx = _stable_topk_indices(ranked, max(1, k))
    top_vals = ranked[top_idx]
    return top_idx, top_vals > float("-inf")


def is_comet_planet(obs_tensors: dict, P: int, device: torch.device) -> Tensor | None:
    """Per-slot mask of active comet planets, or ``None`` if absent."""
    comet_ids = obs_tensors.get("comet_planet_ids")
    planets = obs_tensors.get("planets")
    if comet_ids is None or planets is None:
        return None
    planet_ids = planets[..., 0].long()                       # [P]
    comet_ids = comet_ids.to(device=device)
    mask = torch.zeros(P, dtype=torch.bool, device=device)
    for c in range(int(comet_ids.shape[-1])):
        cid = comet_ids[c]
        mask = mask | ((planet_ids == cid) & (cid >= 0))
    return mask


def reinforcement_timing_factor(
    eta: Tensor,
    *,
    eta_free: float,
    eta_scale: float,
) -> Tensor:
    """Reaction-likelihood ramp ``ρ(eta) ∈ [0, 1]`` for reinforcement risk.

    ``ρ = clamp((eta − eta_free) / eta_scale, 0, 1)``. Below ``eta_free`` turns of
    flight the enemy has no time to react (ρ=0); over the next ``eta_scale`` turns
    reaction likelihood ramps linearly to 1. Pure arithmetic → CPU/CUDA agree.
    """
    scale = max(float(eta_scale), 1e-6)
    return ((eta - float(eta_free)) / scale).clamp(0.0, 1.0)


def capture_floor(
    garrison_status: PlanetGarrisonStatus,
    *,
    target_idx: Tensor,        # [T] long
    k_max: int,
    capture_overhead: float,
    player_id: int,
    reinforcement: Tensor | None = None,   # [T, K'>=K] float; added before ceil
) -> Tensor:
    """Owner-aware send floor per target at arrival turn ``k``. ``[T, K]``.

    - If I **own** the target at ``k`` (reinforcement), the floor is 1 — arriving
      ships add to my garrison, there is nothing to clear.
    - Otherwise (capture / retake), the floor is ``ceil(projected_defenders_at_k +
      overhead)``.
    - ``reinforcement`` (optional, ``[T, K' ≥ K]``) is added to the defender count
      before the ceil on capture cells (not on ``mine_at_k`` reinforcement cells) —
      the ETA-aware reactive-reinforcement margin. ``None`` ⇒ today's behaviour.

    Assumes ``k_max <= H``.
    """
    ships = garrison_status.ships
    owner = garrison_status.owner
    dtype = ships.dtype if ships.is_floating_point() else torch.float32
    T = target_idx.shape[0]
    H_axis = int(ships.shape[-1])
    P = int(ships.shape[0])
    K = max(0, min(int(k_max), H_axis - 1))
    if K == 0:
        return torch.empty(T, 0, dtype=dtype, device=ships.device)
    tgt = target_idx.clamp(min=0, max=max(P - 1, 0))
    gathered = ships[tgt].to(dtype=dtype)                       # [T, H+1]
    owner_g = owner[tgt]                                        # [T, H+1]
    k_idx = torch.arange(1, K + 1, device=ships.device).view(1, K).expand(T, K)
    defenders = gathered.gather(-1, k_idx)                      # [T, K]
    mine_at_k = owner_g.gather(-1, k_idx) == int(player_id)
    if reinforcement is not None:
        # Caller passes a margin with K' >= K (built from k_max=K_eta, while this
        # function's K = min(k_max, H-1) <= K_eta); slice down to our own K.
        assert reinforcement.shape[-1] >= K, (
            f"reinforcement last dim {reinforcement.shape[-1]} < capture_floor K={K}"
        )
        extra = reinforcement[..., :K].to(dtype=dtype, device=ships.device)
    else:
        extra = 0.0
    cap = (defenders + float(capture_overhead) + extra).clamp(min=1.0).ceil()
    return torch.where(mine_at_k, torch.ones_like(cap), cap)


def attack_target_mask(obs, obs_tensors: dict) -> Tensor:
    """Enemy ∪ neutral, alive, non-comet. ``[P]`` bool."""
    mask = (obs.is_enemy | obs.is_neutral) & obs.alive
    comet = is_comet_planet(obs_tensors, obs.P, obs.device)
    if comet is not None:
        mask = mask & ~comet
    return mask


def friendly_flip_targets(
    obs, garrison_status: PlanetGarrisonStatus, *, H: int, prod: Tensor,
) -> tuple[Tensor, Tensor]:
    """Own planets the do-nothing projection shows flipping within H.

    Returns ``(mask [P] bool, urgency [P] float)``. ``urgency`` ≈ projected
    ships lost if unaddressed = ``prod·(H − flip_turn) + garrison_now`` — same ship
    units as the ROI, used to fill the reserved defensive sub-quota.
    """
    P = obs.P
    device = obs.device
    pid = int(obs.player_id)
    if H <= 0:
        z = torch.zeros(P, device=device)
        return torch.zeros(P, dtype=torch.bool, device=device), z
    owner_h = garrison_status.owner[..., 1:]                     # [P, H]
    flips = obs.owned.unsqueeze(-1) & (owner_h != pid)           # currently mine, not mine at some k
    any_flip = flips.any(dim=-1)                                 # [P]
    # earliest flip turn (lowest-index True); _stable_argmax instead of raw argmax
    # so the tie among post-flip turns resolves identically on CPU and CUDA.
    flip_turn = _stable_argmax(flips.to(torch.int64)) + 1        # 1-based; valid where any_flip
    remaining = (float(H) - flip_turn.to(prod.dtype)).clamp(min=0.0)
    urgency = prod * remaining + obs.ships
    urgency = torch.where(any_flip, urgency, torch.full_like(urgency, float("-inf")))
    return any_flip, urgency


def build_target_shortlist(
    obs, obs_tensors, garrison_status, cache, *, config, K_eta, H, prod, source_mask,
):
    """Single unified shortlist: ``max_offensive_targets`` enemy/neutral targets by
    proximity ∪ ``max_defensive_targets`` friendly-flip targets by urgency., The
    two caps are independent (shortlist width == offensive + defensive), so each can
    be swept on its own. Returns ``(target_idx, target_exists)``."""
    P = obs.P
    device = obs.device
    n_attack = max(1, min(int(config.max_offensive_targets), P))
    R = max(0, min(int(config.max_defensive_targets), P))

    attack_mask = attack_target_mask(obs, obs_tensors)
    proximity = min_distance_to_targets(cache, source_mask, attack_mask, max_k=K_eta)
    attack_pref = torch.where(attack_mask, -proximity, torch.full_like(proximity, float("-inf")))
    atk_idx, atk_exists = _candidate_indices(attack_pref, attack_mask, n_attack)

    if R > 0:
        flip_mask, urgency = friendly_flip_targets(obs, garrison_status, H=H, prod=prod)
        def_idx, def_exists = _candidate_indices(urgency, flip_mask, R)
        target_idx = torch.cat([atk_idx, def_idx], dim=0)
        target_exists = torch.cat([atk_exists, def_exists], dim=0)
    else:
        target_idx, target_exists = atk_idx, atk_exists
    return target_idx, target_exists


def reachable_mask(
    movement: PlanetMovement,
    *,
    source_idx: Tensor,      # [S] long
    target_idx: Tensor,      # [T] long
    fleet_sizes: Tensor,     # [S, T, G] float
    eta_cap: Tensor,         # [T] float (per-target reach cap)
    eps: float = 1e-4,
) -> Tensor:
    """Strict-superset reachability gate for the body screen, ``[S, T, G]`` bool.

    A cell is reachable iff some step interval ``k in [1, eta_cap[b,t]]`` admits the
    straight-line shot: ``(d_k - gap) <= fleet_speed(size) * k * (1 + eps)`` where
    ``d_k`` is the distance from the source centre @ turn 0 to the target's **swept
    segment** ``[tgt@(k-1), tgt@k]`` and ``gap = src_r + tgt_r + offsets``.

    Using the swept segment (not the point ``tgt@k``) and the surface gap makes this
    a provable *necessary condition* for ``intercept_angle`` viability: a viable shot
    contacts the target at some continuous ``t_c <= eta_cap`` with
    ``dist(src@0, tgt@t_c) - gap <= speed * t_c <= speed * ceil(t_c)``, and the
    segment distance over the interval containing ``t_c`` is ``<= dist(src@0, tgt@t_c)``.
    Hence ``viable => reachable`` (the ``eps`` absorbs fp32 boundary noise) — the gate
    never false-prunes a launch the agent would otherwise aim. ``intercept_angle``
    re-validates every survivor, so the surplus kept beyond true viability is harmless.
    """
    S, T, G = fleet_sizes.shape
    P = int(movement.P)
    dt = movement.dtype
    K = max(1, min(int(movement.movement_horizon), int(torch.ceil(eta_cap.max()).item())))
    src = source_idx.clamp(0, P - 1)
    tgt = target_idx.clamp(0, P - 1)

    # Source centre @ turn 0; target positions @ turns 0..K (segment endpoints).
    sx = movement.x[0][src].view(S, 1, 1)                                   # [S,1,1]
    sy = movement.y[0][src].view(S, 1, 1)
    tx = movement.x[: K + 1].gather(1, tgt.view(1, T).expand(K + 1, T))     # [K+1,T]
    ty = movement.y[: K + 1].gather(1, tgt.view(1, T).expand(K + 1, T))
    ax = tx[:K, :].view(1, K, T); ay = ty[:K, :].view(1, K, T)             # tgt@(k-1)
    bx = tx[1:, :].view(1, K, T); by = ty[1:, :].view(1, K, T)             # tgt@k

    # Point-to-segment distance from (sx,sy) to segment [(ax,ay),(bx,by)] → [S,K,T].
    abx = bx - ax; aby = by - ay
    apx = sx - ax; apy = sy - ay
    denom = (abx * abx + aby * aby).clamp(min=1e-12)
    u = ((apx * abx + apy * aby) / denom).clamp(0.0, 1.0)
    cx = ax + u * abx; cy = ay + u * aby
    seg_dist = torch.sqrt(((sx - cx) ** 2 + (sy - cy) ** 2).clamp(min=0.0))  # [S,K,T]

    src_r = movement.radii[src].view(S, 1, 1)
    tgt_r = movement.radii[tgt].view(1, 1, T)
    gap = src_r + tgt_r + (LAUNCH_SURFACE_OFFSET + TARGET_HIT_SURFACE_OFFSET)
    surf = (seg_dist - gap).clamp(min=0.0)                                   # [S,K,T]

    kv = torch.arange(1, K + 1, device=movement.device, dtype=dt).view(1, K, 1)
    ratio = surf / kv
    within = kv <= eta_cap.view(1, 1, T)                                    # [1,K,T]
    ratio = torch.where(within, ratio, torch.full_like(ratio, float("inf")))
    min_ratio = ratio.amin(dim=1)                                          # [S,T]

    speed = fleet_speed(fleet_sizes.clamp(min=1.0))                          # [S,T,G]
    reachable = min_ratio.unsqueeze(-1) <= speed * (1.0 + float(eps))        # [S,T,G]
    distinct = (src.view(S, 1) != tgt.view(1, T)).unsqueeze(-1)             # [S,T,1]
    return reachable & distinct


def _greedy_select(
    *, P, W, device, dtype, score, cand_src, cand_send, cand_angle, cand_eta,
    cand_active, cand_tgt_slot, cand_tgt_short, cand_is_def, source_budget,
    target_exists, roi_threshold,
) -> LaunchEntries:
    """Masking-only greedy over [C, L] candidates: pick the best wave each iter,
    one per target, source-budget aware across all L contributors. Enforces the
    role mutex: a reinforced planet can't also be a source, and vice-versa."""
    C, L = int(cand_src.shape[0]), int(cand_src.shape[1])
    target_taken = ~target_exists.clone()                                        # [T]
    defended = torch.zeros(P, dtype=torch.bool, device=device)                   # reinforced this turn
    used_src = torch.zeros(P, dtype=torch.bool, device=device)                   # contributed this turn

    w_src = torch.zeros(W, L, dtype=torch.long, device=device)
    w_send = torch.zeros(W, L, dtype=dtype, device=device)
    w_angle = torch.zeros(W, L, dtype=dtype, device=device)
    w_eta = torch.ones(W, L, dtype=dtype, device=device)
    w_tgt = torch.zeros(W, L, dtype=torch.long, device=device)
    w_active = torch.zeros(W, L, dtype=torch.bool, device=device)

    for w in range(W):
        taken_cand = target_taken[cand_tgt_short]                               # [C]
        budget_at = source_budget[cand_src]                                     # [C, L]
        can_fund = ((cand_send <= budget_at) | ~cand_active).all(dim=-1)        # [C]
        # role mutex: target not already drained as a source; no contributor is a
        # planet we're reinforcing this turn.
        tgt_used_as_src = used_src[cand_tgt_slot]                               # [C]
        contrib_defended = (defended[cand_src] & cand_active).any(dim=-1)       # [C]
        mask = torch.isfinite(score) & ~taken_cand & can_fund & ~tgt_used_as_src & ~contrib_defended
        masked = torch.where(mask, score, torch.full_like(score, float("-inf")))
        best_c = _stable_argmax(masked)                                         # scalar, device-stable
        best_score = masked[best_c]
        fired = bool(torch.isfinite(best_score) & (best_score > roi_threshold))
        if not fired:
            break

        sel_src = cand_src[best_c]                   # [L]
        sel_send = cand_send[best_c]
        sel_active = cand_active[best_c]
        w_src[w] = sel_src
        w_send[w] = torch.where(sel_active, sel_send, torch.zeros_like(sel_send))
        w_angle[w] = cand_angle[best_c]
        w_eta[w] = cand_eta[best_c]
        w_tgt[w] = cand_tgt_slot[best_c]
        w_active[w] = sel_active

        # debit all contributors' sends from their source budgets.
        debit = torch.zeros_like(source_budget)
        debit.scatter_add_(0, sel_src, torch.where(sel_active, sel_send, torch.zeros_like(sel_send)))
        source_budget = (source_budget - debit).clamp(min=0.0)
        # mark target taken (one wave per target).
        target_taken[cand_tgt_short[best_c]] = True
        # role mutex bookkeeping: mark contributors used; mark reinforced targets
        # defended. Sum active marks per planet (order-independent) and OR them in.
        src_mark = torch.zeros(P, dtype=torch.long, device=device)
        src_mark.scatter_add_(0, sel_src, sel_active.to(torch.long))
        used_src = used_src | (src_mark > 0)
        sel_tgt = cand_tgt_slot[best_c]
        sel_is_def = bool(cand_is_def[best_c])
        defended[sel_tgt] = defended[sel_tgt] | sel_is_def

    # Flatten waves x contributors into a LaunchEntries table.
    WL = W * L
    entries = LaunchEntries(
        source_slots=w_src.reshape(WL),
        target_slots=w_tgt.reshape(WL),
        ships=torch.where(w_active, w_send, torch.zeros_like(w_send)).reshape(WL),
        angle=torch.where(w_active, w_angle, torch.zeros_like(w_angle)).reshape(WL),
        eta=torch.where(w_active, w_eta, torch.ones_like(w_eta)).reshape(WL),
        valid=w_active.reshape(WL),
    )
    return entries, source_budget   # source_budget = leftover ships per planet


def _plan_regroup(
    *, movement, obs, obs_tensors, garrison_status, leftover, original_ships,
    pressure, config, H,
) -> LaunchEntries:
    """Pressure-gradient marshalling of leftover ships.

    Moves ships from low-pressure planets toward nearby higher-pressure owned
    planets, capped by ``safe_drain`` (minus what attacks already drew), only when
    the destination is materially more stressed, reachable within
    ``max_regroup_time``, and **still owned at the fleet's arrival turn**.
    """
    P = obs.P
    device = obs.device
    dtype = original_ships.dtype
    pid = int(obs.player_id)
    min_send = float(config.min_ships_to_launch)

    src_mask = obs.owned & obs.alive & (leftover >= min_send)
    if not bool(src_mask.any()):
        return _empty_entries(device, dtype)
    S_cap = max(1, min(int(config.max_regroup_sources_per_lane), P))
    src_idx, src_exists = _candidate_indices(leftover, src_mask, S_cap)          # rank by leftover
    S = int(src_idx.shape[0])
    leftover_s = leftover[src_idx.clamp(0, P - 1)]
    orig_s = original_ships[src_idx.clamp(0, P - 1)]
    H_eff = torch.full((), float(H), dtype=dtype, device=device)
    drain_s = safe_drain(
        garrison_status, source_idx=src_idx, source_ships=orig_s,
        H_eff=H_eff, player_id=pid,
    )
    committed_s = (orig_s - leftover_s).clamp(min=0.0)
    regroup_cap = torch.minimum(leftover_s, (drain_s - committed_s).clamp(min=0.0)).floor()
    can_send = src_exists & (regroup_cap >= min_send)
    if not bool(can_send.any()):
        return _empty_entries(device, dtype)

    # Destinations are owned, alive, non-comet planets (do-nothing projection).
    dst_mask = obs.owned & obs.alive
    comet = is_comet_planet(obs_tensors, P, device)
    if comet is not None:
        dst_mask = dst_mask & ~comet
    T_cap = max(1, min(int(config.max_regroup_targets_per_source), P))
    dst_idx, dst_exists = _candidate_indices(pressure, dst_mask, T_cap)          # rank by pressure
    T = int(dst_idx.shape[0])

    # Fixed-size regroup aim via the continuous-intercept aimer (sub-turn lead + a
    # swept first-contact body screen on an AABB-culled shortlist).
    # Strict-superset reachability precheck defers the body screen to destinations a
    # source can reach within max_regroup_time (bit-identical to the ungated path).
    regroup_active = reachable_mask(
        movement, source_idx=src_idx, target_idx=dst_idx,
        fleet_sizes=regroup_cap.view(S, 1, 1).expand(S, T, 1),
        eta_cap=torch.full((T,), float(config.max_regroup_time), device=device),
    ).squeeze(-1)                                                                # [S, T]
    aim = intercept_angle(
        movement,
        src_idx.unsqueeze(1),                                                    # [S, 1]
        dst_idx.unsqueeze(0),                                                     # [1, T]
        regroup_cap.unsqueeze(1),                                                 # [S, 1]
        active=regroup_active,
    )
    angle = aim["angle"]                                                         # [S, T]
    eta = aim["eta"]
    viable = aim["viable"]

    src_pres = pressure[src_idx.clamp(0, P - 1)].view(S, 1)
    dst_pres = pressure[dst_idx.clamp(0, P - 1)].view(1, T)
    gap = dst_pres - src_pres                                                    # [S, T]

    # arrival-turn ownership check: dst must still be mine at k = ceil(eta).
    owner = garrison_status.owner                                               # [P, H+1]
    H_axis = int(owner.shape[-1])
    dst_owner = owner[dst_idx.clamp(0, P - 1)]                                  # [T, H+1]
    k = torch.ceil(eta).clamp(min=0, max=H_axis - 1).to(torch.long)             # [S, T]
    owner_at_k = dst_owner.unsqueeze(0).expand(S, T, H_axis).gather(-1, k.unsqueeze(-1)).squeeze(-1)
    still_mine = owner_at_k == pid

    src_neq_dst = src_idx.view(S, 1) != dst_idx.view(1, T)
    valid = (
        viable & still_mine & src_neq_dst
        & (gap > float(config.regroup_pressure_delta_min))
        & (eta <= float(config.max_regroup_time))
        & can_send.view(S, 1) & dst_exists.view(1, T)
    )
    sc = torch.where(
        valid,
        gap - float(config.regroup_time_penalty_weight) * eta,
        torch.full_like(gap, float("-inf")),
    )
    best_t = _stable_argmax(sc)                                                  # [S] device-stable
    best_score = sc.gather(-1, best_t.unsqueeze(-1)).squeeze(-1)                 # [S]
    best_valid = torch.isfinite(best_score)
    s_ar = torch.arange(S, device=device)
    best_dst = dst_idx[best_t]                                                   # [S]
    best_angle = angle[s_ar, best_t]
    best_eta = eta[s_ar, best_t]

    return LaunchEntries(
        source_slots=src_idx,
        target_slots=best_dst,
        ships=torch.where(best_valid, regroup_cap, torch.zeros_like(regroup_cap)),
        angle=torch.where(best_valid, best_angle, torch.zeros_like(best_angle)),
        eta=torch.where(best_valid, best_eta, torch.ones_like(best_eta)),
        valid=best_valid,
    )


def _empty_entries(device: torch.device, dtype: torch.dtype) -> LaunchEntries:
    z = torch.zeros(0, dtype=dtype, device=device)
    zl = torch.zeros(0, dtype=torch.long, device=device)
    return LaunchEntries(
        source_slots=zl, target_slots=zl, ships=z, angle=z, eta=z,
        valid=torch.zeros(0, dtype=torch.bool, device=device),
    )


def entries_to_sparse_payload(entries: LaunchEntries, *, planet_ids: Tensor) -> dict[str, Tensor]:
    """Convert a LaunchEntries table to the sparse action-row payload."""
    L = entries.source_slots.shape[0]
    device = entries.source_slots.device
    P = int(planet_ids.shape[0])
    valid_long = entries.valid.to(torch.int64)
    counts = valid_long.sum().to(torch.int32)
    max_count = int(counts.item())
    out_from = torch.full((max_count,), -1, dtype=torch.int32, device=device)
    out_angle = torch.zeros((max_count,), dtype=torch.float32, device=device)
    out_ships = torch.zeros((max_count,), dtype=torch.float32, device=device)
    if max_count == 0:
        return {"from_planet_id": out_from, "angle": out_angle, "num_ships": out_ships, "counts": counts}
    safe_src = entries.source_slots.clamp(min=0, max=max(P - 1, 0))
    from_pid_full = planet_ids[safe_src].to(torch.int32)
    launch_rank = valid_long.cumsum(0) - valid_long
    l_idx = torch.where(entries.valid)[0]
    pos = launch_rank[l_idx]
    out_from[pos] = from_pid_full[l_idx]
    out_angle[pos] = entries.angle[l_idx].to(torch.float32)
    out_ships[pos] = entries.ships[l_idx].to(torch.float32)
    return {"from_planet_id": out_from, "angle": out_angle, "num_ships": out_ships, "counts": counts}


def empty_action_row(device: torch.device) -> dict[str, Tensor]:
    """Sparse launch payload with zero launches."""
    return {
        "from_planet_id": torch.full((0,), -1, dtype=torch.int32, device=device),
        "angle": torch.zeros((0,), dtype=torch.float32, device=device),
        "num_ships": torch.zeros((0,), dtype=torch.float32, device=device),
        "counts": torch.zeros((), dtype=torch.int32, device=device),
    }


def safe_drain(
    garrison_status: PlanetGarrisonStatus,
    *,
    source_idx: Tensor,            # [S] long — planet slots to evaluate
    source_ships: Tensor,          # [S] float — current garrison at those slots
    H_eff: Tensor,                 # scalar float — horizon to protect the source over
    player_id: int = 0,
) -> Tensor:
    """Max ships a source can shed while staying held over ``H_eff``. ``[S]``.

    Closed form, no scoring. For every source slot, over the turns ``t = 1..H``
    where the do-nothing projection still has us holding the planet (``owner == me``,
    ``ships > 0``) within ``H_eff``, the largest amount we can remove now while the
    projected garrison stays non-negative on every such turn is
    ``min_t(ships_traj[t])`` — leaving the planet at 0 ships on the worst held turn
    is allowed. Capped by ``source_ships`` (can't send more than we hold now):

        safe_drain = clamp(min(min_t held(ships_traj), source_ships), 0)

    A *doomed* source (no turn is held within ``H_eff``) has nothing to protect:
    ``min_slack`` is ``+inf`` and the cap collapses to ``source_ships`` naturally.
    """
    S = source_idx.shape[0]
    ships_cache = garrison_status.ships
    dtype = ships_cache.dtype if ships_cache.is_floating_point() else torch.float32
    device = ships_cache.device

    H_axis = int(ships_cache.shape[-1])
    H = max(H_axis - 1, 0)
    P = int(ships_cache.shape[0])
    if H == 0:
        return torch.zeros(S, dtype=dtype, device=device)

    src_idx_safe = source_idx.clamp(min=0, max=max(P - 1, 0))

    src_ships_traj = ships_cache[src_idx_safe][..., 1:].to(dtype=dtype)          # [S, H]
    src_owner_traj = garrison_status.owner[src_idx_safe][..., 1:]                 # [S, H]
    me_owned = src_owner_traj == int(player_id)

    turn_grid = torch.arange(1, H + 1, device=device, dtype=dtype).view(1, H)
    within_horizon = turn_grid <= H_eff                                          # H_eff scalar

    held = me_owned & within_horizon & (src_ships_traj > 0.0)
    inf_fill = torch.full_like(src_ships_traj, float("inf"))
    cap_traj = torch.where(held, src_ships_traj, inf_fill)
    min_slack = cap_traj.min(dim=-1).values                                       # [S]
    return torch.minimum(min_slack, source_ships.to(dtype)).clamp(min=0.0)
