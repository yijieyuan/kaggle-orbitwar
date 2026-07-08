"""Fixed-fleet intercept aim — sub-turn-accurate angle for an orbiting target.

Solves the **continuous** intercept time ``t*`` (root of
``v·t = dist(target_pos(t), source) − gap`` with the target on its analytic
orbit), aims at ``target_pos(t*)``, and verifies that angle with a
fully-vectorized analytic first-contact check.

* **Root** — a continuous fixed-point iteration (no grid scan / argmax /
  bisection), free of grid-resolution artifacts.
* **Verify** — :func:`_analytic_first_contact` reproduces the engine's
  first-contact verdict exactly (swept-pair vs every planet, sun, bounds,
  lowest-slot same-step tie-break) with no engine state and no per-step loop.
  A shot is viable iff it contacts the target first.

Returns ``angle`` / ``eta`` / ``viable``.
"""
from __future__ import annotations

import torch
from torch import Tensor

from .geometry import fleet_speed
from .movement import PlanetMovement
from .movement_aiming import (
    LAUNCH_SURFACE_OFFSET,
    TARGET_HIT_SURFACE_OFFSET,
    _swept_pair_hit_mask,
)
from .constants import BOARD_SIZE, CENTER, SUN_RADIUS

_FP_ITERS = 6  # continuous fixed-point iterations for the intercept time
_BIG = 1_000_000.0


def intercept_angle(
    movement: PlanetMovement,
    source_slots: Tensor,
    target_slots: Tensor,
    fleet_sizes: Tensor,
    *,
    fp_iters: int = _FP_ITERS,
    active: Tensor | None = None,
) -> dict[str, Tensor]:
    """Continuous-intercept aim for a fixed fleet size (the root angle only).

    Broadcastable slot/size tensors in; ``{angle, eta, viable}`` out (same shape).
    Non-viable candidates get ``eta == inf``.

    ``active`` (optional, broadcastable to the candidate shape): a reachability
    precheck that gates the expensive body screen. The lead angle is still solved
    on the full grid, so kept candidates' angles are bit-identical; only the
    integer-exact first-contact screen is compacted to the active candidates.
    Candidates with ``active`` False resolve to non-viable. Pass a strict superset
    of viability (e.g. :func:`planner_core.reachable_mask`) for a zero-behaviour-change
    speedup — ``None`` screens everything.
    """
    dev = movement.device
    dt = movement.dtype
    H = int(movement.movement_horizon)

    src, tgt, ships = torch.broadcast_tensors(
        source_slots.to(device=dev),
        target_slots.to(device=dev),
        fleet_sizes.to(device=dev, dtype=dt),
    )
    shape = src.shape
    src = src.long().clamp(0, max(movement.P - 1, 0)).reshape(-1)
    tgt = tgt.long().clamp(0, max(movement.P - 1, 0)).reshape(-1)
    ships = ships.to(dt).clamp(min=1.0).reshape(-1)
    M = src.shape[0]

    sx, sy = movement.position_at_slots(src, 0)                       # [M]
    src_r = movement.radii[src]
    tgt_r = movement.radii[tgt]
    speed = fleet_speed(ships).clamp(min=1e-6)                        # [M]

    # Target orbit params from its integer positions: centre-relative radius +
    # phase at t=0 and the per-step angular step (auto-zero for static planets).
    t0x, t0y = movement.position_at_slots(tgt, 0)
    t1x, t1y = movement.position_at_slots(tgt, 1)
    R = torch.sqrt(((t0x - CENTER) ** 2 + (t0y - CENTER) ** 2).clamp(min=0.0))
    a0 = torch.atan2(t0y - CENTER, t0x - CENTER)
    a1 = torch.atan2(t1y - CENTER, t1x - CENTER)
    omega = torch.atan2(torch.sin(a1 - a0), torch.cos(a1 - a0))       # wrapped Δangle/step
    gap = src_r + LAUNCH_SURFACE_OFFSET + tgt_r + TARGET_HIT_SURFACE_OFFSET

    def target_pos(t: Tensor):
        ang = a0 + omega * t
        return CENTER + R * torch.cos(ang), CENTER + R * torch.sin(ang)

    # Continuous fixed point t = (dist(target_pos(t), src) - gap)/v, seeded with
    # the static-target estimate. A contraction whenever the target's radial speed
    # stays below the fleet speed (true for reachable shots); divergent guesses
    # just produce a bad angle that the verify rejects.
    d0 = torch.sqrt(((t0x - sx) ** 2 + (t0y - sy) ** 2).clamp(min=0.0))
    t_star = ((d0 - gap) / speed).clamp(min=0.0, max=float(H))
    for _ in range(int(fp_iters)):
        tx, ty = target_pos(t_star)
        d = torch.sqrt(((tx - sx) ** 2 + (ty - sy) ** 2).clamp(min=0.0))
        t_star = ((d - gap) / speed).clamp(min=0.0, max=float(H))

    tx, ty = target_pos(t_star)
    angle = torch.atan2(ty - sy, tx - sx)                             # [M]
    cos_a = torch.cos(angle)
    sin_a = torch.sin(angle)
    launch_x = sx + cos_a * (src_r + LAUNCH_SURFACE_OFFSET)           # [M]
    launch_y = sy + sin_a * (src_r + LAUNCH_SURFACE_OFFSET)

    # Relevant flight length = distance to the intercept (+margins for the arrival
    # step and the target radius). Bounds the broad-phase cull segment to the
    # fleet's actual launch→target path. Planets beyond the target can never be the
    # first contact for a target-reaching fleet, so this preserves `viable`
    # (contact==tgt) and the viable-case `eta` exactly.
    eta_cap = (t_star + 2.0).clamp(max=float(H))
    seg_len = speed * eta_cap + tgt_r + 2.0                            # [M]

    px = movement.x[: H + 1, :]                                       # [H+1, P] (already cached)
    py = movement.y[: H + 1, :]
    radii_p = movement.radii
    alive0 = movement.alive_at(0)
    if active is None:
        contact, eta_c = _analytic_first_contact(
            launch_x=launch_x, launch_y=launch_y, cos_a=cos_a, sin_a=sin_a,
            speed=speed, px=px, py=py, p_alive0=alive0,
            radii=radii_p, H=H, seg_len=seg_len,
        )                                                             # [M]
    else:
        # Reachability gate: screen only the active candidates. The per-candidate
        # integer contact/eta are shortlist-independent, so kept candidates' verdicts
        # are bit-identical to the full screen. Compact to the active candidates,
        # screen, then scatter home; inactive cells resolve to contact = -1.
        act = active.broadcast_to(shape).reshape(M).to(torch.bool)
        n_max = max(1, int(act.sum().item()))
        order = (~act).to(torch.int8).argsort(stable=True)           # active cells first
        midx = order[:n_max]                                         # [n_max]
        keep = act[midx]
        contact_m, eta_cm = _analytic_first_contact(
            launch_x=launch_x[midx], launch_y=launch_y[midx],
            cos_a=cos_a[midx], sin_a=sin_a[midx],
            speed=speed[midx], px=px, py=py, p_alive0=alive0,
            radii=radii_p, H=H, seg_len=seg_len[midx],
        )                                                            # [n_max]
        contact = torch.full((M,), -1, dtype=contact_m.dtype, device=dev)
        eta_c = torch.full((M,), float(H), dtype=eta_cm.dtype, device=dev)
        contact[midx] = torch.where(keep, contact_m, torch.full_like(contact_m, -1))
        eta_c[midx] = torch.where(keep, eta_cm, torch.full_like(eta_cm, float(H)))

    viable = contact == tgt                                           # [M]
    eta_out = torch.where(viable, eta_c.to(dt), torch.full_like(eta_c.to(dt), float("inf")))
    return {
        "angle": angle.reshape(shape),
        "eta": eta_out.reshape(shape),
        "viable": viable.reshape(shape),
    }


def _analytic_first_contact(
    *,
    launch_x: Tensor,
    launch_y: Tensor,
    cos_a: Tensor,
    sin_a: Tensor,
    speed: Tensor,
    px: Tensor,
    py: Tensor,
    p_alive0: Tensor,
    radii: Tensor,
    H: int,
    seg_len: Tensor | None = None,
    max_bytes: int = 256 * 1024 * 1024,
):
    """First planet a fleet contacts, engine-faithful, shaped ``[M, C]``.

    Reproduces batch ``_move_fleets`` exactly: straight fleet motion at ``speed``,
    swept-pair collision vs every step-0-alive planet, OOB + point-to-segment sun
    kill (only when no planet was hit that step), and the lowest-slot same-step
    tie-break. ``launch_*``/``cos_a``/``sin_a``/``speed`` are ``[M]``; ``px``,
    ``py`` are ``[H+1, P]`` planet positions per step; ``p_alive0`` is ``[P]``
    (step-0 alive); ``radii`` is ``[P]``.
    Returns ``(contact_slot, eta)`` — ``contact_slot == -1`` and ``eta == H`` when
    the fleet contacts no planet (or dies first).

    Two-phase to keep the exact swept-pair off the common clear-shot path:

    * **Broad phase** — an AABB cull (the fleet's full-horizon segment box vs each
      planet's swept box inflated by its radius). A planet whose box can't overlap
      the segment can never be hit, so it's dropped. The per-candidate shortlist
      collapses to the few real near-path planets (~1-3 for a clear shot vs ``P``).
      Conservative → the kept set always contains every hittable planet, so the
      result is **byte-identical** to checking all ``P``.
    * **Narrow phase** — the exact swept-pair only on the shortlisted planets,
      flattened to ``N = M`` candidates and run in byte-budgeted chunks (the
      dense ``[N,K,H]`` form would OOM when the regroup grid makes ``M`` large).

    ``amin`` reductions are order-independent so chunking/culling don't perturb the
    values (byte-exact + CPU≡CUDA guarantees hold). Runs eager; the one host sync
    (max shortlist length) is cheap.
    """
    M = cos_a.shape[0]
    P = px.shape[-1]
    dev = cos_a.device
    dt = launch_x.dtype
    N = M
    big = _BIG

    lx = launch_x.reshape(N); ly = launch_y.reshape(N)
    ca = cos_a.reshape(N); sa = sin_a.reshape(N); sp = speed.reshape(N)

    # --- Broad phase: AABB cull (no time axis → cheap). The conservative segment box
    # runs launch → launch + u·seg_len, where seg_len bounds the fleet's relevant
    # flight (distance to the intercept; falls back to the full horizon v·H). The
    # planet box is its swept extent over [0,H] inflated by its radius. ---
    slen = (sp * float(H)) if seg_len is None else seg_len.reshape(N)
    end_x = lx + ca * slen; end_y = ly + sa * slen
    seg_xmin = torch.minimum(lx, end_x); seg_xmax = torch.maximum(lx, end_x)   # [N]
    seg_ymin = torch.minimum(ly, end_y); seg_ymax = torch.maximum(ly, end_y)
    bb_xmin = px.amin(0) - radii                                              # [P]
    bb_xmax = px.amax(0) + radii
    bb_ymin = py.amin(0) - radii
    bb_ymax = py.amax(0) + radii
    keep = ~(
        (seg_xmax.unsqueeze(1) < bb_xmin) | (seg_xmin.unsqueeze(1) > bb_xmax)
        | (seg_ymax.unsqueeze(1) < bb_ymin) | (seg_ymin.unsqueeze(1) > bb_ymax)
    )                                                                          # [N, P]
    K = max(1, int(keep.sum(1).amax().item()))            # one host sync (eager-cheap)
    order = (~keep).to(torch.int8).argsort(dim=1, stable=True)                 # kept first
    shortlist = order[:, :K]                                                   # [N, K]
    valid = keep.gather(1, shortlist)                                          # [N, K]

    k = torch.arange(H + 1, device=dev, dtype=dt)                              # [H+1]
    t_ax = torch.arange(H + 1, device=dev).view(1, H + 1, 1)                   # [1,H+1,1]
    step_h = torch.arange(1, H + 1, device=dev, dtype=dt).view(1, H, 1)        # [1,H,1]

    # ~16 float intermediates of [chunk, H, K] dominate; budget the largest tensor.
    bytes_per = max(1, 16 * H * K * 4)
    chunk = max(4096, max_bytes // bytes_per)
    chunk = min(chunk, max(N, 1))

    contacts: list[Tensor] = []
    etas: list[Tensor] = []
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        sl = shortlist[s:e]                                                   # [n, K]
        fx = lx[s:e].view(-1, 1) + ca[s:e].view(-1, 1) * sp[s:e].view(-1, 1) * k   # [n, H+1]
        fy = ly[s:e].view(-1, 1) + sa[s:e].view(-1, 1) * sp[s:e].view(-1, 1) * k
        # advanced-index the K shortlisted planets directly → [n, H+1, K] (no [n,H+1,P])
        sl_e = sl.view(-1, 1, K)
        pxc = px[t_ax, sl_e]                                                  # [n, H+1, K]
        pyc = py[t_ax, sl_e]
        radc = radii[sl]                                                      # [n, K]
        alivec = p_alive0[sl] & valid[s:e]                                    # [n, K]
        real_slot = sl.to(dt)                                                 # [n, K]

        fx0 = fx[:, :-1].unsqueeze(-1); fy0 = fy[:, :-1].unsqueeze(-1)        # [n,H,1]
        fx1 = fx[:, 1:].unsqueeze(-1);  fy1 = fy[:, 1:].unsqueeze(-1)
        hit = _swept_pair_hit_mask(
            fx0, fy0, fx1, fy1,
            pxc[:, :-1, :], pyc[:, :-1, :], pxc[:, 1:, :], pyc[:, 1:, :],
            radc.unsqueeze(1),
        )                                                                     # [n,H,K]
        hit = hit & alivec.unsqueeze(1)

        planet_hit_step = torch.where(hit, step_h, torch.full_like(step_h, big)).amin(1)  # [n,K]
        first_planet_step = planet_hit_step.amin(1)                           # [n]
        is_first = planet_hit_step == first_planet_step.unsqueeze(-1)
        contact_planet = torch.where(is_first, real_slot, torch.full_like(real_slot, big)).amin(1)  # [n]

        # env death: OOB at the new position OR the segment grazes the sun (static).
        nfx = fx[:, 1:]; nfy = fy[:, 1:]; ofx = fx[:, :-1]; ofy = fy[:, :-1]   # [n,H]
        oob = (nfx < 0) | (nfx > BOARD_SIZE) | (nfy < 0) | (nfy > BOARD_SIZE)
        vx = nfx - ofx; vy = nfy - ofy
        wx = CENTER - ofx; wy = CENTER - ofy
        vv = (vx * vx + vy * vy).clamp(min=1e-12)
        t = ((wx * vx + wy * vy) / vv).clamp(0.0, 1.0)
        cxp = ofx + t * vx; cyp = ofy + t * vy
        sun = ((cxp - CENTER) ** 2 + (cyp - CENTER) ** 2) < (SUN_RADIUS * SUN_RADIUS)
        env = oob | sun                                                       # [n,H]
        death_step = torch.where(env, step_h.squeeze(-1), torch.full_like(env, big, dtype=dt)).amin(1)  # [n]

        # Planet collision resolves BEFORE env removal in the same step (<=).
        ht = (first_planet_step <= death_step) & (first_planet_step < big)
        contacts.append(torch.where(ht, contact_planet, torch.full_like(contact_planet, -1.0)).long())
        etas.append(torch.where(ht, first_planet_step, torch.full_like(first_planet_step, float(H))))

    contact = (contacts[0] if len(contacts) == 1 else torch.cat(contacts)).view(M)
    eta = (etas[0] if len(etas) == 1 else torch.cat(etas)).view(M)
    return contact, eta
