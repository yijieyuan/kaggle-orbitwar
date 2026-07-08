"""exp19 lead-angle + K=6 reachability (vmap/jit-able).

ACTION SPACE (exp19 design): 6 fraction options per (source,target):
  bins 0..4: send round(frac * garrison) with frac in {1/4, 1/3, 1/2, 3/4, 1}
  bin  5   : PRECISE-KILL — send exactly (projected garrison of the target AT ARRIVAL) + 1,
             so the surviving attacker just captures (passive forecast). The ship count and the
             arrival turn are mutually dependent (fleet speed depends on size), resolved with a
             2-iteration fixed point seeded from the 1/2-fraction arrival estimate.
             Infeasible (need<1, need>my garrison, arrival>50-turn forecast horizon, lead not
             converged/swept-verified) -> masked like any other bin -> the policy's choice is
             REJECTED -> no launch (no launch head in exp19; hold = self-target). NOTE this module
             is SEAT-INDEPENDENT pure geometry; the seat-side rules (own targets = reinforcement
             via bins 0-4 ONLY, bin5 masked on own) live in train.py's Rm mask (2026-06-11).

reachability6(state, gar_raw) -> (R, ANG, TURNS, SHIPS) each (P,P,6); gar_raw from
env.garrison_forecast_raw (RAW projected garrison (P,50), seat-independent).
solve_lead / lead grids ported unchanged from the parity-verified 17v12 set.
"""
import jax
import jax.numpy as jnp

from constants import SUN_X, SUN_Y, LAUNCH_CLEARANCE, EPISODE_STEPS
from state import fleet_speed_jax
from physics import seg_hits_sun, swept_pair_hit

LEAD_ITERS = 5
LEAD_POS_TOL = 0.5
FRACS = (0.25, 1.0 / 3.0, 0.5, 0.75, 1.0)   # bins 0..4
K_BINS = 6                                   # 5 fractions + precise-kill
KILL_ITERS = 2                               # ships<->arrival fixed point for bin 5
FORECAST_H = 50


def solve_lead(sx, sy, sr, ships,
               tcx, tcy, torbr, torba, t_is_orb, t_is_com,
               tcpx, tcpy, tcidx, tclen, trad, av):
    """Scalar lead solve (vmap over (src,tgt)). Returns (angle, turns int32, sun_blocked, converged).
    Verbatim from 17v12 (parity-verified lineage)."""
    sp = fleet_speed_jax(ships)
    lc = sr + LAUNCH_CLEARANCE
    L = tcpx.shape[0]

    def estimate(px, py):
        ang = jnp.arctan2(py - sy, px - sx)
        ca = jnp.cos(ang); sa = jnp.sin(ang)
        lx = sx + ca * lc; ly = sy + sa * lc
        center_d = jnp.sqrt((px - sx) ** 2 + (py - sy) ** 2)
        hit_d = jnp.maximum(0.0, center_d - lc - trad)
        ex = lx + ca * hit_d; ey = ly + sa * hit_d
        sun_blk = seg_hits_sun(lx, ly, ex, ey)
        turns = jnp.maximum(1, jnp.ceil(hit_d / sp).astype(jnp.int32))
        return ang, turns, sun_blk

    def predict(steps):
        eff = jnp.maximum(steps - 1, 0).astype(jnp.float32)
        a = torba + av * eff
        ox = SUN_X + torbr * jnp.cos(a); oy = SUN_Y + torbr * jnp.sin(a)
        nidx = jnp.clip(tcidx + eff.astype(jnp.int32), 0, L - 1)
        comx = tcpx[nidx]; comy = tcpy[nidx]
        px = jnp.where(t_is_com, comx, jnp.where(t_is_orb, ox, tcx))
        py = jnp.where(t_is_com, comy, jnp.where(t_is_orb, oy, tcy))
        return px, py

    ang, turns, sun_blk = estimate(tcx, tcy)
    px, py = tcx, tcy
    prev_turns = turns; prev_px = px; prev_py = py
    for _ in range(LEAD_ITERS):
        prev_turns = turns; prev_px = px; prev_py = py
        px, py = predict(turns)
        ang, turns, sun_blk = estimate(px, py)
    pos_d = jnp.sqrt((px - prev_px) ** 2 + (py - prev_py) ** 2)
    converged = (jnp.abs(turns - prev_turns) <= 1) & (pos_d < LEAD_POS_TOL)
    return ang, turns, sun_blk, converged


def _lead_grid_pp(state, ships_pp):
    """(P,P) lead solve where ships_pp is PER-PAIR (P,P) — needed by the precise-kill bin
    (each (s,t) sends a different count). Fraction bins pass a broadcast per-source count."""
    px, py, pr = state.p_x, state.p_y, state.p_radius
    orbr, orba = state.p_orbital_r, state.p_orbital_a
    isorb, iscom = state.p_is_orbiting, state.p_is_comet
    cpx, cpy = state.p_comet_path_x, state.p_comet_path_y
    cidx, clen = state.p_comet_idx, state.p_comet_len
    av = state.av; P = px.shape[0]; si = jnp.arange(P)

    def per_pair(s, t):
        return solve_lead(px[s], py[s], pr[s], ships_pp[s, t],
                          px[t], py[t], orbr[t], orba[t], isorb[t], iscom[t],
                          cpx[t], cpy[t], cidx[t], clen[t], pr[t], av)
    return jax.vmap(lambda s: jax.vmap(lambda t: per_pair(s, t))(si))(si)


def _planet_pos_grid(state, kgrid):
    px, py = state.p_x, state.p_y
    orbr, orba = state.p_orbital_r, state.p_orbital_a
    isorb, iscom = state.p_is_orbiting, state.p_is_comet
    cpx, cpy = state.p_comet_path_x, state.p_comet_path_y
    cidx = state.p_comet_idx; av = state.av
    P = px.shape[0]; L = cpx.shape[1]
    kf = kgrid.astype(jnp.float32)
    a = orba[None, :] + av * kf
    ox = SUN_X + orbr[None, :] * jnp.cos(a); oy = SUN_Y + orbr[None, :] * jnp.sin(a)
    ci = jnp.clip(cidx[None, :] + kgrid.astype(jnp.int32), 0, L - 1)
    cols = jnp.broadcast_to(jnp.arange(P)[None, :], (P, P))
    comx = cpx[cols, ci]; comy = cpy[cols, ci]
    bx = jnp.broadcast_to(px[None, :], (P, P)); by = jnp.broadcast_to(py[None, :], (P, P))
    tx = jnp.where(iscom[None, :], comx, jnp.where(isorb[None, :], ox, bx))
    ty = jnp.where(iscom[None, :], comy, jnp.where(isorb[None, :], oy, by))
    return tx, ty


def _swept_verify(state, ang, turns, sp_pp):
    """(P,P) bool: env swept_pair_hit fires near the lead arrival. sp_pp = per-pair speeds (P,P)."""
    px, py, pr = state.p_x, state.p_y, state.p_radius
    ca = jnp.cos(ang); sa = jnp.sin(ang)
    lc = pr[:, None] + LAUNCH_CLEARANCE
    spx = px[:, None] + lc * ca; spy = py[:, None] + lc * sa
    vx = sp_pp * ca; vy = sp_pp * sa
    trad = pr[None, :]

    def at(dk):
        k = jnp.maximum(turns + dk, 1)
        fax = spx + (k - 1) * vx; fay = spy + (k - 1) * vy
        fbx = spx + k * vx; fby = spy + k * vy
        tax, tay = _planet_pos_grid(state, k - 1)
        tbx, tby = _planet_pos_grid(state, k)
        return swept_pair_hit(fax, fay, fbx, fby, tax, tay, tbx, tby, trad)
    return at(-1) | at(0) | at(1)


def _validated(state, ships_pp, extra_ok=None):
    """Full validation for a (P,P) ship-count grid -> (R, ang, turns)."""
    remaining = EPISODE_STEPS - 1 - state.step
    iscom = state.p_is_comet; cidx = state.p_comet_idx; clen = state.p_comet_len
    ang, turns, sun_blk, conv = _lead_grid_pp(state, ships_pp)
    comet_alive = (~iscom[None, :]) | ((cidx[None, :] + turns) < clen[None, :])
    sv = _swept_verify(state, ang, turns, fleet_speed_jax(ships_pp))
    R = (~sun_blk) & conv & (turns <= remaining) & state.p_mask[None, :] & comet_alive & sv & (ships_pp > 0)
    if extra_ok is not None:
        R = R & extra_ok
    return R, ang, turns.astype(jnp.int32)


def _gather_garrison(gar_raw, turns):
    """gar_raw (P,H) RAW projected garrison of each TARGET; turns (P,P) arrival -> (P,P) garrison at
    arrival (index turns-1; ships_h[h] = garrison after h+1 turns). Clipped into the horizon."""
    H = gar_raw.shape[1]
    idx = jnp.clip(turns - 1, 0, H - 1)                       # (P,P)
    return gar_raw[jnp.arange(gar_raw.shape[0])[None, :], idx]   # target = COLUMN -> gar_raw[t, idx[s,t]]


def reachability6(state, gar_raw):
    """(R, ANG, TURNS, SHIPS) each (P,P,6). Bins 0-4 fractions of source garrison; bin 5 precise-kill."""
    P = state.p_x.shape[0]
    ps = state.p_ships; psf = ps.astype(jnp.float32)
    fr = jnp.asarray(FRACS, jnp.float32)

    def frac_bin(k):
        ships_src = jnp.clip(jnp.round(fr[k] * psf).astype(jnp.int32), 0, ps)        # (P,)
        ships_pp = jnp.broadcast_to(ships_src[:, None], (P, P))
        R, ang, turns = _validated(state, ships_pp)
        return R, ang, turns, ships_pp

    R5f, A5f, T5f, S5f = jax.vmap(frac_bin)(jnp.arange(5))                            # (5,P,P) each

    # ---- bin 5: precise-kill, 2-iteration ships<->arrival fixed point + CLOSURE check ----
    # seed arrival with the 1/2-fraction estimate (bin 2)
    turns_est = T5f[2]
    need = jnp.ceil(_gather_garrison(gar_raw, turns_est)).astype(jnp.int32) + 1       # garrison@arrival + 1
    for _ in range(KILL_ITERS - 1):
        _, _, turns_i = _validated(state, jnp.clip(need, 1, None))
        need = jnp.ceil(_gather_garrison(gar_raw, turns_i)).astype(jnp.int32) + 1
    feas = (need >= 1) & (need <= ps[:, None])                                        # can actually send it
    Rk, Ak, Tk = _validated(state, jnp.where(feas, need, 1), extra_ok=feas)
    # CLOSURE: the garrison at the FINAL arrival turn (the Tk we report, for the need-sized fleet)
    # must reproduce the need we computed — otherwise need/TURNS/garrison-feature disagree by the
    # fixed point not settling (speed change shifted arrival a turn). Non-closed pairs are MASKED
    # ("if it can't be done exactly -> don't launch"), keeping bin5 exact by construction and the
    # garrison-feature alignment (verify B4) an identity.
    need_chk = jnp.ceil(_gather_garrison(gar_raw, Tk)).astype(jnp.int32) + 1
    Rk = Rk & (need_chk == need) & (Tk <= FORECAST_H)
    Sk = need

    R = jnp.concatenate([jnp.transpose(R5f, (1, 2, 0)), Rk[:, :, None]], axis=2)      # (P,P,6)
    ANG = jnp.concatenate([jnp.transpose(A5f, (1, 2, 0)), Ak[:, :, None]], axis=2)
    TURNS = jnp.concatenate([jnp.transpose(T5f, (1, 2, 0)), Tk[:, :, None]], axis=2)
    SHIPS = jnp.concatenate([jnp.transpose(S5f, (1, 2, 0)), Sk[:, :, None]], axis=2)
    return R, ANG, TURNS, SHIPS


def lead_for_ships(state, ships_src):
    """exp19 v2 (continuous Beta fraction): full validated lead solve for ONE per-source ship
    count (the SAMPLED fleet sizes). Returns (R, ANG, TURNS) each (P,P) — same validation chain
    as the discrete bins (converged + sun + in-time + comet-alive + swept-verified)."""
    P = state.p_x.shape[0]
    ships_pp = jnp.broadcast_to(ships_src[:, None], (P, P))
    return _validated(state, ships_pp)


def reach_probe4(state):
    """Pointer-legality reach probe (2026-06-11): 4 PARALLEL validated solves at the K4 ceil
    counts {ceil(g/4), ceil(g/2), ceil(3g/4), g} -> (P,P) ANY-reach. Measured on 6 boards:
    covers 98.1% of the old 6-bin union (single g-probe was only 90.7%); the residual gap is
    narrow speed-window geometry that the executed-count solve rejects at runtime anyway."""
    P = state.p_x.shape[0]
    ps = state.p_ships; psf = ps.astype(jnp.float32)

    def one(k):
        ships = jnp.clip(jnp.ceil(psf * (k + 1) / 4.0).astype(jnp.int32), 0, ps)
        R, _, _ = _validated(state, jnp.broadcast_to(ships[:, None], (P, P)))
        return R
    return jnp.any(jax.vmap(one)(jnp.arange(4)), axis=0)


def reach_probe_static(state):
    """Pointer-legality mask (2026-06-11 STATIC-EXACT design, user decision):
    - STATIC targets: EXACT — one full-garrison solve decides ∃-reachability (for a fixed target
      the launch ray is speed-independent, so sun-block is count-invariant, and arrival time is
      monotone in fleet size -> the fastest fleet decides). Measured gap vs 6-bin union: 0/2828.
    - MOVING targets (orbiting | comet): NEVER masked — speed-window intercept geometry is left
      entirely to the post-sample executed-count solve + first-hit gate. Measured cost: ~49% of
      always-legal moving pairs are unreachable at any probe count (junk options the policy must
      learn to avoid); accepted for ZERO false negatives ("if it can be hit, it is selectable").
    """
    P = state.p_x.shape[0]
    R, _, _ = _validated(state, jnp.broadcast_to(state.p_ships[:, None], (P, P)))
    moving = state.p_is_orbiting | state.p_is_comet
    return jnp.where(moving[None, :], state.p_mask[None, :], R)


def reach_solve_static(state):
    """exp21 hot-path SHARE (2026-06-14): ONE full-garrison _validated solve -> (R, ANG, TURNS) AND the
    pointer-legality mask Rg. BYTE-IDENTICAL to calling lead_for_ships(state, state.p_ships) for
    (R,ANG,TURNS) AND reach_probe_static(state) for Rg separately — both use the SAME full-garrison
    ship count, so the (P,P) lead grid + swept-verify was being solved TWICE. This solves it once and
    derives both (Rg = reach_probe_static's moving-target override applied to the same R)."""
    P = state.p_x.shape[0]
    R, ANG, TURNS = _validated(state, jnp.broadcast_to(state.p_ships[:, None], (P, P)))
    moving = state.p_is_orbiting | state.p_is_comet
    Rg = jnp.where(moving[None, :], state.p_mask[None, :], R)
    return R, ANG, TURNS, Rg
