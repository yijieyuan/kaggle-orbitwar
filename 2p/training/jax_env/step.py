"""One tick of the JAX 2p forward sim (kaggle 1.30.1). Pure jnp, jit/vmap-able.

Order mirrors shared/sim/forward_sim._move_collide_swept + step():
  1. comet start-expire (idx>=len)        4. production (owner>=0)
  2. (comet spawn handled in env.py)       5+6. move + swept-collide -> arrivals
  3. launch (apply actions)                7. combat ; planet motion ; comet end-expire ; step+=1

Action = per-planet-slot launch (the policy is factored per source planet):
  launch (P,) bool, angle (P,) f32, ships (P,) int32  — slot i launches from planet i
  (owner = p_owner[i]); both players' choices are merged into these three arrays.
"""
import jax
import jax.numpy as jnp

from constants import LAUNCH_CLEARANCE, MAX_FLEETS, FLEET_CAP_PER_PLAYER, SUN_X, SUN_Y, N_COMET_SLOTS
from state import JaxState, fleet_speed_jax
from physics import swept_pair_hit, seg_hits_sun, in_board, planet_next_positions
from comet import spawn_inject

# First-hit walk length (jit can't early-exit). Bounded by the RECOMPUTE interval: a fleet's
# first-hit only changes when a comet appears, so we only need to cover until the next recompute.
# Comets spawn at 50/150/250/350/450 (every 100) -> comet-only recompute needs ~100; H=50 here
# assumes a periodic recompute every 50 turns (cheaper per walk). Carried after launch (NOT recomputed
# each step); expired fleets are already masked out by the env.
H_MAX_FIRSTHIT = 100      # comet period (comets every 100: 50/150/250/350/450) -> walk to next recompute


def _planet_traj(state, H):
    """Planet (x,y) at relative turns 0..H (orbital rotate / comet path / static). (H+1,P) each."""
    P = state.p_x.shape[0]
    ks = jnp.arange(H + 1).astype(jnp.float32)
    ang = state.p_orbital_a[None, :] + state.av * ks[:, None]                      # (H+1, P)
    orb_x = SUN_X + state.p_orbital_r[None, :] * jnp.cos(ang)
    orb_y = SUN_Y + state.p_orbital_r[None, :] * jnp.sin(ang)
    L = state.p_comet_path_x.shape[1]
    cidx = jnp.clip(state.p_comet_idx[None, :] + jnp.arange(H + 1)[:, None], 0, L - 1)   # (H+1, P)
    pcol = jnp.arange(P)[None, :]
    com_x = state.p_comet_path_x[pcol, cidx]                                       # (H+1, P)
    com_y = state.p_comet_path_y[pcol, cidx]
    bx = jnp.broadcast_to(state.p_x[None, :], (H + 1, P))
    by = jnp.broadcast_to(state.p_y[None, :], (H + 1, P))
    tx = jnp.where(state.p_is_comet[None, :], com_x, jnp.where(state.p_is_orbiting[None, :], orb_x, bx))
    ty = jnp.where(state.p_is_comet[None, :], com_y, jnp.where(state.p_is_orbiting[None, :], orb_y, by))
    return tx, ty


def predict_first_hits(state, sx, sy, angle, ships, H=H_MAX_FIRSTHIT):
    """REAL first-hit per source slot: walk the launched fleet to its FIRST event (planet > oob >
    sun), fixed H turns. Returns (target_slot (P,) = first-hit planet or -1, arrival_rel (P,) =
    turns-to-hit or -1). Mirrors shared.physics.first_hit_from (raw-k planet motion, list-order
    first-planet wins). Computed ONCE per fleet at launch (carried after)."""
    P = state.p_x.shape[0]
    p_mask, p_radius = state.p_mask, state.p_radius
    tx, ty = _planet_traj(state, H)
    sp = fleet_speed_jax(ships)
    vx = jnp.cos(angle) * sp
    vy = jnp.sin(angle) * sp

    def per_fleet(sx_i, sy_i, vx_i, vy_i):
        def turn(carry, k):
            done, slot, kind, arr = carry
            kf = k.astype(jnp.float32)
            ax = sx_i + (kf - 1.0) * vx_i; ay = sy_i + (kf - 1.0) * vy_i
            bxx = sx_i + kf * vx_i; byy = sy_i + kf * vy_i
            hit = swept_pair_hit(ax, ay, bxx, byy, tx[k - 1], ty[k - 1], tx[k], ty[k], p_radius) & p_mask
            any_hit = jnp.any(hit)
            ev = jnp.where(any_hit, 1, jnp.where(~in_board(bxx, byy), 2,
                           jnp.where(seg_hits_sun(ax, ay, bxx, byy), 3, 0))).astype(jnp.int32)
            new = (~done) & (ev > 0)
            slot = jnp.where(new, jnp.where(any_hit, jnp.argmax(hit).astype(jnp.int32), -1), slot)
            kind = jnp.where(new, ev, kind)
            arr = jnp.where(new, k.astype(jnp.int32), arr)
            return (done | (ev > 0), slot, kind, arr), None
        (_, slot, kind, arr), _ = jax.lax.scan(
            turn, (jnp.bool_(False), jnp.int32(-1), jnp.int32(0), jnp.int32(-1)), jnp.arange(1, H + 1))
        return jnp.where(kind == 1, slot, -1).astype(jnp.int32), arr.astype(jnp.int32)

    return jax.vmap(per_fleet)(sx, sy, vx, vy)


def recompute_comet_hits(state, H=H_MAX_FIRSTHIT):
    """Fix OLD in-flight fleets vs comets that appeared AFTER they launched: each fleet walks vs the
    COMET slots ONLY (<=4, cheap) and if a comet intercepts EARLIER than its cached arrival, retargets
    to it. New fleets already saw current comets in their launch walk -> not affected (intentional)."""
    P = state.p_x.shape[0]
    cs = P - N_COMET_SLOTS                                         # first reserved comet slot (= 44)
    tx, ty = _planet_traj(state, H)
    txc, tyc = tx[:, cs:], ty[:, cs:]                              # (H+1, 4) — ONLY the comet columns
    prc = state.p_radius[cs:]                                      # (4,)
    cmaskc = (state.p_is_comet & state.p_mask)[cs:]                # (4,)
    vx = state.f_speed * jnp.cos(state.f_angle)
    vy = state.f_speed * jnp.sin(state.f_angle)
    cached_rel = state.f_arrival - state.step                      # (F,) cached arrival, relative

    def per_fleet(fx, fy, vxx, vyy):
        def turn(carry, k):
            done, slot, arr = carry
            kf = k.astype(jnp.float32)
            ax = fx + (kf - 1.0) * vxx; ay = fy + (kf - 1.0) * vyy
            bxx = fx + kf * vxx; byy = fy + kf * vyy
            hit = swept_pair_hit(ax, ay, bxx, byy, txc[k - 1], tyc[k - 1], txc[k], tyc[k], prc) & cmaskc  # (4,)
            any_hit = jnp.any(hit)
            new = (~done) & any_hit
            slot = jnp.where(new, (cs + jnp.argmax(hit)).astype(jnp.int32), slot)
            arr = jnp.where(new, k.astype(jnp.int32), arr)
            return (done | any_hit, slot, arr), None
        (_, slot, arr), _ = jax.lax.scan(
            turn, (jnp.bool_(False), jnp.int32(-1), jnp.int32(-1)), jnp.arange(1, H + 1))
        return slot, arr

    c_slot, c_arr = jax.vmap(per_fleet)(state.f_x, state.f_y, vx, vy)
    earlier = state.f_mask & (c_arr >= 0) & ((cached_rel < 1) | (c_arr < cached_rel))
    return state._replace(
        f_target=jnp.where(earlier, c_slot, state.f_target),
        f_arrival=jnp.where(earlier, state.step + c_arr, state.f_arrival))


def step(state: JaxState, launch, angle, ships, target=None, arrival=None) -> JaxState:
    P = state.p_owner.shape[0]
    F = state.f_owner.shape[0]
    ar_P = jnp.arange(P)
    # target planet SLOT + absolute arrival turn for each launching source (projection metadata;
    # does NOT affect physics). Default -1 (e.g. parity_test, which only checks physics).
    if target is None:
        target = jnp.full((P,), -1, jnp.int32)
    if arrival is None:
        arrival = jnp.full((P,), -1, jnp.int32)

    # ---- 1. comet start-expire: comets whose path already ran out are gone ----
    p_mask = state.p_mask & ~(state.p_is_comet & (state.p_comet_idx >= state.p_comet_len))
    # ---- 2. comet spawn (turns 50/150/250/350/450): inject batch into reserved slots ----
    state = spawn_inject(state._replace(p_mask=p_mask))
    p_mask = state.p_mask
    state = recompute_comet_hits(state)        # exp13: OLD fleets re-check vs (possibly new) comets

    # ---- 3. launch: validate + spawn fleets into free slots ----
    valid = (launch & p_mask & (ships > 0) & (ships <= state.p_ships) & (state.p_owner >= 0))
    # exp24: per-player IN-FLIGHT FLEET CAP (TRAIN-ONLY regularizer; eval/deploy port rl_infer is uncapped).
    # A player already holding FLEET_CAP_PER_PLAYER in-flight has further launches REJECTED this tick; a
    # multi-launch tick crossing the cap drops the excess (planet-slot order, per-owner cumsum rank). This
    # bounds total in-flight <= 2*cap = MAX_FLEETS, so the global free-slot drop below never fires. (state.f_*
    # here is the CURRENT in-flight set; spawn_inject/recompute_comet_hits touch comets/targets, not f_owner.)
    _ex0 = jnp.sum(state.f_mask & (state.f_owner == 0))
    _ex1 = jnp.sum(state.f_mask & (state.f_owner == 1))
    _rank0 = jnp.cumsum((valid & (state.p_owner == 0)).astype(jnp.int32)) - 1   # 0-idx rank among p0's valids
    _rank1 = jnp.cumsum((valid & (state.p_owner == 1)).astype(jnp.int32)) - 1
    _cap_ok = jnp.where(state.p_owner == 0, _ex0 + _rank0 < FLEET_CAP_PER_PLAYER,
                        jnp.where(state.p_owner == 1, _ex1 + _rank1 < FLEET_CAP_PER_PLAYER, True))
    valid = valid & _cap_ok
    sp = fleet_speed_jax(ships)
    spawn_d = state.p_radius + LAUNCH_CLEARANCE
    nfx = state.p_x + spawn_d * jnp.cos(angle)
    nfy = state.p_y + spawn_d * jnp.sin(angle)
    # map each valid launch to a free fleet slot via cumsum ranks
    vrank = jnp.cumsum(valid.astype(jnp.int32)) - 1            # (P,) rank among valids
    free = ~state.f_mask                                       # (F,)
    frank = jnp.cumsum(free.astype(jnp.int32)) - 1            # (F,) rank among frees
    # rank -> free-slot index (scatter slot id into its rank; drop non-free)
    rank2slot = jnp.zeros((F,), jnp.int32).at[jnp.where(free, frank, F)].set(jnp.arange(F, dtype=jnp.int32), mode="drop")
    tgt = jnp.where(valid, rank2slot[jnp.clip(vrank, 0, F - 1)], F)   # F => dropped

    f_owner = state.f_owner.at[tgt].set(state.p_owner, mode="drop")
    f_x = state.f_x.at[tgt].set(nfx, mode="drop")
    f_y = state.f_y.at[tgt].set(nfy, mode="drop")
    f_angle = state.f_angle.at[tgt].set(angle, mode="drop")
    f_ships = state.f_ships.at[tgt].set(ships, mode="drop")
    f_speed = state.f_speed.at[tgt].set(sp, mode="drop")
    f_mask = state.f_mask.at[tgt].set(jnp.ones((P,), bool), mode="drop")
    # exp13: REAL first-hit (predicted swept walk) for the NEW fleets, stamped ONCE here; old fleets
    # keep their carried f_target/f_arrival (the passed `target`/`arrival` = chosen-tid are ignored).
    fh_slot, fh_arr = predict_first_hits(state, nfx, nfy, angle, ships)
    new_target = jnp.where(fh_arr >= 0, fh_slot, -1)
    new_arrival = jnp.where(fh_arr >= 0, state.step + fh_arr, -1)
    f_target = state.f_target.at[tgt].set(new_target, mode="drop")
    f_arrival = state.f_arrival.at[tgt].set(new_arrival, mode="drop")
    p_ships = state.p_ships - jnp.where(valid, ships, 0)
    next_fid = state.next_fleet_id + jnp.sum(valid.astype(jnp.int32))

    # ---- 4. production (owned planets, incl. captured comets) ----
    p_ships = p_ships + jnp.where((state.p_owner >= 0) & p_mask, state.p_prod, 0)

    # ---- 5+6. move fleets + swept-pair collision ----
    s2 = state._replace(p_mask=p_mask, p_ships=p_ships,
                        f_owner=f_owner, f_x=f_x, f_y=f_y, f_angle=f_angle,
                        f_ships=f_ships, f_speed=f_speed, f_mask=f_mask)
    ox, oy, pnx, pny, check = planet_next_positions(s2)        # (P,) each
    ax, ay = f_x, f_y
    bx = f_x + f_speed * jnp.cos(f_angle)
    by = f_y + f_speed * jnp.sin(f_angle)
    # (F,P) hit matrix: fleet segment vs each planet's old->new segment
    hit = swept_pair_hit(ax[:, None], ay[:, None], bx[:, None], by[:, None],
                         ox[None, :], oy[None, :], pnx[None, :], pny[None, :], state.p_radius[None, :])
    hit = hit & check[None, :] & p_mask[None, :] & f_mask[:, None]
    any_hit = jnp.any(hit, axis=1)                             # (F,)
    hit_planet = jnp.where(any_hit, jnp.argmax(hit, axis=1), -1)   # first planet in slot order
    oob = ~in_board(bx, by)
    sun = seg_hits_sun(ax, ay, bx, by)
    consumed = any_hit | oob | sun                            # fleet removed this tick
    survive = f_mask & ~consumed
    # update surviving fleets to B; mask off consumed
    f_x = jnp.where(survive, bx, f_x)
    f_y = jnp.where(survive, by, f_y)
    f_mask = survive

    # ---- 7. combat: aggregate arrivals per planet (by owner), resolve ----
    arrived = (hit_planet >= 0)
    oh = (hit_planet[:, None] == ar_P[None, :]) & arrived[:, None]   # (F,P)
    inc0 = jnp.sum(oh * (f_owner == 0)[:, None] * f_ships[:, None], axis=0)   # (P,)
    inc1 = jnp.sum(oh * (f_owner == 1)[:, None] * f_ships[:, None], axis=0)
    s0, s1 = inc0, inc1
    both = (s0 > 0) & (s1 > 0)
    tie = both & (s0 == s1)
    no_attack = (s0 == 0) & (s1 == 0)
    surv_owner = jnp.where(s0 > 0, jnp.where(s1 > 0, jnp.where(s0 >= s1, 0, 1), 0),
                           jnp.where(s1 > 0, 1, -1)).astype(jnp.int32)
    surv_ships = jnp.where(both, jnp.abs(s0 - s1), jnp.where(s0 > 0, s0, s1))
    surv_ships = jnp.where(tie, 0, surv_ships)
    has_surv = (~no_attack) & (~tie)
    same = has_surv & (surv_owner == state.p_owner)
    capture = has_surv & (surv_owner != state.p_owner) & (surv_ships > p_ships)
    repel = has_surv & (surv_owner != state.p_owner) & (surv_ships <= p_ships)
    new_ships = jnp.where(same, p_ships + surv_ships,
                jnp.where(capture, surv_ships - p_ships,
                jnp.where(repel, p_ships - surv_ships, p_ships)))
    new_owner = jnp.where(capture, surv_owner, state.p_owner)

    # ---- planet motion (apply now) + comet end-expire ----
    new_orbital_a = state.p_orbital_a + jnp.where(state.p_is_orbiting, state.av, 0.0)
    new_comet_idx = state.p_comet_idx + jnp.where(state.p_is_comet & p_mask, 1, 0)
    p_x2 = jnp.where(p_mask, pnx, state.p_x)
    p_y2 = jnp.where(p_mask, pny, state.p_y)
    p_mask2 = p_mask & ~(state.p_is_comet & (new_comet_idx >= state.p_comet_len))

    return state._replace(
        p_owner=new_owner, p_x=p_x2, p_y=p_y2, p_ships=new_ships, p_mask=p_mask2,
        p_orbital_a=new_orbital_a, p_comet_idx=new_comet_idx,
        f_owner=f_owner, f_x=f_x, f_y=f_y, f_angle=f_angle, f_ships=f_ships,
        f_speed=f_speed, f_mask=f_mask, f_target=f_target, f_arrival=f_arrival,
        step=state.step + 1, next_fleet_id=next_fid,
    )
