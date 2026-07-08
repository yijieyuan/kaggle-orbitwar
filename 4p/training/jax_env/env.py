"""exp20 v37 4p-IL env: 4-PLAYER (FFA) port of the v37 econ-CNN feature layer.

This is the 4-player port of the 2p exp20-v37 env (econ-CNN agent: static planet_dyn +
me-vs-opp econ LEAD curves fed to a 1D-conv econ encoder). The geometry / N-way forecast /
seat-canonical machinery are copied VERBATIM from the proven, machine-verified 17v12-4p env
(verify_17v12_4p_rotation.py gate A7); only the FEATURE channels are the v37 econ-CNN family
(NOT the exp022-4p / exp023-imitate-4p 61-d-glob family).

PUBLIC API kept BYTE-COMPATIBLE with the 2p v37 env so train.py (edge) + train_il_v5.py import
unchanged: `_forecast(state, H)` is SEAT-INDEPENDENT and returns RAW (un-/500) ships (so the edge /
targeting path keeps its contract); `garrison_forecast_raw`; `basic_features(state, me, fc=None)`
returns the **5-tuple** (static, ts, glob, m, econ_curves). `me` is the acting SEAT id in {0,1,2,3}
(IL uses {0,3}: the two diagonal seats the 2p data is remapped onto — owner 0 -> seat 0, owner 1 -> seat 3).

4p CANONICAL VIEW (C4 ROTATIONAL, 90deg about (50,50) — the board's TRUE 4p symmetry; NOT mirror).
Seat homes sit at FIXED quadrants (obs coords, screen y-down):
    seat 0 -> BR (x>50,y>50)   seat 1 -> BL (x<50,y>50)
    seat 2 -> TR (x>50,y<50)   seat 3 -> TL (x<50,y<50)
with seat equivalences p1=R+90(p0), p2=R-90(p0), p3=R180(p0); ring order along av>0 is
0 -> 1 -> 3 -> 2 -> 0. `_rot_xy`/`_rot_vec` map acting seat `me`'s home to the canonical TOP-LEFT
anchor; `_ring_role(owner, me) = (RINGPOS[owner]-RINGPOS[me])%4` (role 0=me, 1=clockwise/next-on-ring/
canonical TR, 2=antipodal/canonical BR, 3=ccw/prev/canonical BL) so q1/q2/q3 ALWAYS mean the same
relative opponent in every seat's view.

FEATURE DIMS (4p expansion of the 2p v37 layout; the per-channel breakdown is in basic_features):
  static (P, 30)      ts (P, 50, 10)      glob (34,)      econ_curves (50, 8)      mask (P,)
2p->4p owner expansions:
  * static owner 3-hot [mine,opp,neu] -> 5-hot [mine,q1,q2,q3,neu]  (+2)
  * static planet_dyn 8 -> 13: flip_player {me,op} 2-ch -> flip_to 4-hot [me,q1,q2,q3] (+2);
    final_owner {me,op} 2-ch -> final_owner 5-hot [me,q1,q2,q3,neu] (+3)  -> 17 base + 13 = 30
  * ts owner {own_me,own_op} 2-ch -> 5 role channels [mine,q1,q2,q3,neu]  (+3)  -> 10
  * econ_curves [ship_lead, prod_lead] (50,2) -> per-role [ship,prod]x{me,q1,q2,q3} (50,8)
  * glob 28 -> 34: base 12 -> 18 (ts/tp/pf each 2->4 per-role), econ scalars 14 -> 12
    (_econ_feats 7->6, me-vs-maxopp), afl 2 -> 4 (per-role in-flight count).
REWARD is FFA per seat (terminal_reward_all); is_done = 4-alive check. gen_init_states uses reset(4).
"""
import os
import sys
import random

import numpy as np
import jax
import jax.numpy as jnp

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from constants import (MAX_PLANETS, MAX_FLEETS, EPISODE_STEPS, COMET_SPAWN_STEPS,
                       SUN_X, SUN_Y, N_PLAYERS)
from state import JaxState, from_kaggle_obs, fleet_speed_jax
import comet as cometmod

MAX_SPEED = 6.0
FORECAST_H = 50
INFLIGHT_CAP_PER_PLAYER = 128   # per-player launch cap (4*128 == MAX_FLEETS=512); used only to /norm afl
N_STATIC = 30      # 17 base (5-hot owner + 3-hot type + 9 geom) + 13 planet_dyn (4p one-hots)
N_GLOBAL = 34      # 18 base (phase 4 + ts/tp/pf 4 each + territory 2) + 12 econ scalars + 4 afl
TS_C = 10          # proj_ships + 5 role channels [mine,q1,q2,q3,neu] + exists + (x,y) + ramp
N_ECON = 8         # econ_curves channels: per-role [ship,prod] x {me,q1,q2,q3}


# ring position along the av>0 direction (0 -> 1 -> 3 -> 2 -> 0), indexed by owner id
RINGPOS = jnp.asarray([0, 1, 3, 2], jnp.int32)


def _rot_xy(x, y, me):
    """Canonical POSITION map (obs -> canonical): the C4 rotation about (50,50) that sends seat
    `me`'s home quadrant to TOP-LEFT.  me=3: identity | me=0: R180 (100-x,100-y) |
    me=1: R+90 (100-y,x) | me=2: R-90 (y,100-x).  Branchless (me is a traced int scalar).
    [VERBATIM from 17v12-4p]"""
    me = jnp.asarray(me, jnp.int32)
    cx = jnp.where(me == 3, x, jnp.where(me == 0, 100.0 - x, jnp.where(me == 1, 100.0 - y, y)))
    cy = jnp.where(me == 3, y, jnp.where(me == 0, 100.0 - y, jnp.where(me == 1, x, 100.0 - x)))
    return cx, cy


def _rot_vec(vx, vy, me):
    """Canonical VECTOR map: linear part of _rot_xy (same rotation, no translation) for
    velocities/displacements.  me=3: (vx,vy) | me=0: (-vx,-vy) | me=1: (-vy,vx) | me=2: (vy,-vx).
    [VERBATIM from 17v12-4p]"""
    me = jnp.asarray(me, jnp.int32)
    cvx = jnp.where(me == 3, vx, jnp.where(me == 0, -vx, jnp.where(me == 1, -vy, vy)))
    cvy = jnp.where(me == 3, vy, jnp.where(me == 0, -vy, jnp.where(me == 1, vx, -vx)))
    return cvx, cvy


def _ring_role(o, me_i):
    """Opponent ROLE = ring distance (RINGPOS[o] - RINGPOS[me]) % 4 along the av>0 ring 0->1->3->2->0.
    role 0 = me; 1 = clockwise neighbor (canonical TR); 2 = antipodal (canonical BR); 3 = ccw neighbor
    (canonical BL). Only meaningful for o >= 0 — callers pre-guard with jnp.maximum(o, 0) and AND the
    result with validity masks. [VERBATIM from 17v12-4p]"""
    return jnp.mod(RINGPOS[o] - RINGPOS[me_i], 4)


def gen_init_states(n_envs: int, seed: int = 0) -> JaxState:
    """n_envs initial (step-0, comet-free) 4-PLAYER JaxStates from the kaggle env (host-side)."""
    from kaggle_environments import make
    states = []
    for i in range(n_envs):
        random.seed(seed + i)                       # determinism: seed BEFORE make()
        env = make("orbit_wars", configuration={"episodeSteps": 500})
        env.reset(N_PLAYERS)
        env.step([[]] * N_PLAYERS)                  # board (planets/ownership) generated on the FIRST step
        obs = env.steps[1][0]["observation"]
        st = from_kaggle_obs(obs)
        sched = cometmod.gen_schedule(obs["initial_planets"], obs["angular_velocity"], seed + i)
        states.append(cometmod.attach_schedule(st, sched))
    return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs, axis=0), *states)


def _forecast(state: JaxState, H=FORECAST_H):
    """SEAT-INDEPENDENT N-WAY (4-owner) passive projection (production + in-flight arrivals + captures,
    mirror of step.py's N-way combat). Returns (ships_raw (P,H) UNNORMALIZED garrison, owner (P,H) int
    actual-owner ids 0..3/-1, exists (P,H)).

    ships_raw is RAW (NOT /500) so the seat-independent edge/targeting path (edge_features,
    garrison_forecast_raw, _gather_garrison) keeps its v37 contract VERBATIM. The N-way one_turn is
    17v12-4p's; the inflow/segment_sum/exists scaffolding is the 2p v37 _forecast's (extended to NP)."""
    f32 = jnp.float32
    P = state.p_x.shape[0]
    turns_remain = state.f_arrival - state.step
    has_t = state.f_mask & (state.f_target >= 0) & (turns_remain >= 1) & (turns_remain <= H)
    a_idx = jnp.clip(turns_remain - 1, 0, H - 1)
    t_idx = jnp.clip(state.f_target, 0, P - 1)
    flat = jnp.where(has_t, a_idx * P + t_idx, H * P)

    def _inflow(owner_val):
        c = jnp.where(has_t & (state.f_owner == owner_val), state.f_ships.astype(f32), 0.0)
        return jax.ops.segment_sum(c, flat, num_segments=H * P + 1)[:H * P].reshape(H, P)
    incs = jnp.stack([_inflow(k) for k in range(N_PLAYERS)], axis=1)              # (H, NP, P)
    p_prod = state.p_prod.astype(f32)

    def one_turn(carry, inc):                                                     # inc (NP,P): arrivals per owner
        ships, owner = carry
        ships = ships + jnp.where((owner >= 0) & state.p_mask, p_prod, 0.0)       # production (players only)
        desc = -jnp.sort(-inc, axis=0)                                           # (NP,P) descending per planet
        top = desc[0]; second = desc[1]
        top_owner = jnp.argmax(inc, axis=0).astype(jnp.int32)                     # first owner holding the max
        has_surv = (top > 0) & (top > second)                                    # strict winner
        surv_owner = jnp.where(has_surv, top_owner, -1).astype(jnp.int32)
        surv = jnp.where(has_surv, top - second, 0.0)
        same = has_surv & (surv_owner == owner)
        capture = has_surv & (surv_owner != owner) & (surv > ships)
        repel = has_surv & (surv_owner != owner) & (surv <= ships)
        new_ships = jnp.where(same, ships + surv,
                    jnp.where(capture, surv - ships,
                    jnp.where(repel, ships - surv, ships)))
        new_owner = jnp.where(capture, surv_owner, owner)
        return (new_ships, new_owner), (new_ships, new_owner)

    _, (ships_h, owner_h) = jax.lax.scan(
        one_turn, (state.p_ships.astype(f32), state.p_owner), incs)
    hrange = jnp.arange(H, dtype=jnp.int32)
    alive = (~state.p_is_comet[:, None]) | ((state.p_comet_idx[:, None] + hrange[None, :] + 1) < state.p_comet_len[:, None])
    exists = (state.p_mask[:, None] & alive).astype(f32)
    return ships_h.T * exists, owner_h.T, exists       # ships RAW (not /500)


def garrison_forecast_raw(state: JaxState, H=FORECAST_H):
    """RAW projected garrison (P,H) for the precise-kill bin (targeting). Seat-independent."""
    ships_raw, _, _ = _forecast(state, H)
    return ships_raw


def _future_positions(state: JaxState, me, exists, H):
    """(P,H,2) future positions /100, C4-ROTATED to seat `me`'s canonical view, zeroed when
    absent/expired. [position machinery is 17v12-4p's; uses _rot_xy.]"""
    f32 = jnp.float32
    P = state.p_x.shape[0]
    k_f = jnp.arange(1, H + 1, dtype=f32)
    k_i = jnp.arange(1, H + 1, dtype=jnp.int32)
    a_k = state.p_orbital_a[:, None] + state.av * k_f[None, :]
    orb_x = SUN_X + state.p_orbital_r[:, None] * jnp.cos(a_k)
    orb_y = SUN_Y + state.p_orbital_r[:, None] * jnp.sin(a_k)
    L = state.p_comet_path_x.shape[1]
    cidx = jnp.clip(state.p_comet_idx[:, None] + k_i[None, :], 0, L - 1)
    com_x = jnp.take_along_axis(state.p_comet_path_x, cidx, axis=1)
    com_y = jnp.take_along_axis(state.p_comet_path_y, cidx, axis=1)
    stat_x = jnp.broadcast_to(state.p_x[:, None], (P, H))
    stat_y = jnp.broadcast_to(state.p_y[:, None], (P, H))
    fx = jnp.where(state.p_is_comet[:, None], com_x, jnp.where(state.p_is_orbiting[:, None], orb_x, stat_x))
    fy = jnp.where(state.p_is_comet[:, None], com_y, jnp.where(state.p_is_orbiting[:, None], orb_y, stat_y))
    fx, fy = _rot_xy(fx, fy, me)                                    # seat-canonical C4 rotation -> top-left
    pos = jnp.stack([fx / 100.0, fy / 100.0], axis=-1)
    return pos * exists[:, :, None]


def _planet_velocity(state: JaxState, me):
    """(P,2) per-turn velocity = pos(t+1) - pos(t): orbiting=chord of one rotation step, comet=path
    step, static=0. /6 norm. Canonical via _rot_vec (the linear part of the C4 rotation)."""
    na = state.p_orbital_a + state.av
    orb_nx = SUN_X + state.p_orbital_r * jnp.cos(na)
    orb_ny = SUN_Y + state.p_orbital_r * jnp.sin(na)
    L = state.p_comet_path_x.shape[1]
    nidx = jnp.clip(state.p_comet_idx + 1, 0, L - 1)
    com_nx = jnp.take_along_axis(state.p_comet_path_x, nidx[:, None], axis=1)[:, 0]
    com_ny = jnp.take_along_axis(state.p_comet_path_y, nidx[:, None], axis=1)[:, 0]
    nx = jnp.where(state.p_is_comet, com_nx, jnp.where(state.p_is_orbiting, orb_nx, state.p_x))
    ny = jnp.where(state.p_is_comet, com_ny, jnp.where(state.p_is_orbiting, orb_ny, state.p_y))
    # engine clamp (physics.planet_next_positions): an EXPIRING comet (idx+1 >= len) stays put this tick.
    expiring = state.p_is_comet & ((state.p_comet_idx + 1) >= state.p_comet_len)
    nx = jnp.where(expiring, state.p_x, nx)
    ny = jnp.where(expiring, state.p_y, ny)
    vx = (nx - state.p_x) / MAX_SPEED
    vy = (ny - state.p_y) / MAX_SPEED
    cvx, cvy = _rot_vec(vx, vy, me)                   # C4-rotate the velocity vector into the seat frame
    return cvx, cvy


def basic_features(state: JaxState, me: int, fc=None):
    """exp20-v37-4p features from seat `me`'s SEAT-CANONICAL (C4-rotated) view. Returns the 5-tuple
    (static (P,30), ts (P,50,10), glob (34,), mask (P,), econ_curves (50,8)). `fc` = optional
    precomputed _forecast(state) = (ships_raw, owner_h, exists), SEAT-INDEPENDENT, so the hot per-step
    path computes it ONCE and threads it into all seats + edge.

    All positions/velocities use _rot_xy/_rot_vec; ALL opponent channels use _ring_role so q1/q2/q3
    ALWAYS mean the same relative player (clockwise / antipodal / ccw) in every seat's canonical view."""
    f32 = jnp.float32
    m = state.p_mask
    P = state.p_x.shape[0]
    me_i = jnp.asarray(me, jnp.int32)
    owner = state.p_owner
    role = _ring_role(jnp.maximum(owner, 0), me_i)                   # (P,) valid where owner>=0
    is_mine = (owner == me_i) & m
    is_opp = (owner != me_i) & (owner >= 0) & m
    is_q1 = is_opp & (role == 1)
    is_q2 = is_opp & (role == 2)
    is_q3 = is_opp & (role == 3)
    is_neu = (owner == -1) & m & ~state.p_is_comet
    is_comet = state.p_is_comet & m
    is_orb = state.p_is_orbiting & m
    is_static = m & ~state.p_is_comet & ~state.p_is_orbiting
    ships_f = state.p_ships.astype(f32)
    pr_f = state.p_prod.astype(f32)
    cx, cy = _rot_xy(state.p_x, state.p_y, me)                       # canonical position
    # unified LIFE: comet = path remaining; non-comet = turns to episode end; min(.,100)/100
    remaining = (f32(EPISODE_STEPS - 1) - state.step.astype(f32))
    comet_rem = jnp.maximum(state.p_comet_len - state.p_comet_idx, 0).astype(f32)
    life = jnp.where(state.p_is_comet, comet_rem, remaining)
    life = jnp.clip(life, 0.0, 100.0) / 100.0
    inv_log_ships = 1.0 / (jnp.log(ships_f + 1.0) + 1.0)
    vx, vy = _planet_velocity(state, me)
    # static_base (17): owner 5-hot [mine,q1,q2,q3,neu] + type 3-hot [static,orb,comet] + geom (9)
    static_base = jnp.stack([
        is_mine.astype(f32), is_q1.astype(f32), is_q2.astype(f32), is_q3.astype(f32), is_neu.astype(f32),
        is_static.astype(f32), is_orb.astype(f32), is_comet.astype(f32),
        cx / 100.0, cy / 100.0, state.p_radius / 5.0,
        ships_f / 500.0, pr_f / 5.0,
        life, inv_log_ships, vx, vy,
    ], axis=-1)                                                                    # (P,17) base

    if fc is None:
        ships_raw, owner_h, exists = _forecast(state)                              # (P,50) each (RAW ships)
    else:
        ships_raw, owner_h, exists = fc                                            # hoisted (seat-shared)
    proj_ships = ships_raw / 500.0
    ex = exists > 0
    prole = _ring_role(jnp.maximum(owner_h, 0), me_i)                              # (P,50) valid where owner_h>=0
    # ts ROLE channels (5): mine / q1 / q2 / q3 / neutral, each gated by existence
    r_mine = ((owner_h == me_i) & ex).astype(f32)
    r_q1 = ((owner_h >= 0) & (owner_h != me_i) & (prole == 1) & ex).astype(f32)
    r_q2 = ((owner_h >= 0) & (owner_h != me_i) & (prole == 2) & ex).astype(f32)
    r_q3 = ((owner_h >= 0) & (owner_h != me_i) & (prole == 3) & ex).astype(f32)
    r_neu = ((owner_h == -1) & ex).astype(f32)
    pos = _future_positions(state, me, exists, FORECAST_H)                         # (P,50,2)
    # ramp = (t+1)/50 in 1/50..50/50 (1-indexed: horizon slot t IS "t+1 turns ahead")
    ramp = jnp.broadcast_to((jnp.arange(1, FORECAST_H + 1, dtype=f32) / f32(FORECAST_H))[None, :, None],
                            (P, FORECAST_H, 1))
    ts = jnp.concatenate([proj_ships[:, :, None],
                          r_mine[:, :, None], r_q1[:, :, None], r_q2[:, :, None],
                          r_q3[:, :, None], r_neu[:, :, None],
                          exists[:, :, None], pos, ramp], axis=-1) * m[:, None, None]   # (P,50,10)

    # --- horizon mask: forecast slot t is turn step+t+1; CUT at the final game turn 499 so
    # percentages / flip-turns normalize by the ACTUAL remaining horizon (<=50). ---
    slot = jnp.arange(FORECAST_H, dtype=jnp.int32)
    abs_turn = state.step + slot + 1                                              # (50,) absolute turn
    valid = abs_turn <= (EPISODE_STEPS - 1)                                       # (50,)
    n_valid = jnp.maximum(jnp.sum(valid.astype(f32)), 1.0)
    last_v = (FORECAST_H - 1) - jnp.argmax(valid[::-1])                           # last valid slot

    def _who4(owner_arr):
        """owner id -> 4-way me-relative one-hot [me, q1, q2, q3] (neutral / invalid -> all 0)."""
        rr = _ring_role(jnp.maximum(owner_arr, 0), me_i)
        is_me = (owner_arr == me_i)
        is_o = (owner_arr >= 0) & (owner_arr != me_i)
        return jnp.stack([is_me.astype(f32),
                          (is_o & (rr == 1)).astype(f32),
                          (is_o & (rr == 2)).astype(f32),
                          (is_o & (rr == 3)).astype(f32)], axis=-1)               # (...,4)

    def _who5(owner_arr):
        """owner id -> 5-way me-relative one-hot [me, q1, q2, q3, neu] (invalid/masked -> all 0)."""
        rr = _ring_role(jnp.maximum(owner_arr, 0), me_i)
        is_me = (owner_arr == me_i)
        is_o = (owner_arr >= 0) & (owner_arr != me_i)
        return jnp.stack([is_me.astype(f32),
                          (is_o & (rr == 1)).astype(f32),
                          (is_o & (rr == 2)).astype(f32),
                          (is_o & (rr == 3)).astype(f32),
                          (owner_arr == -1).astype(f32)], axis=-1)                # (...,5)

    # --- PER-PLANET forecast dynamics (-> static): when/if each planet flips, who ends up holding it,
    # and what fraction of the horizon me / its-current-owner hold it. argmax over a boolean mask
    # returns the FIRST true slot (0 when none -> guarded by any_flip / has_act). ---
    cur_owner = state.p_owner                                                     # (P,) live owner
    act = valid[None, :] & (exists > 0)                                          # (P,50) active slots
    flip = act & (owner_h != cur_owner[:, None])                                # owner differs from now
    any_flip = jnp.any(flip, axis=1)                                            # (P,)
    ft = jnp.argmax(flip, axis=1)                                               # (P,) first flip slot
    flip_turn = jnp.where(any_flip, (ft.astype(f32) + 1.0) / f32(FORECAST_H), 1.0)  # 1 = never flips
    ft_owner = jnp.take_along_axis(owner_h, ft[:, None], axis=1)[:, 0]
    flip_to = _who4(ft_owner) * any_flip[:, None].astype(f32)                    # (P,4) who holds post-flip
    secured = (~any_flip).astype(f32)                                           # owner constant over horizon
    has_act = jnp.any(act, axis=1)
    last_act = (FORECAST_H - 1) - jnp.argmax(act.astype(f32)[:, ::-1], axis=1)   # last active slot
    fo_owner = jnp.take_along_axis(owner_h, last_act[:, None], axis=1)[:, 0]
    fin_owner_id = jnp.where(has_act, fo_owner, cur_owner)
    final_owner = _who5(fin_owner_id)                                           # (P,5) owner at horizon end
    pct_me_hold = jnp.sum((act & (owner_h == me_i)).astype(f32), axis=1) / n_valid
    pct_cur_hold = jnp.sum((act & (owner_h == cur_owner[:, None])).astype(f32), axis=1) / n_valid
    planet_dyn = jnp.concatenate([
        flip_turn[:, None], flip_to, secured[:, None], final_owner,
        pct_me_hold[:, None], pct_cur_hold[:, None]], axis=-1)                    # (P, 1+4+1+5+1+1 = 13)
    static = jnp.concatenate([static_base, planet_dyn], axis=-1) * m[:, None]     # (P,17+13 = 30)

    # --- GLOBAL econ dynamics + econ_curves: per-turn per-ROLE garrison & prod curves over the
    # valid horizon (-> econ-CNN), plus me-vs-maxopp crossover scalars (-> glob). ---
    fmine = (owner_h == me_i) & ex                                               # (P,50)
    fopp = (owner_h >= 0) & (owner_h != me_i) & ex
    f_q1 = fopp & (prole == 1); f_q2 = fopp & (prole == 2); f_q3 = fopp & (prole == 3)
    # per-role ship & prod summed over planets, per future turn (4,50) each
    ship_me = jnp.sum(ships_raw * fmine.astype(f32), axis=0)                     # (50,)
    ship_q1 = jnp.sum(ships_raw * f_q1.astype(f32), axis=0)
    ship_q2 = jnp.sum(ships_raw * f_q2.astype(f32), axis=0)
    ship_q3 = jnp.sum(ships_raw * f_q3.astype(f32), axis=0)
    prod_me = jnp.sum(pr_f[:, None] * fmine.astype(f32), axis=0)
    prod_q1 = jnp.sum(pr_f[:, None] * f_q1.astype(f32), axis=0)
    prod_q2 = jnp.sum(pr_f[:, None] * f_q2.astype(f32), axis=0)
    prod_q3 = jnp.sum(pr_f[:, None] * f_q3.astype(f32), axis=0)
    # econ-CNN curve input (8 channels): per-role [ship, prod] x {me, q1, q2, q3}, normalized.
    econ_curves = jnp.stack([ship_me / 2000.0, ship_q1 / 2000.0, ship_q2 / 2000.0, ship_q3 / 2000.0,
                             prod_me / 40.0, prod_q1 / 40.0, prod_q2 / 40.0, prod_q3 / 40.0],
                            axis=-1)                                             # (50,8)

    # me-vs-maxopp LEAD curves (50,) for the econ summary scalars (kept in glob, supplement to the CNN)
    ship_maxopp = jnp.maximum(jnp.maximum(ship_q1, ship_q2), ship_q3)
    prod_maxopp = jnp.maximum(jnp.maximum(prod_q1, prod_q2), prod_q3)
    ship_lead = ship_me - ship_maxopp                                            # (50,)
    prod_lead = prod_me - prod_maxopp

    def _econ_feats(L):                               # L (50,) signed lead (me - maxopp) over horizon
        s = jnp.sign(L)                                                          # {-1,0,1}
        s0 = s[0]                                                                # slot 0 = next turn
        sl = s[last_v]
        lead_pct = jnp.sum(jnp.where(valid, (L >= 0).astype(f32), 0.0)) / n_valid  # me ahead-or-tied %
        flp = valid & (s != s0)
        anyf = jnp.any(flp)
        ftn = jnp.argmax(flp)
        next_flip = jnp.where(anyf, (ftn.astype(f32) + 1.0) / f32(FORECAST_H), 1.0)  # 1 = never
        sec = (~anyf).astype(f32)                                               # lead sign constant
        return jnp.stack([lead_pct, next_flip, sec,                             # 3
                          (s0 > 0).astype(f32),                                 # me ahead next turn
                          (sl > 0).astype(f32), (sl < 0).astype(f32)])           # me ahead / behind at last valid -> (6,)

    econ = jnp.concatenate([_econ_feats(ship_lead), _econ_feats(prod_lead)])      # (12,)

    # --- glob base (18): phase 4 + ts/tp/pf per-role 4 each + territory 2 ---
    step_f = state.step.astype(f32)
    g_turn = step_f / 500.0
    g_left = (f32(EPISODE_STEPS - 1) - step_f) / 500.0
    g_rot = state.av.astype(f32) * 10.0
    spawns = jnp.asarray(COMET_SPAWN_STEPS, dtype=f32)
    g_cd = jnp.clip(jnp.min(jnp.where(spawns > step_f, spawns - step_f, 1000.0)) / 100.0, 0.0, 1.0)

    def _sum(mask_, vals):
        return jnp.sum(jnp.where(mask_, vals, 0.0))
    # ship totals per role = owned-planet garrison + in-flight fleet ships (matches terminal totals)
    fl_owner = state.f_owner
    frole = _ring_role(jnp.maximum(fl_owner, 0), me_i)
    fl_mine = (fl_owner == me_i) & state.f_mask
    fl_q1 = (fl_owner >= 0) & (fl_owner != me_i) & (frole == 1) & state.f_mask
    fl_q2 = (fl_owner >= 0) & (fl_owner != me_i) & (frole == 2) & state.f_mask
    fl_q3 = (fl_owner >= 0) & (fl_owner != me_i) & (frole == 3) & state.f_mask
    f_ships_f = state.f_ships.astype(f32)
    ts_me = _sum(is_mine, ships_f) + _sum(fl_mine, f_ships_f)
    ts_q1 = _sum(is_q1, ships_f) + _sum(fl_q1, f_ships_f)
    ts_q2 = _sum(is_q2, ships_f) + _sum(fl_q2, f_ships_f)
    ts_q3 = _sum(is_q3, ships_f) + _sum(fl_q3, f_ships_f)
    tp_me = _sum(is_mine, pr_f); tp_q1 = _sum(is_q1, pr_f); tp_q2 = _sum(is_q2, pr_f); tp_q3 = _sum(is_q3, pr_f)
    rem = jnp.maximum(f32(EPISODE_STEPS - 1) - step_f, 0.0)                        # remaining turns
    pf_me = tp_me * rem; pf_q1 = tp_q1 * rem; pf_q2 = tp_q2 * rem; pf_q3 = tp_q3 * rem
    total_pl = jnp.sum(m.astype(f32)); comets_on = jnp.sum(is_comet.astype(f32))
    # --- afl (4): in-flight fleet COUNT per role / INFLIGHT_CAP_PER_PLAYER ---
    afl = jnp.stack([jnp.sum(fl_mine.astype(f32)), jnp.sum(fl_q1.astype(f32)),
                     jnp.sum(fl_q2.astype(f32)), jnp.sum(fl_q3.astype(f32))]) / f32(INFLIGHT_CAP_PER_PLAYER)

    glob = jnp.concatenate([
        jnp.stack([g_turn, g_left, g_rot, g_cd]),                                # phase (4)
        jnp.stack([ts_me, ts_q1, ts_q2, ts_q3]) / 2000.0,                        # ship totals per-role (4)
        jnp.stack([tp_me, tp_q1, tp_q2, tp_q3]) / 40.0,                          # prod totals per-role (4)
        jnp.stack([pf_me, pf_q1, pf_q2, pf_q3]) / 10000.0,                       # prod x remaining per-role (4)
        jnp.stack([total_pl / 20.0, comets_on / 5.0]),                           # territory (2)
        econ,                                                                    # econ scalars (12)
        afl,                                                                     # active-fleet count per-role (4)
    ])                                                                            # 4+4+4+4+2+12+4 = 34
    return static, ts, glob, m, econ_curves


def ship_totals(state: JaxState):
    """(4,) total ships per owner incl. in-flight (resign-share + terminal reward base)."""
    return jnp.stack([
        (jnp.sum(jnp.where((state.p_owner == k) & state.p_mask, state.p_ships, 0))
         + jnp.sum(jnp.where((state.f_owner == k) & state.f_mask, state.f_ships, 0))).astype(jnp.float32)
        for k in range(N_PLAYERS)
    ])


def terminal_reward_all(state: JaxState):
    """FFA per-seat terminal reward (4,): reward_i = sign(ships_i - max_{j!=i} ships_j), where
    ships_i = owned-planet garrison + in-flight fleet ships (matches forward_sim.score_for_player).
    Leader +1, else -1; tie at the top => 0 for the tied seats. [VERBATIM from 17v12-4p]"""
    tot = ship_totals(state)                                                      # (4,)
    eye = jnp.eye(N_PLAYERS, dtype=bool)
    others_max = jnp.max(jnp.where(eye, -1e18, tot[None, :]), axis=1)             # (4,) max over j!=i
    return jnp.sign(tot - others_max).astype(jnp.float32)                         # (4,)


def terminal_reward_p0(state: JaxState):
    """Convenience: seat-0 FFA reward (used by greedy-vs-snipers eval)."""
    return terminal_reward_all(state)[0]


def terminal_rewards(state: JaxState, lam: float = 0.0, D: float = 300.0):
    """exp022 per-seat MARGIN reward, FFA generalization: diff_i = ships_i - max_{j!=i} ships_j;
    leader (diff>0) gets +1 + lam*tanh(|diff|/D); everyone else FLAT -1; tie 0. lam=0 collapses to
    pure FFA win/loss (== terminal_reward_all). Returns (4,). (IL doesn't use rewards; kept for the
    train.py import + RL warm-start lineage.)"""
    tot = ship_totals(state)                                                      # (4,)
    eye = jnp.eye(N_PLAYERS, dtype=bool)
    others_max = jnp.max(jnp.where(eye, -1e18, tot[None, :]), axis=1)             # (4,)
    diff = (tot - others_max).astype(jnp.float32)                                # (4,) leader>0
    bonus = lam * jnp.tanh(jnp.abs(diff) / D)
    return jnp.where(diff > 0, 1.0 + bonus, jnp.where(diff < 0, -1.0, 0.0))


def is_done(state: JaxState):
    """4p game over (matches forward_sim.terminated): step >= EPISODE_STEPS-1 (499) OR alive_players
    <= 1 (a seat is alive iff it owns any planet OR has any in-flight fleet). [VERBATIM from 17v12-4p]"""
    alive = jnp.stack([
        (jnp.sum((state.p_owner == k) & state.p_mask) > 0) | (jnp.sum((state.f_owner == k) & state.f_mask) > 0)
        for k in range(N_PLAYERS)
    ])
    n_alive = jnp.sum(alive.astype(jnp.int32))
    return (state.step >= EPISODE_STEPS - 1) | (n_alive <= 1)
