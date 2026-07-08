"""exp19 v1 env wrapper: batched init states, FEATURES (locked design 2026-06-10/11), terminal reward.

Engine files (state/step/physics/comet/constants) are byte-copies of the parity-verified 2p 17v12
set — engine parity is inherited, only the FEATURE layer differs.

Feature contract (notes/2026-06-10/exp19v1_design.md):
  static (P,15)   ownership 3-hot + type 3-hot + cx,cy/100 + radius/5 + ships/500 + prod/5
                  + life min(.,100)/100 (comet: path-remaining; else: turns-to-end)
                  + 1/(ln(ships+1)+1) + planet velocity vx,vy /6 (canonical sign-flip)
  ts (P,50,6)     traj [ships/500, owner_sign, exists] ⊕ traj_pos [(x,y)/100 canonical] ⊕ ramp (t+1)/50
  glob (16,)      turn/500, left/500, av*10, comet_cd/100, ts_me/2000, ts_op/2000, ts_diff/2000,
                  tp_me/40, tp_op/40, tp_diff/40, share ts_diff/(ts_me+ts_op+100),
                  pc_me/20, pc_op/20, neu/20, total/20, comets_on/5
  mask (P,)
Plus (for the K=6 action space): garrison_forecast_raw(state) -> RAW projected garrison (P,50)
(seat-independent), and ship_totals(state) for the resign mechanism.
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

from constants import MAX_PLANETS, MAX_FLEETS, EPISODE_STEPS, COMET_SPAWN_STEPS, SUN_X, SUN_Y
from state import JaxState, from_kaggle_obs, fleet_speed_jax
import comet as cometmod

MAX_SPEED = 6.0
FORECAST_H = 50
N_STATIC = 23      # v26: 15 base + 8 planet_dyn (flip_player/final_owner each 2-ch) ; was 21 (signed)
N_GLOBAL = 28      # v32 (supplement): 12 base + 14 econ scalars (KEPT) + 2 afl = 28 (as v30) + econ-CNN curve
TS_C = 7           # v26: traj 4 (ships + owner 2-ch + exists) + traj_pos 2 + ramp 1 ; was 6
# (gecon20 economy-momentum feature REMOVED 2026-06-11 by user decision — it was a 15v7-only
#  pattern; exp19 board readout is [gtok ‖ masked-mean emb] = 256, nothing concatenated.)


def gen_init_states(n_envs: int, seed: int = 0) -> JaxState:
    """n_envs initial (step-0, comet-free) JaxStates from the kaggle env (host-side)."""
    from kaggle_environments import make
    states = []
    for i in range(n_envs):
        random.seed(seed + i)                       # determinism: seed BEFORE make()
        env = make("orbit_wars", configuration={"episodeSteps": 500})
        env.reset(2)
        env.step([[], []])
        obs = env.steps[1][0]["observation"]
        st = from_kaggle_obs(obs)
        sched = cometmod.gen_schedule(obs["initial_planets"], obs["angular_velocity"], seed + i)
        states.append(cometmod.attach_schedule(st, sched))
    return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs, axis=0), *states)


def _forecast(state: JaxState, H=FORECAST_H):
    """SEAT-INDEPENDENT passive projection (production + in-flight arrivals + captures, mirror of
    step.py combat). Returns (ships_raw (P,H) UNNORMALIZED garrison, owner (P,H) int ids, exists (P,H))."""
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
    inc0, inc1 = _inflow(0), _inflow(1)
    p_prod = state.p_prod.astype(f32)

    def one_turn(carry, inc):
        ships, owner = carry
        s0, s1 = inc
        ships = ships + jnp.where((owner >= 0) & state.p_mask, p_prod, 0.0)
        both = (s0 > 0) & (s1 > 0)
        tie = both & (s0 == s1)
        no_atk = (s0 == 0) & (s1 == 0)
        surv_owner = jnp.where(s0 > 0, jnp.where(s1 > 0, jnp.where(s0 >= s1, 0, 1), 0),
                               jnp.where(s1 > 0, 1, -1)).astype(jnp.int32)
        surv = jnp.where(both, jnp.abs(s0 - s1), jnp.where(s0 > 0, s0, s1))
        surv = jnp.where(tie, 0.0, surv)
        has_s = (~no_atk) & (~tie)
        same = has_s & (surv_owner == owner)
        capture = has_s & (surv_owner != owner) & (surv > ships)
        repel = has_s & (surv_owner != owner) & (surv <= ships)
        new_ships = jnp.where(same, ships + surv,
                    jnp.where(capture, surv - ships,
                    jnp.where(repel, ships - surv, ships)))
        new_owner = jnp.where(capture, surv_owner, owner)
        return (new_ships, new_owner), (new_ships, new_owner)

    _, (ships_h, owner_h) = jax.lax.scan(
        one_turn, (state.p_ships.astype(f32), state.p_owner), (inc0, inc1))
    hrange = jnp.arange(H, dtype=jnp.int32)
    alive = (~state.p_is_comet[:, None]) | ((state.p_comet_idx[:, None] + hrange[None, :] + 1) < state.p_comet_len[:, None])
    exists = (state.p_mask[:, None] & alive).astype(f32)
    return ships_h.T * exists, owner_h.T, exists       # ships RAW (not /500)


def garrison_forecast_raw(state: JaxState, H=FORECAST_H):
    """RAW projected garrison (P,H) for the precise-kill bin (targeting). Seat-independent."""
    ships_raw, _, _ = _forecast(state, H)
    return ships_raw


def _future_positions(state: JaxState, me, exists, H):
    """(P,H,2) future positions /100, canonical reflect when me==1, zeroed when absent/expired."""
    f32 = jnp.float32
    me = jnp.asarray(me, jnp.int32)
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
    fx = jnp.where(me == 1, 100.0 - fx, fx)
    fy = jnp.where(me == 1, 100.0 - fy, fy)
    pos = jnp.stack([fx / 100.0, fy / 100.0], axis=-1)
    return pos * exists[:, :, None]


def _planet_velocity(state: JaxState, me):
    """(P,2) per-turn velocity = pos(t+1) − pos(t): orbiting=chord of one rotation step, comet=path
    step, static=0. /6 norm. Canonical reflection flips the sign (vel computed on reflected coords)."""
    na = state.p_orbital_a + state.av
    orb_nx = SUN_X + state.p_orbital_r * jnp.cos(na)
    orb_ny = SUN_Y + state.p_orbital_r * jnp.sin(na)
    L = state.p_comet_path_x.shape[1]
    nidx = jnp.clip(state.p_comet_idx + 1, 0, L - 1)
    com_nx = jnp.take_along_axis(state.p_comet_path_x, nidx[:, None], axis=1)[:, 0]
    com_ny = jnp.take_along_axis(state.p_comet_path_y, nidx[:, None], axis=1)[:, 0]
    nx = jnp.where(state.p_is_comet, com_nx, jnp.where(state.p_is_orbiting, orb_nx, state.p_x))
    ny = jnp.where(state.p_is_comet, com_ny, jnp.where(state.p_is_orbiting, orb_ny, state.p_y))
    # engine clamp (physics.planet_next_positions): an EXPIRING comet (idx+1 >= len) stays put this
    # tick. Without this the zero-padded path reads (0,0) and vx/vy spike to ~-x/6 ≈ -16 on every
    # comet's final alive turn (audit fix 2026-06-11).
    expiring = state.p_is_comet & ((state.p_comet_idx + 1) >= state.p_comet_len)
    nx = jnp.where(expiring, state.p_x, nx)
    ny = jnp.where(expiring, state.p_y, ny)
    vx = (nx - state.p_x) / MAX_SPEED
    vy = (ny - state.p_y) / MAX_SPEED
    me = jnp.asarray(me, jnp.int32)
    sgn = jnp.where(me == 1, -1.0, 1.0)               # reflected frame flips velocity
    return vx * sgn, vy * sgn


def basic_features(state: JaxState, me: int, fc=None):
    """exp21 features from seat `me`'s canonical view. Returns (static (P,21), ts (P,50,6),
    glob (30,), mask (P,)). `fc` = optional precomputed _forecast(state) = (ships_raw, owner_h, exists);
    SEAT-INDEPENDENT, so the hot per-step path computes it ONCE and threads it into both seats + edge
    (was recomputed per call)."""
    f32 = jnp.float32
    m = state.p_mask
    P = state.p_x.shape[0]
    is_mine = (state.p_owner == me) & m
    is_opp = (state.p_owner != me) & (state.p_owner >= 0) & m
    is_neu = (state.p_owner == -1) & m & ~state.p_is_comet
    is_comet = state.p_is_comet & m
    is_orb = state.p_is_orbiting & m
    is_static = m & ~state.p_is_comet & ~state.p_is_orbiting
    ships_f = state.p_ships.astype(f32)
    pr_f = state.p_prod.astype(f32)
    cx = jnp.where(me == 1, 100.0 - state.p_x, state.p_x)
    cy = jnp.where(me == 1, 100.0 - state.p_y, state.p_y)
    # unified LIFE: comet = path remaining; non-comet = turns to episode end; min(.,100)/100
    remaining = (f32(EPISODE_STEPS - 1) - state.step.astype(f32))
    comet_rem = jnp.maximum(state.p_comet_len - state.p_comet_idx, 0).astype(f32)
    life = jnp.where(state.p_is_comet, comet_rem, remaining)
    life = jnp.clip(life, 0.0, 100.0) / 100.0
    inv_log_ships = 1.0 / (jnp.log(ships_f + 1.0) + 1.0)
    vx, vy = _planet_velocity(state, me)
    static_base = jnp.stack([
        is_mine.astype(f32), is_opp.astype(f32), is_neu.astype(f32),
        is_static.astype(f32), is_orb.astype(f32), is_comet.astype(f32),
        cx / 100.0, cy / 100.0, state.p_radius / 5.0,
        ships_f / 500.0, pr_f / 5.0,
        life, inv_log_ships, vx, vy,
    ], axis=-1)                                                                    # (P,15) base

    if fc is None:
        ships_raw, owner_h, exists = _forecast(state)                              # (P,50) each
    else:
        ships_raw, owner_h, exists = fc                                            # hoisted (seat-shared)
    proj_ships = ships_raw / 500.0
    me_i = jnp.asarray(me, jnp.int32)
    opp_i = 1 - me_i
    own_me_ts = (owner_h == me_i).astype(f32) * exists                             # v26: owner 2-ch (was signed own_sign)
    own_op_ts = ((owner_h != me_i) & (owner_h >= 0)).astype(f32) * exists
    pos = _future_positions(state, me, exists, FORECAST_H)                         # (P,50,2)
    # ramp = (t+1)/50 in 1/50..50/50 (1-indexed 2026-06-11: horizon slot t IS "t+1 turns ahead")
    ramp = jnp.broadcast_to((jnp.arange(1, FORECAST_H + 1, dtype=f32) / f32(FORECAST_H))[None, :, None],
                            (P, FORECAST_H, 1))
    ts = jnp.concatenate([proj_ships[:, :, None], own_me_ts[:, :, None], own_op_ts[:, :, None], exists[:, :, None],
                          pos, ramp], axis=-1) * m[:, None, None]                  # (P,50,7) v26: owner 2-ch

    # --- exp21 horizon mask (user 2026-06-13): forecast slot t is turn step+t+1; CUT at the final
    # game turn 499 so percentages / flip-turns normalize by the ACTUAL remaining horizon (<=50). ---
    slot = jnp.arange(FORECAST_H, dtype=jnp.int32)
    abs_turn = state.step + slot + 1                                              # (50,) absolute turn
    valid = abs_turn <= (EPISODE_STEPS - 1)                                       # (50,)
    n_valid = jnp.maximum(jnp.sum(valid.astype(f32)), 1.0)
    last_v = (FORECAST_H - 1) - jnp.argmax(valid[::-1])                           # last valid slot

    def _sgn(owner):                                  # owner id -> me-relative signed {+1 me,-1 opp,0 neu}
        return jnp.where(owner == me_i, f32(1.0), jnp.where(owner == opp_i, f32(-1.0), f32(0.0)))

    # --- exp21 PER-PLANET forecast dynamics (-> static): when / if each planet flips, who ends up
    # holding it, and what fraction of the horizon me / its-current-owner hold it. argmax over a
    # boolean mask returns the FIRST true slot (0 when none -> guarded by any_flip / has_act). ---
    cur_owner = state.p_owner                                                     # (P,) live owner
    act = valid[None, :] & (exists > 0)                                          # (P,50) active slots
    flip = act & (owner_h != cur_owner[:, None])                                # owner differs from now
    any_flip = jnp.any(flip, axis=1)                                            # (P,)
    ft = jnp.argmax(flip, axis=1)                                               # (P,) first flip slot
    flip_turn = jnp.where(any_flip, (ft.astype(f32) + 1.0) / f32(FORECAST_H), 1.0)  # 1 = never flips
    ft_owner = jnp.take_along_axis(owner_h, ft[:, None], axis=1)[:, 0]
    flip_player_me = jnp.where(any_flip & (ft_owner == me_i), f32(1.0), f32(0.0))   # v26: owner 2-ch (was signed flip_player)
    flip_player_op = jnp.where(any_flip & (ft_owner == opp_i), f32(1.0), f32(0.0))
    secured = (~any_flip).astype(f32)                                           # owner constant over horizon
    has_act = jnp.any(act, axis=1)
    last_act = (FORECAST_H - 1) - jnp.argmax(act.astype(f32)[:, ::-1], axis=1)   # last active slot
    fo_owner = jnp.take_along_axis(owner_h, last_act[:, None], axis=1)[:, 0]
    fin_owner = jnp.where(has_act, fo_owner, cur_owner)                          # owner id at horizon end
    final_owner_me = (fin_owner == me_i).astype(f32)                            # v26: owner 2-ch (was signed final_owner)
    final_owner_op = (fin_owner == opp_i).astype(f32)
    pct_me_hold = jnp.sum((act & (owner_h == me_i)).astype(f32), axis=1) / n_valid
    pct_cur_hold = jnp.sum((act & (owner_h == cur_owner[:, None])).astype(f32), axis=1) / n_valid
    planet_dyn = jnp.stack([flip_turn, flip_player_me, flip_player_op, secured, final_owner_me, final_owner_op,
                            pct_me_hold, pct_cur_hold], axis=-1)                  # (P,8) v26: owner 2-ch
    static = jnp.concatenate([static_base, planet_dyn], axis=-1) * m[:, None]     # (P,21)

    # --- exp21 GLOBAL econ dynamics (-> glob): per-turn me-vs-opp LEAD curves (ship & prod) over the
    # valid horizon + crossover / dominance scalars. Cross-planet x cross-time aggregates
    # (Sum_p owner[p,t]*{ships|prod}) the net cannot sum itself -> compute them. ---
    fmine = (owner_h == me_i) & (exists > 0)                                      # (P,50)
    fopp = (owner_h == opp_i) & (exists > 0)
    ship_lead = jnp.sum(ships_raw * (fmine.astype(f32) - fopp.astype(f32)), axis=0)      # (50,)
    prod_lead = jnp.sum(pr_f[:, None] * (fmine.astype(f32) - fopp.astype(f32)), axis=0)  # (50,)

    # v32 (SUPPLEMENT, user 2026-06-17 "v32先加上"): KEEP the 14 econ summary scalars in glob (as v30)
    # AND ALSO feed the full me-vs-opp lead curves to an econ-CNN in the model (econ_emb -> gtok).
    def _econ_feats(L):                               # L (50,) signed lead (me - opp) over horizon
        s = jnp.sign(L)                                                          # {-1,0,1}
        s0 = s[0]                                                                # slot 0 = next turn
        lead_pct = jnp.sum(jnp.where(valid, (L >= 0).astype(f32), 0.0)) / n_valid  # me ahead-or-tied %
        flp = valid & (s != s0)
        anyf = jnp.any(flp)
        ftn = jnp.argmax(flp)
        next_flip = jnp.where(anyf, (ftn.astype(f32) + 1.0) / f32(FORECAST_H), 1.0)  # 1 = never
        sec = (~anyf).astype(f32)                                               # lead sign constant
        sl = s[last_v]
        return jnp.stack([lead_pct, next_flip, sec,                             # sign 2-ch (me_ahead / op_ahead) x {s0, s_last}
                          (s0 > 0).astype(f32), (s0 < 0).astype(f32),
                          (sl > 0).astype(f32), (sl < 0).astype(f32)])           # (7,)

    econ = jnp.concatenate([_econ_feats(ship_lead), _econ_feats(prod_lead)])      # (14,) v26 scalars (KEPT)
    # econ-CNN curve input (ADDITIONAL to the 14 scalars): full lead curves -> conv-CNN -> econ_emb -> gtok
    econ_curves = jnp.stack([ship_lead / 2000.0, prod_lead / 40.0], axis=-1)      # (50,2)

    # glob (20 base + 10 econ = 30)
    step_f = state.step.astype(f32)
    g_turn = step_f / 500.0
    g_left = (f32(EPISODE_STEPS - 1) - step_f) / 500.0
    g_rot = state.av.astype(f32) * 10.0
    spawns = jnp.asarray(COMET_SPAWN_STEPS, dtype=f32)
    g_cd = jnp.clip(jnp.min(jnp.where(spawns > step_f, spawns - step_f, 1000.0)) / 100.0, 0.0, 1.0)
    fl_mine = (state.f_owner == me_i) & state.f_mask
    fl_opp = (state.f_owner != me_i) & (state.f_owner >= 0) & state.f_mask
    ts_me = jnp.sum(jnp.where(is_mine, ships_f, 0.0)) + jnp.sum(jnp.where(fl_mine, state.f_ships.astype(f32), 0.0))
    ts_op = jnp.sum(jnp.where(is_opp, ships_f, 0.0)) + jnp.sum(jnp.where(fl_opp, state.f_ships.astype(f32), 0.0))
    tp_me = jnp.sum(jnp.where(is_mine, pr_f, 0.0)); tp_op = jnp.sum(jnp.where(is_opp, pr_f, 0.0))
    share = (ts_me - ts_op) / (ts_me + ts_op + 100.0)
    pc_me = jnp.sum(is_mine.astype(f32)); pc_op = jnp.sum(is_opp.astype(f32))
    neu_rem = jnp.sum(is_neu.astype(f32)); total_pl = jnp.sum(m.astype(f32)); comets_on = jnp.sum(is_comet.astype(f32))
    rem = jnp.maximum(f32(EPISODE_STEPS - 1) - step_f, 0.0)                        # remaining turns
    pf_me = tp_me * rem; pf_op = tp_op * rem
    # exp22 v4 feature: in-flight fleet COUNTS (not ships) per player, /128 (= FLEET_CAP_PER_PLAYER)
    # so the model knows how close it / the opp is to the launch cap. fl_mine/fl_opp computed above.
    afl_me = jnp.sum(fl_mine.astype(f32)); afl_op = jnp.sum(fl_opp.astype(f32))
    glob = jnp.concatenate([jnp.stack([
        g_turn, g_left, g_rot, g_cd,                                              # phase (4)
        ts_me / 2000.0, ts_op / 2000.0,                                           # v30: ship totals (2) [diff REMOVED]
        tp_me / 40.0, tp_op / 40.0,                                               # v30: production totals (2) [diff REMOVED]
        pf_me / 10000.0, pf_op / 10000.0,                                         # v30: prod x remaining (2) [diff REMOVED]
        # v30: share, planet counts (pc me/op/diff), neu_rem ALL REMOVED
        total_pl / 20.0, comets_on / 5.0,                                          # v30: territory (2) [neu_rem REMOVED]
    ]), econ,                                                                      # v32: 12 base + 14 econ scalars (KEPT, supplement)
        jnp.stack([afl_me / 256.0, afl_op / 256.0]),                              # v25 ABLATION: afl /256 (=exp23-v10/v11 FLEET_CAP), was /128 (v13) -> isolates fleet-norm effect on IL
    ])                                                                            # glob = 12 base + 14 econ + 2 afl = 28
    return static, ts, glob, m, econ_curves


def ship_totals(state: JaxState):
    """(t0, t1) total ships incl. in-flight — resign-share + terminal reward base."""
    t0 = jnp.sum(jnp.where((state.p_owner == 0) & state.p_mask, state.p_ships, 0)) \
        + jnp.sum(jnp.where((state.f_owner == 0) & state.f_mask, state.f_ships, 0))
    t1 = jnp.sum(jnp.where((state.p_owner == 1) & state.p_mask, state.p_ships, 0)) \
        + jnp.sum(jnp.where((state.f_owner == 1) & state.f_mask, state.f_ships, 0))
    return t0.astype(jnp.float32), t1.astype(jnp.float32)


def terminal_reward_p0(state: JaxState):
    t0, t1 = ship_totals(state)
    return jnp.sign(t0 - t1)


def terminal_rewards(state: JaxState, lam: float = 0.4, D: float = 300.0):
    """exp022 per-seat MARGIN reward (user 2026-06-14): winner = +1 + lam*tanh(|ship_diff|/D),
    loser = -1 (FLAT), tie = 0. NOT zero-sum (loser is flat -1, not -(winner)) so the rollout records
    BOTH seats' rewards instead of flipping r0. ship_diff = total ships incl in-flight. lam small +
    tanh saturating keeps the win/loss spine primary; the margin term restores a gradient in the
    already-won (flat-reward) region -> the agent consolidates instead of wild-launching."""
    t0, t1 = ship_totals(state)
    diff = (t0 - t1).astype(jnp.float32)
    bonus = lam * jnp.tanh(jnp.abs(diff) / D)
    r0 = jnp.where(diff > 0, 1.0 + bonus, jnp.where(diff < 0, -1.0, 0.0))
    r1 = jnp.where(diff < 0, 1.0 + bonus, jnp.where(diff > 0, -1.0, 0.0))
    return r0, r1


def is_done(state: JaxState):
    """Official game over: step>=499 OR <=1 player alive. (Resign-early-end lives in train's rollout.)"""
    p0_alive = (jnp.sum((state.p_owner == 0) & state.p_mask) > 0) | (jnp.sum((state.f_owner == 0) & state.f_mask) > 0)
    p1_alive = (jnp.sum((state.p_owner == 1) & state.p_mask) > 0) | (jnp.sum((state.f_owner == 1) & state.f_mask) > 0)
    n_alive = p0_alive.astype(jnp.int32) + p1_alive.astype(jnp.int32)
    return (state.step >= EPISODE_STEPS - 1) | (n_alive <= 1)
