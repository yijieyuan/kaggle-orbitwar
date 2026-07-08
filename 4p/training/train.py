"""exp24 ilselfplay train.py (forked from exp020 v36) — econ-CNN + 2-tree {net,frac}, 6-d (all-in)
frac edge_tid (matches train_il_v5). ⚠ the self-play rollout here (per_env_actions / loss_fn) is the
LEGACY pointer-ONLY ALL-IN path and does NOT train the frac head; the REAL 2-tree {net,frac} self-play
is train_league.py (league 80/20, no resign). This module is imported by train_league.py / train_il_v5.py
as `tb` for its battle-tested fns: cg_sample/cg_logp/cg_entropy, edge_features, first_hit_gate, gae,
_value, make_eval, greedy_action. ALL of those are econ-CORRECT here (5-tuple basic_features +
7-arg net.apply with econ) and use 6-d frac edge_tid so a train_il_v5 warm-start msgpack restores
WITHOUT a shape mismatch (the IL frac edge_enc Dense is 6->32). greedy_action(net,frac,...) + make_eval
DO use frac, so evaluating a train.py-only ckpt would measure an UNTRAINED frac (invalid).
Original exp21 design doc follows.

exp21 edge-margin — pointer-ONLY + ALL-IN action, with EDGE FEATURES as trunk attention-bias.

The design that this whole session converged on, in its simplest action form (user 2026-06-13):

  ACTION: each planet either HOLDS (pointer self-target) or sends its ENTIRE garrison (ALL-IN) to a
  target. NO launch gate, NO fraction head. So the action is JUST the pointer over (targets ∪ self).
  This collapses the per-(s,j,k) decision to per-(s,j): one fleet size (full garrison) -> one arrival
  -> one margin. Coordination is much easier (discrete target assignment), and "won't all-in" is no
  longer learnable noise.

  EDGE FEATURES (the load-bearing part): per ordered pair (s->j) for the ALL-IN fleet, all gather-level
  from existing solves —
     dist/diag, reach(bool), arrival/50, eff@arrival/500, margin/500, (margin>0)
  where eff@arrival = garrison_forecast[j, TURNS_allin[s,j]] and margin = garrison_s - eff - 1. These
  are exactly the cross-entity / cross-time aggregates the net CANNOT derive from per-planet token
  summaries. Injected TWICE (model.py): (a) GRAPHORMER-STYLE attention bias INSIDE the trunk (per-head,
  pre-softmax, all 3 layers) so emb AND the global token absorb them -> board -> the VALUE head sees
  them; (b) a DIRECT pointer-logit term (small EdgeMLP added to tgt) so the SELECTION uses margin
  without having to reconstruct it from emb.

  Edge is SEAT-INDEPENDENT (geometry + raw garrison forecast), so it is computed ONCE per env-step and
  shared by both seats (recorded as a separate stream to keep the batch memory bounded).

  Base = exp19 feature pipeline (static21 / ts6 / glob30 — adds per-planet forecast dynamics to static
  and global econ ship/prod curves to glob) + OrbitNet19 trunk (edge-biased). Pure self-play + resign
  (no league in v1; add later like v6). Params = ONE tree {net}.
"""
import os, sys, time, argparse

os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".jax_cache"))

import csv, json
import numpy as np
import jax
import jax.numpy as jnp
import optax
import flax.serialization as fser

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "jax_env"))

from state import JaxState, fleet_speed_jax  # noqa: E402
from step import step as env_step, predict_first_hits  # noqa: E402
from constants import LAUNCH_CLEARANCE        # noqa: E402
from env import (gen_init_states, basic_features, garrison_forecast_raw, ship_totals,
                 terminal_reward_p0, terminal_rewards, is_done, EPISODE_STEPS, _forecast)  # noqa: E402
from targeting import (lead_for_ships, reach_probe_static, reach_solve_static,
                       _gather_garrison)   # noqa: E402
from model import OrbitNet19, CoordFracGauss   # noqa: E402  (net takes `edge`; frac = clipped-Gaussian head)

GATE_H = 50
_DIAG = 100.0 * 1.4142135623730951
E_EDGE = 6
_LOG_SQRT_2PI = 0.5 * float(jnp.log(2.0 * jnp.pi))


# ---- clipped-Gaussian fraction f = clip(N(mu,sigma), 0, 1): f=0 hold, f=1 all-in (exp022) ----
def cg_sample(key, mu, sigma):
    eps = jax.random.normal(key, mu.shape)
    return jnp.clip(mu + sigma * eps, 0.0, 1.0)


def cg_logp(f, mu, sigma):
    """log-density of f under clip(N(mu,sigma),0,1): Gaussian interior + CDF atoms at the boundaries.
    log P(f=0)=log Phi(-mu/sigma); log P(f=1)=log Phi((mu-1)/sigma); interior=Gaussian logpdf."""
    z = (f - mu) / sigma
    log_interior = -0.5 * z * z - jnp.log(sigma) - _LOG_SQRT_2PI
    log_atom0 = jax.scipy.special.log_ndtr(-mu / sigma)             # P(raw <= 0)
    log_atom1 = jax.scipy.special.log_ndtr((mu - 1.0) / sigma)      # P(raw >= 1)
    return jnp.where(f <= 0.0, log_atom0, jnp.where(f >= 1.0, log_atom1, log_interior))


def cg_entropy(mu, sigma):
    """Gaussian differential-entropy proxy (encourages sigma>0 exploration; ignores the clip atoms)."""
    return jnp.log(sigma) + 0.5 + _LOG_SQRT_2PI


def first_hit_gate(st, tid, angle, ships, H=GATE_H):
    lc = st.p_radius + LAUNCH_CLEARANCE
    sx = st.p_x + lc * jnp.cos(angle)
    sy = st.p_y + lc * jnp.sin(angle)
    fh, _ = predict_first_hits(st, sx, sy, angle, ships, H=H)
    return fh == tid


def edge_features(st: JaxState, fc=None, lead=None):
    """Per ordered pair (s->j) features for the ALL-IN fleet (ships = source garrison).
    SEAT-INDEPENDENT. Returns (R, ANG, TURNS, edge(P,P,E_EDGE)). `fc` = precomputed _forecast output
    (only fc[0]=ships_raw is needed here); `lead` = precomputed (R,ANG,TURNS) full-garrison solve. Both
    let the hot per-step path share the ONE forecast + ONE lead solve instead of recomputing them."""
    f32 = jnp.float32
    ships = st.p_ships                                   # all-in = full garrison
    if lead is None:
        R, ANG, TURNS = lead_for_ships(st, ships)        # (P,P) each (validated full-garrison solve)
    else:
        R, ANG, TURNS = lead                             # shared with reach_solve_static (same count)
    gar_raw = fc[0] if fc is not None else garrison_forecast_raw(st)   # (P,H) raw projected garrison
    eff = _gather_garrison(gar_raw, TURNS)               # (P,P) defense at arrival
    shipsf = ships.astype(f32)
    margin = shipsf[:, None] - eff - 1.0                 # (P,P) can the all-in capture
    dx = st.p_x[:, None] - st.p_x[None, :]
    dy = st.p_y[:, None] - st.p_y[None, :]
    dist = jnp.sqrt(dx * dx + dy * dy) / _DIAG
    Rf = R.astype(f32)
    geo = jnp.stack([jnp.clip(TURNS.astype(f32), 0.0, 50.0) / 50.0,
                     eff / 500.0, margin / 500.0, (margin > 0).astype(f32)], axis=-1) * Rf[:, :, None]
    edge = jnp.concatenate([dist[:, :, None], Rf[:, :, None], geo], axis=-1)   # (P,P,6)
    return R, ANG, TURNS, edge


def edge_partial(st: JaxState, frac, fc=None):
    """exp022 v2: the 5 f-DEPENDENT edge dims at a SECOND operating point ships=round(frac*garrison)>=1:
    [reach, arrival/50, eff/500, margin/500, can_capture], each (P,P), geo gated by reach (matches
    edge_features' [Rf ‖ geo] layout, MINUS the f-independent `dist`). Gives the fraction head the
    speed-dependent demand SLOPE (a partial fleet is slower -> later arrival -> more accrued defense),
    which the single all-in edge cannot express. Costs ONE extra lead_for_ships solve per call."""
    f32 = jnp.float32
    ships = jnp.maximum(1, jnp.round(frac * st.p_ships.astype(f32)).astype(jnp.int32))   # >=1 (0-ship fleet is degenerate)
    R, ANG, TURNS = lead_for_ships(st, ships)
    gar_raw = fc[0] if fc is not None else garrison_forecast_raw(st)
    eff = _gather_garrison(gar_raw, TURNS)
    margin = ships.astype(f32)[:, None] - eff - 1.0
    Rf = R.astype(f32)
    geo = jnp.stack([jnp.clip(TURNS.astype(f32), 0.0, 50.0) / 50.0,
                     eff / 500.0, margin / 500.0, (margin > 0).astype(f32)], axis=-1) * Rf[:, :, None]
    return jnp.concatenate([Rf[:, :, None], geo], axis=-1)   # (P,P,5)


HALF_FRAC = 0.5   # exp022 v2 second operating point (besides all-in=1.0)


def per_env_actions(net, params, st: JaxState, rng):
    """pointer-only ALL-IN, both seats sampled (LEGACY train.py self-play path; train_league uses the
    2-tree per_env_actions_vs with the fraction head). 2-tree-SAFE: reads params['net'] for the pointer/
    value (frac subtree unused here). rec = (static,ts,glob,reach,mask,econ,tid,acting,logp,v); edge (P,P,6)
    returned SEPARATELY (seat-shared)."""
    P = st.p_owner.shape[0]
    ar = jnp.arange(P)
    k0, k1 = jax.random.split(rng, 2)
    fc = _forecast(st)                                   # seat-independent forecast, computed ONCE
    R, ANG, TURNS, Rg = reach_solve_static(st)           # ONE full-garrison solve -> edge lead + pointer mask
    _, _, _, edge = edge_features(st, fc=fc, lead=(R, ANG, TURNS))
    ships_all = st.p_ships

    def side(me, kk):
        static, ts, glob, m, econ = basic_features(st, me, fc=fc)   # v36 econ-CNN: 5-tuple (+ econ_curves)
        is_mine = (st.p_owner == me) & st.p_mask
        reach_me = Rg & is_mine[:, None]
        tgt, _emb, _ctx, _board, v = net.apply(params['net'], static, ts, glob, reach_me, m, edge, econ)   # + econ
        acting = is_mine & (st.p_ships > 0)
        tid = jax.random.categorical(kk, tgt)            # self = hold (always legal)
        lpt = jax.nn.log_softmax(tgt, -1)[ar, tid]
        logp = jnp.sum(jnp.where(acting, lpt, 0.0))
        return (static, ts, glob, reach_me, m, econ, tid, acting, logp, v)

    rec0 = side(0, k0)
    rec1 = side(1, k1)
    tid0, tid1 = rec0[5], rec1[5]
    act0, act1 = rec0[6], rec1[6]
    owner = st.p_owner
    tid = jnp.where(owner == 0, tid0, jnp.where(owner == 1, tid1, ar))
    acting = act0 | act1
    ships = ships_all                                    # ALL-IN
    angle = ANG[ar, tid]; turns = TURNS[ar, tid]
    cand = acting & (tid != ar) & R[ar, tid] & (ships > 0)
    launch = cand & first_hit_gate(st, tid, angle, ships)
    ships = jnp.where(launch, ships, 0)
    return rec0, rec1, edge, launch, angle, ships, tid, st.step + turns


def sniper_action(state, me):
    P = state.p_owner.shape[0]; ar = jnp.arange(P)
    px, py = state.p_x, state.p_y
    d2 = (px[:, None] - px[None, :]) ** 2 + (py[:, None] - py[None, :]) ** 2
    is_mine = (state.p_owner == me) & state.p_mask & (state.p_ships > 0)
    is_tgt = state.p_mask & (state.p_owner != me) & ~state.p_is_comet
    md2 = jnp.where(is_tgt[None, :], d2, jnp.inf)
    nearest = jnp.argmin(md2, axis=1)
    has = jnp.isfinite(jnp.min(md2, axis=1))
    needed = jnp.maximum(state.p_ships[nearest] + 1, 20)
    can = is_mine & (state.p_ships >= needed) & has
    ships = jnp.where(can, needed, 0).astype(jnp.int32)
    ang = jnp.arctan2(py[nearest] - py, px[nearest] - px)
    dist = jnp.where(has, jnp.sqrt(jnp.min(md2, axis=1)), 0.0)
    turns = jnp.maximum(1, jnp.ceil(dist / fleet_speed_jax(ships)).astype(jnp.int32))
    target = jnp.where(can, nearest, ar)
    return can, ang, ships, target, state.step + turns


def greedy_action(net, frac, params, state, me):
    """Deterministic: argmax pointer (self=hold) -> clipped-Gaussian MEAN fraction (exp022 2-tree
    {net,frac}). Hoisted (1 forecast + 1 all-in solve + 1 EXECUTED-count solve for the chosen fleet)."""
    P = state.p_owner.shape[0]; ar = jnp.arange(P); f32 = jnp.float32
    fc = _forecast(state)
    R, ANG, TURNS, Rg = reach_solve_static(state)
    static, ts, glob, m, econ = basic_features(state, me, fc=fc)   # v36 econ-CNN: 5-tuple (+ econ_curves)
    is_mine = (state.p_owner == me) & state.p_mask
    reach = Rg & is_mine[:, None]
    _, _, _, edge = edge_features(state, fc=fc, lead=(R, ANG, TURNS))
    tgt, emb, ctx, _b, _v = net.apply(params['net'], static, ts, glob, reach, m, edge, econ)   # + econ
    tid = jnp.argmax(tgt, -1)
    is_real = is_mine & (tid != ar) & R[ar, tid]
    emb_tid = emb[tid]                                   # (P,E) chosen-target embedding
    edge_tid = edge[ar, tid]                             # v36: ALL-IN 6 only (matches train_il_v5 frac head)
    mu, sigma = frac.apply(params['frac'], ctx, emb_tid, edge_tid, tid, is_real)
    f = jnp.clip(mu, 0.0, 1.0)                           # greedy fraction = clipped mean
    garrison = state.p_ships
    ships = jnp.clip(jnp.round(f * garrison.astype(f32)).astype(jnp.int32), 0, garrison)
    Rx, ANGx, TURNSx = lead_for_ships(state, ships)      # EXECUTED-count solve (partial != all-in speed)
    angle = ANGx[ar, tid]; turns = TURNSx[ar, tid]
    cand = is_real & (ships > 0) & Rx[ar, tid]
    launch = cand & first_hit_gate(state, tid, angle, ships)
    ships = jnp.where(launch, ships, 0)
    return launch, angle, ships, tid, state.step + turns


def make_eval(net, frac, T):
    def ev(params, states):
        def one(carry, _):
            st, = carry
            l0, a0, s0, t0, ar0 = jax.vmap(lambda s: greedy_action(net, frac, params, s, 0))(st)
            l1, a1, s1, t1, ar1 = jax.vmap(lambda s: sniper_action(s, 1))(st)
            o = st.p_owner
            launch = jnp.where(o == 0, l0, jnp.where(o == 1, l1, False))
            angle = jnp.where(o == 0, a0, a1); ships = jnp.where(o == 0, s0, s1)
            target = jnp.where(o == 0, t0, t1); arrival = jnp.where(o == 0, ar0, ar1)
            return (jax.vmap(env_step)(st, launch, angle, ships, target, arrival),), None
        (stf,), _ = jax.lax.scan(one, (states,), None, length=T)
        Rr = jax.vmap(terminal_reward_p0)(stf)
        return (Rr > 0).mean(), (Rr < 0).mean()
    return ev


def _value(net, params, st, me):
    fc = _forecast(st)
    R, ANG, TURNS, Rg = reach_solve_static(st)
    static, ts, glob, m, econ = basic_features(st, me, fc=fc)   # v36 econ-CNN: 5-tuple (+ econ_curves)
    is_mine = (st.p_owner == me) & st.p_mask
    reach = Rg & is_mine[:, None]
    _, _, _, edge = edge_features(st, fc=fc, lead=(R, ANG, TURNS))
    _t, _e, _c, _b, v = net.apply(params['net'], static, ts, glob, reach, m, edge, econ)   # 2-tree: read net subtree (+ econ)
    return v


def make_rollout(net, T, pool_size, resign_p, resign_share, resign_turns):
    def rollout(params, states, rng, pool, cnt, resign_on):
        n = states.p_owner.shape[0]

        def one_step(carry, _):
            states, rng, cnt, resign_on = carry
            rng, sub, ridx, rres = jax.random.split(rng, 4)
            keys = jax.random.split(sub, n)
            rec0, rec1, edge, launch, angle, ships, target, arrival = jax.vmap(
                lambda s, k: per_env_actions(net, params, s, k))(states, keys)
            states2 = jax.vmap(env_step)(states, launch, angle, ships, target, arrival)
            t0, t1 = jax.vmap(ship_totals)(states2)
            share0 = t0 / (t0 + t1 + 1e-6)
            cnt = jnp.where(share0 > resign_share, jnp.maximum(cnt, 0) + 1,
                  jnp.where(share0 < 1.0 - resign_share, jnp.minimum(cnt, 0) - 1, 0))
            resign_done = resign_on & (jnp.abs(cnt) >= resign_turns)
            done = jax.vmap(is_done)(states2) | resign_done
            rew = jax.vmap(terminal_reward_p0)(states2).astype(jnp.float32) * done.astype(jnp.float32)
            idx = jax.random.randint(ridx, (n,), 0, pool_size)
            reset = jax.tree_util.tree_map(lambda p: p[idx], pool)
            states_next = jax.tree_util.tree_map(
                lambda a, b: jnp.where(done.reshape((n,) + (1,) * (a.ndim - 1)), b, a), states2, reset)
            cnt = jnp.where(done, 0, cnt)
            resign_on = jnp.where(done, jax.random.bernoulli(rres, resign_p, (n,)), resign_on)
            lcnt = launch.sum().astype(jnp.float32)
            etsum = jnp.where(done, states2.step, 0).sum().astype(jnp.float32)
            ecnt = done.sum().astype(jnp.float32)
            return (states_next, rng, cnt, resign_on), (rec0, rec1, edge, done, rew,
                                                        jnp.stack([lcnt, etsum, ecnt]))

        (states_f, _, cnt_f, resign_f), (recs0, recs1, edges, dones, rews, stats) = jax.lax.scan(
            one_step, (states, rng, cnt, resign_on), None, length=T)
        fv0 = jax.vmap(lambda s: _value(net, params, s, 0))(states_f)
        fv1 = jax.vmap(lambda s: _value(net, params, s, 1))(states_f)
        return recs0, recs1, edges, dones, rews, fv0, fv1, states_f, cnt_f, resign_f, stats.sum(0)
    return rollout


def gae(values, fv, rewards, dones, gamma, lam):
    def step(carry, t):
        gae_, next_v = carry
        m = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_v * m - values[t]
        gae_ = delta + gamma * lam * m * gae_
        return (gae_, values[t]), gae_
    (_, _), adv_rev = jax.lax.scan(step, (jnp.zeros(values.shape[1]), fv),
                                   jnp.arange(values.shape[0] - 1, -1, -1))
    adv = adv_rev[::-1]
    return adv, adv + values


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_envs", type=int, default=8)
    ap.add_argument("--board_pool", type=int, default=256)
    ap.add_argument("--board_pool_path", default=None)
    ap.add_argument("--board_pool_n", type=int, default=0)
    ap.add_argument("--T", type=int, default=128)
    ap.add_argument("--updates", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.999)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--ent", type=float, default=0.05)
    ap.add_argument("--vf", type=float, default=0.5)
    ap.add_argument("--E", type=int, default=64)
    ap.add_argument("--n_layers", type=int, default=4)
    ap.add_argument("--n_heads", type=int, default=4)
    ap.add_argument("--resign_p", type=float, default=0.5)
    ap.add_argument("--resign_share", type=float, default=0.75)
    ap.add_argument("--resign_turns", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval_every", type=int, default=0)
    ap.add_argument("--eval_games", type=int, default=64)
    ap.add_argument("--save_every", type=int, default=0)
    ap.add_argument("--save_dir", type=str, default=os.path.join(_HERE, "checkpoints"))
    ap.add_argument("--init_from", type=str, default="")
    ap.add_argument("--start_update", type=int, default=0)
    ap.add_argument("--num_minibatches", type=int, default=8)
    ap.add_argument("--lr_schedule", type=str, default="warmflat")
    ap.add_argument("--warmup_updates", type=int, default=1000)
    args = ap.parse_args()
    print("DEVICES:", jax.devices())

    # exp022 2-tree {net, frac}: net = pointer/value trunk; frac = clipped-Gaussian fraction head.
    net = OrbitNet19(E=args.E, n_layers=args.n_layers, n_heads=args.n_heads)
    frac = CoordFracGauss(E=args.E, n_heads=args.n_heads)
    if args.board_pool_path:
        _z = np.load(args.board_pool_path)
        pool = JaxState(**{f: jnp.asarray(_z[f]) for f in JaxState._fields})
        if args.board_pool_n:
            pool = jax.tree_util.tree_map(lambda p: p[:args.board_pool_n], pool)
        print(f"BOARD_POOL loaded {pool.p_owner.shape[0]} boards from {args.board_pool_path}")
    else:
        pool = gen_init_states(max(args.board_pool, args.n_envs), args.seed)
    pool_size = pool.p_owner.shape[0]
    states = jax.tree_util.tree_map(lambda p: p[:args.n_envs], pool)
    P = pool.p_owner.shape[1]
    rng = jax.random.PRNGKey(args.seed)
    rng, ki, ki2, kr0 = jax.random.split(rng, 4)
    single0 = jax.tree_util.tree_map(lambda x: x[0], pool)
    st0, ts0, gl0, m0, econ0 = basic_features(single0, 0)   # v36 econ-CNN: 5-tuple (+ econ_curves)
    _Rg0 = reach_probe_static(single0)
    reach0 = _Rg0 & ((single0.p_owner == 0) & single0.p_mask)[:, None]
    _R0, _A0, _T0, edge0 = edge_features(single0)
    net_params = net.init(ki, st0, ts0, gl0, reach0, m0, edge0, econ0)   # + econ
    # run the init'd net to get ctx0/emb0 -> seed the frac.init (frac needs emb[tid] + 6-d edge[tid] + tid)
    _ar0 = jnp.arange(P)
    tgt0, emb0, ctx0, _b0, _v0 = net.apply(net_params, st0, ts0, gl0, reach0, m0, edge0, econ0)
    tid0 = jnp.argmax(tgt0, -1)
    emb_tid0 = emb0[tid0]
    edge_tid0 = edge0[_ar0, tid0]                        # v36: ALL-IN 6 only (matches train_il_v5 frac head)
    intend0 = ((single0.p_owner == 0) & single0.p_mask) & (single0.p_ships > 0)
    frac_params = frac.init(ki2, ctx0, emb_tid0, edge_tid0, tid0, intend0)
    params = {'net': net_params, 'frac': frac_params}
    if args.init_from:
        with open(args.init_from, "rb") as _fh:
            params = jax.tree_util.tree_map(jnp.asarray, fser.msgpack_restore(_fh.read()))
        print(f"RESUME init_from={args.init_from} start_update={args.start_update}")
    n_params = sum(int(np.prod(p.shape)) for p in jax.tree_util.tree_leaves(params))
    print(f"MODEL_PARAMS {n_params}  (exp21 EDGE-MARGIN: pointer-only ALL-IN, edge-bias trunk, "
          f"E={args.E}, P={P}, gamma={args.gamma} lam={args.lam}, "
          f"resign p={args.resign_p} share>{args.resign_share} x{args.resign_turns}t)")

    total_opt = max(1, args.updates * args.epochs * args.num_minibatches)
    _off = args.start_update * args.epochs * args.num_minibatches
    if args.lr_schedule == "warmflat":
        _ws = max(1, args.warmup_updates * args.epochs * args.num_minibatches)
        lr_sched = lambda s: args.lr * jnp.minimum((s + _off + 1.0) / _ws, 1.0)
    else:
        lr_sched = args.lr
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(lr_sched))
    opt_state = opt.init(params)
    rollout = make_rollout(net, args.T, pool_size, args.resign_p, args.resign_share, args.resign_turns)
    rollout_jit = jax.jit(rollout)
    eval_fn = jax.jit(make_eval(net, frac, EPISODE_STEPS - 2)) if args.eval_every else None
    eval_states = (jax.tree_util.tree_map(lambda p: p[-min(args.eval_games, pool_size):], pool)
                   if (args.eval_every and args.board_pool_path) else
                   (gen_init_states(args.eval_games, seed=10_000) if args.eval_every else None))

    cnt = jnp.zeros((args.n_envs,), jnp.int32)
    resign_on = jax.random.bernoulli(kr0, args.resign_p, (args.n_envs,))
    ar = jnp.arange(P)

    def save_ckpt(u):
        os.makedirs(args.save_dir, exist_ok=True)
        with open(os.path.join(args.save_dir, f"ckpt_u{u:05d}.msgpack"), "wb") as fh:
            fh.write(fser.msgpack_serialize(jax.device_get(params)))
        json.dump({"update": int(u), "n_params": int(n_params), "E": args.E, "P": int(P),
                   "action": "pointer-only ALL-IN", "edge_bias": True, "E_edge": E_EDGE,
                   "gamma": args.gamma, "lam": args.lam,
                   "resign": {"p": args.resign_p, "share": args.resign_share, "turns": args.resign_turns},
                   "args": vars(args)}, open(os.path.join(args.save_dir, "meta.json"), "w"), default=str)

    def loss_fn(params, batch):                          # LEGACY train.py self-play: pointer-only PPO
        static, ts, glob, reach, mask, edge, econ, tid, acting, logp_old, adv, ret = batch

        def one(s_, t_, gl_, r_, m_, ed_, ec_, ti, ac):
            tgt, _e, _c, _b, v = net.apply(params['net'], s_, t_, gl_, r_, m_, ed_, ec_)   # 2-tree: net subtree (+ econ)
            lpt = jax.nn.log_softmax(tgt, -1)[ar, ti]
            lp = jnp.sum(jnp.where(ac, lpt, 0.0))
            p_t = jax.nn.softmax(tgt, -1)
            ent = -jnp.sum(jnp.where(ac, jnp.sum(p_t * jax.nn.log_softmax(tgt, -1), -1), 0.0)) / jnp.clip(jnp.sum(ac), 1.0)
            return lp, v, ent

        lp, v, ent = jax.vmap(one)(static, ts, glob, reach, mask, edge, econ, tid, acting)
        ratio = jnp.exp(lp - logp_old)
        adv_n = (adv - adv.mean()) / (adv.std() + 1e-8)
        pg = -jnp.minimum(ratio * adv_n, jnp.clip(ratio, 1 - args.clip, 1 + args.clip) * adv_n).mean()
        vloss = ((v - ret) ** 2).mean()
        entropy = ent.mean()
        loss = pg + args.vf * vloss - args.ent * entropy
        ev = 1.0 - jnp.var(ret - v) / (jnp.var(ret) + 1e-8)
        kl = (logp_old - lp).mean()
        clipfrac = (jnp.abs(ratio - 1.0) > args.clip).mean()
        return loss, (pg, vloss, entropy, ev, kl, clipfrac)

    @jax.jit
    def update(params, opt_state, batch):
        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, batch)
        upd, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, upd)
        return params, opt_state, loss, aux

    os.makedirs(args.save_dir, exist_ok=True)
    mf = open(os.path.join(args.save_dir, "metrics.csv"), ("a" if args.start_update else "w"), newline="")
    mw = csv.writer(mf)
    if not args.start_update:
        mw.writerow(["update", "sps", "gen_s", "done", "fsend", "endturn", "loss", "pg", "vloss",
                     "ent", "ev", "kl", "clipfrac", "eval_win", "eval_loss"])

    _tn = args.T * args.n_envs
    _COLS = [("upd", 6), ("samples", 8), ("SPS", 7), ("effSPS", 7), ("gen", 5), ("tot", 5),
             ("mem", 5), ("done", 5), ("fsend", 6), ("eturn", 6), ("loss", 8), ("pg", 9),
             ("vloss", 7), ("ent", 6), ("EV", 6), ("KL", 9), ("clipf", 6), ("eval", 9)]
    _HDR = " | ".join(f"{c:>{w}}" for c, w in _COLS)
    def _h(x):
        return f"{x/1e9:.2f}B" if x >= 1e9 else (f"{x/1e6:.2f}M" if x >= 1e6 else f"{x/1e3:.0f}K")
    print(f"ROLLOUT: T={args.T} x n_envs={args.n_envs} = {_tn:,} turns/seat (x2 = {2*_tn:,} samples/upd).",
          flush=True)
    print(_HDR, flush=True)

    for u in range(args.start_update, args.updates):
        t0 = time.time()
        rng, kr = jax.random.split(rng)
        (recs0, recs1, edges, dones, rews, fv0, fv1,
         states, cnt, resign_on, rstats) = rollout_jit(params, states, kr, pool, cnt, resign_on)
        jax.block_until_ready(rews)
        gen_t = time.time() - t0
        Tn, n = recs0[0].shape[0], recs0[0].shape[1]
        flat = lambda x: x.reshape((Tn * n,) + x.shape[2:])
        edge_flat = flat(edges)                                # (Ns, P,P,6) shared by both seats

        def make_batch(recs, rew_side, fv):
            static, ts, glob, reach, mask, econ, tid, acting, logp, value = recs   # v36: + econ (seat-specific)
            adv, ret = gae(value, fv, rew_side, dones, args.gamma, args.lam)
            return [flat(static), flat(ts), flat(glob), flat(reach), flat(mask), flat(econ),
                    flat(tid), flat(acting), flat(logp), flat(adv), flat(ret)]

        b0 = make_batch(recs0, rews, fv0)
        b1 = make_batch(recs1, -rews, fv1)
        Ns = b0[0].shape[0]; N = 2 * Ns; mb = max(1, N // args.num_minibatches); accs = []
        for _ in range(args.epochs):
            rng, kp = jax.random.split(rng)
            perm = jax.random.permutation(kp, N)
            for mbi in range(args.num_minibatches):
                idx = perm[mbi * mb:(mbi + 1) * mb]
                seat = idx >= Ns
                loc = jnp.where(seat, idx - Ns, idx)
                fields = tuple(jnp.where(seat.reshape((seat.shape[0],) + (1,) * (x0.ndim - 1)), x1[loc], x0[loc])
                               for x0, x1 in zip(b0, b1))
                edge_mb = edge_flat[loc]                        # seat-shared: gather by loc only
                # loss_fn order: static,ts,glob,reach,mask,EDGE,ECON,tid,acting,logp,adv,ret
                mbatch = (fields[0], fields[1], fields[2], fields[3], fields[4], edge_mb, fields[5],
                          fields[6], fields[7], fields[8], fields[9], fields[10])
                params, opt_state, loss, aux = update(params, opt_state, mbatch)
                accs.append(jnp.stack((loss,) + tuple(aux)))
        (loss, pg, vloss, entropy, ev, kl, clipfrac) = np.mean(np.asarray(jax.device_get(accs)), axis=0)
        tot_t = time.time() - t0
        sps = (Tn * n) / gen_t; eff_sps = (Tn * n) / tot_t
        peak_gb = jax.devices()[0].memory_stats().get('peak_bytes_in_use', 0) / 1e9
        _ls, _ets, _ec = np.asarray(jax.device_get(rstats))
        fsend = 1.0                                            # all-in: every launch IS full garrison
        endturn = _ets / max(_ec, 1.0)
        ew = el = None
        last = (u == args.updates - 1)
        if eval_fn is not None and (u % args.eval_every == 0 or last):
            win, loss_r = eval_fn(params, eval_states); ew, el = float(win), float(loss_r)
        evstr = f"{ew:.2f}/{el:.2f}" if ew is not None else ""
        _samp = 2 * (u + 1) * _tn
        if (u - args.start_update) % 40 == 0 and u != args.start_update:
            print(_HDR, flush=True)
        print(f"{u:6d} | {_h(_samp):>8} | {sps:7.0f} | {eff_sps:7.0f} | {gen_t:5.2f} | {tot_t:5.2f} | "
              f"{peak_gb:5.1f} | {_ec:5.0f} | {fsend:6.3f} | {endturn:6.1f} | {loss:+8.3f} | {pg:+9.5f} | "
              f"{vloss:7.4f} | {entropy:6.3f} | {ev:+6.3f} | {kl:+9.5f} | {clipfrac:6.3f} | {evstr:>9}",
              flush=True)
        mw.writerow([u, f"{sps:.0f}", f"{gen_t:.3f}", f"{_ec:.0f}", f"{fsend:.3f}", f"{endturn:.1f}",
                     f"{loss:.4f}", f"{pg:.4f}", f"{vloss:.4f}", f"{entropy:.4f}", f"{ev:.4f}",
                     f"{kl:.5f}", f"{clipfrac:.4f}", ("" if ew is None else f"{ew:.3f}"),
                     ("" if el is None else f"{el:.3f}")]); mf.flush()
        if args.save_every and (u % args.save_every == 0 or last):
            save_ckpt(u)
    mf.close()
    print("PIPELINE_OK")


if __name__ == "__main__":
    main()
