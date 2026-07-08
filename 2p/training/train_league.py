"""exp24 ilselfplay — LEAGUE training (forked from exp023 v9 train_league.py, ADAPTED to exp020 v36's
model: econ-CNN OrbitNet19 + CoordFracGauss, 2-tree {net,frac}, 6-d (ALL-IN ONLY) frac edge_tid).

Ports the exp17 v12 league regime (n_sp self-play lanes + n_lg league frozen-pool lanes, 80/20 split,
winrate-gated incremental snapshot admission, PFSP sampling, NO RESIGN) to v36's architecture so a
train_il_v5 warm-start checkpoint (2-tree {net,frac} msgpack) is directly --init_from loadable here.

DIFFERENCES vs the exp023 v9 source (the ONLY two arch changes needed to fit v36):
  (1) basic_features(st, me, fc=fc)  -> v36's signature has NO `keep_need` param (and there is NO
      min_hold_garrison in v36's env). So the `kn = min_hold_garrison(...)` hoist + keep_need= kwarg
      are DROPPED. Returns the 5-tuple (static, ts, glob, m, econ_curves) exactly like train_il_v5.
  (2) frac edge_tid = ALL-IN 6 ONLY (edge[ar, tid]); the half-garrison edge50 op-point + edge_partial
      + HALF_FRAC are DROPPED everywhere (per_env_actions_vs, loss_fn, frac.init seed). This MATCHES
      train_il_v5 (v36), whose CoordFracGauss edge_enc Dense is 6->32; an 11-d edge_tid would make the
      warm-start msgpack shape-MISMATCH on restore. The rec tuple therefore drops `edge50_tid` (13 ->
      12 fields after also having dropped minfrac in exp23 v9's 14).

Everything else (the LeaguePool gated-admission class, the 3-block SP/B/C rollout, per-seat margin
reward, the jit-scan PPO update) is IDENTICAL to exp023 v9.

LANES (static, jit-stable), n_envs split: n_sp self-play games + n_lg league games (80% / 20%):
  SP [0      : n_sp )        SELF, BOTH seats = current policy, BOTH sample + record (rec0, rec1)  [2x]
  B  [n_sp   : n_sp+nh)      LEAGUE, learner@p0 vs frozen POOL@p1  (record p0 only)                [1x]
  C  [n_sp+nh: n_envs)       LEAGUE, learner@p1 vs frozen POOL@p0  (record p1 only)                [1x]
total trained = 2*n_sp + n_lg per step. NO RESIGN (done = is_done only). gamma 0.999, lam 0.95.
Generated samples/update = 2 * n_envs * T (both seats rolled out everywhere).
"""
import os, sys, time, argparse, json, csv, glob, re

os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".jax_cache"))

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "jax_env"))

import numpy as np
import jax
import jax.numpy as jnp
import optax
import flax.serialization as fser

import train as tb                                    # exp24 econ-correct fns (edge_features, cg_*, gae, _value, make_eval)
from state import JaxState                            # noqa: E402
from step import step as env_step                     # noqa: E402
from env import (gen_init_states, basic_features, terminal_reward_p0, terminal_rewards, is_done,
                 EPISODE_STEPS, _forecast)            # noqa: E402  (v36: NO min_hold_garrison)
from targeting import reach_solve_static, lead_for_ships   # noqa: E402  (lead_for_ships = EXECUTED-count solve)
from model import OrbitNet19, SimpleFracMLP          # noqa: E402  (v41 frac = SimpleFracMLP -> clipped-Gaussian (mu,sigma))


def per_env_actions_vs(net, frac, p_learn, p_opp, lseat, st: JaxState, rng, greedy_opp=False, record_both=False):
    """exp24 league-lane action fn: pointer (target) + clipped-Gaussian FRACTION (amount). 2-tree
    {net,frac}; prm = {'net':.., 'frac':..}. Learner SAMPLES (target categorical + fraction Gaussian),
    opponent ARGMAX (greedy_opp) / sampled. record_both=True (SELF-PLAY): BOTH seats use p_learn, BOTH
    sample (on-policy), BOTH recorded -> 2x training data per self-play game; returns
    (rec0, rec1, launch, angle, ships, target, arrival). record_both=False (league lane / pure 1-seat):
    ONLY the learner seat is recorded; rec carries f + is_real (which sources launched). The
    EXECUTED-count lead solve (lead_for_ships(ships_sent)) is REQUIRED: a partial fleet's speed/lead
    differs from the all-in solve, so partials would fail the first-hit gate if launched at the all-in
    angle. Returns (rec_learner, launch, angle, ships, target, arrival).
    v36 ADAPTATION: basic_features 5-tuple (+ econ, NO keep_need); frac edge_tid = ALL-IN 6 only."""
    P = st.p_owner.shape[0]; ar = jnp.arange(P); f32 = jnp.float32
    k0t, k0f, k1t, k1f = jax.random.split(rng, 4)      # per-seat target + fraction keys
    fc = _forecast(st)                                 # seat-independent forecast, ONCE
    R, ANG, TURNS, Rg = reach_solve_static(st)         # all-in solve -> pointer mask + edge
    _, _, _, edge = tb.edge_features(st, fc=fc, lead=(R, ANG, TURNS))   # (P,P,6) all-in edge (seat-shared)
    garrison = st.p_ships

    def side(me, prm, kt, kf, record, greedy):
        static, ts, glob, m, econ = basic_features(st, me, fc=fc)   # v36: 5-tuple (+ econ_curves; NO keep_need)
        is_mine = (st.p_owner == me) & st.p_mask
        reach_me = Rg & is_mine[:, None]
        tgt, emb, gemb, _board, v = net.apply(prm['net'], static, ts, glob, reach_me, m, edge, econ)   # v41: slot3 = gemb (global token)
        acting = is_mine & (st.p_ships > 0)
        tid = jnp.argmax(tgt, -1) if greedy else jax.random.categorical(kt, tgt)
        is_real = acting & (tid != ar) & R[ar, tid]    # picked a real reachable target -> a launch
        emb_tid = emb[tid]                             # (P,E) chosen-target embedding
        mu, sigma = frac.apply(prm['frac'], emb, emb_tid, gemb)   # v41 SimpleFracMLP: [emb || emb_tid || gemb] -> (mu,sigma)
        f = jnp.clip(mu, 0.0, 1.0) if greedy else tb.cg_sample(kf, mu, sigma)
        ships_i = jnp.clip(jnp.round(f * garrison.astype(f32)).astype(jnp.int32), 0, garrison)
        rec = None
        if record:
            lpt = jax.nn.log_softmax(tgt, -1)[ar, tid]                       # pointer logp (acting)
            lpf = tb.cg_logp(f, mu, sigma)                                   # fraction logp (real targets)
            logp = jnp.sum(jnp.where(acting, lpt, 0.0)) + jnp.sum(jnp.where(is_real, lpf, 0.0))
            # v36 rec = 13 fields (exp23 v9 had 14 w/ edge50_tid; that op-point removed): drop edge50_tid.
            rec = (static, ts, glob, reach_me, m, edge, econ, tid, f, acting, is_real, logp, v)
        return rec, tid, ships_i, is_real

    if record_both:                                    # SELF-PLAY: both seats = p_learn, both SAMPLE + record
        rec0, tid0, sh0, rl0 = side(0, p_learn, k0t, k0f, True, False)
        rec1, tid1, sh1, rl1 = side(1, p_learn, k1t, k1f, True, False)
    elif lseat == 0:
        recL, tid0, sh0, rl0 = side(0, p_learn, k0t, k0f, True,  False)
        _,    tid1, sh1, rl1 = side(1, p_opp,   k1t, k1f, False, greedy_opp)
    else:
        _,    tid0, sh0, rl0 = side(0, p_opp,   k0t, k0f, False, greedy_opp)
        recL, tid1, sh1, rl1 = side(1, p_learn, k1t, k1f, True,  False)

    owner = st.p_owner
    tid = jnp.where(owner == 0, tid0, jnp.where(owner == 1, tid1, ar))
    ships_sent = jnp.where(owner == 0, sh0, jnp.where(owner == 1, sh1, 0))
    is_real = jnp.where(owner == 0, rl0, jnp.where(owner == 1, rl1, False))
    Rx, ANGx, TURNSx = lead_for_ships(st, ships_sent)  # EXECUTED-count lead (partial != all-in speed)
    angle = ANGx[ar, tid]; turns = TURNSx[ar, tid]
    cand = is_real & (ships_sent > 0) & Rx[ar, tid]
    launch = cand & tb.first_hit_gate(st, tid, angle, ships_sent)
    ships_final = jnp.where(launch, ships_sent, 0)
    if record_both:                                    # both on-policy seats recorded (p0, p1)
        return rec0, rec1, launch, angle, ships_final, tid, st.step + turns
    return recL, launch, angle, ships_final, tid, st.step + turns


def make_league_rollout(net, frac, T, pool_size, n_sp, n_lg, greedy_opp, margin_lam=0.0, margin_D=300.0):
    """exp24 league rollout over 3 blocks: SP[0:n_sp) SELF-PLAY (both seats = current policy, BOTH sample
    on-policy + BOTH recorded -> 2x training data/game), B[n_sp:n_sp+nh) learner@p0 vs frozen pool@p1,
    C[n_sp+nh:n) learner@p1 vs frozen pool@p0 (1 learner rec/game; frozen-opp off-policy -> dropped).
    Returns record halves h0 = SP.p0 ++ B (use seat0 reward R0), h1 = SP.p1 ++ C (seat1 reward R1) -> EACH
    has n_sp+nh rows, so total trained = 2*n_sp + n_lg per step. Plus per-half dones/PER-SEAT-rews/fv,
    carried states, per-league-lane (games, learner-wins) for PFSP EMA, and [launches, full-garrison
    launches, sum(end step), ends] stats. PER-SEAT margin reward (no negation). NO RESIGN."""
    nh = n_lg // 2
    nh_sp = n_sp // 2                                   # SP A0/A1 split (v13 argmax-opp regime): A0[0:nh_sp) A1[nh_sp:n_sp)
    n1, n2 = n_sp, n_sp + nh                            # block starts: SP[0:n_sp) B[n1:n2) C[n2:n)

    def rollout(params, opp_b, opp_c, states, rng, pool):
        n = states.p_owner.shape[0]
        sl = lambda tr, a, b: jax.tree_util.tree_map(lambda p: p[a:b], tr)

        def one_step(carry, _):
            states, rng = carry
            rng, sub, ridx = jax.random.split(rng, 3)
            keys = jax.random.split(sub, n)
            cat = lambda *xs: jnp.concatenate(xs, 0)
            if greedy_opp:
                # SP A0/A1 (v13 argmax-opp regime): learner SAMPLES vs CURRENT-SELF GREEDY, learner-seat only.
                # A0[0:nh_sp): learner@p0 vs self-greedy@p1 (rec p0). A1[nh_sp:n_sp): learner@p1 vs self-greedy@p0 (rec p1).
                st_a0 = sl(states, 0, nh_sp)
                recSP0, lA0, aA0, sA0, tA0, rrA0 = jax.vmap(
                    lambda s, k: per_env_actions_vs(net, frac, params, params, 0, s, k, True, False))(st_a0, keys[0:nh_sp])
                st_a1 = sl(states, nh_sp, n_sp)
                recSP1, lA1, aA1, sA1, tA1, rrA1 = jax.vmap(
                    lambda s, k: per_env_actions_vs(net, frac, params, params, 1, s, k, True, False))(st_a1, keys[nh_sp:n_sp])
                lSP, aSP, sSP = cat(lA0, lA1), cat(aA0, aA1), cat(sA0, sA1)
                tSP, rrSP = cat(tA0, tA1), cat(rrA0, rrA1)
            else:
                # ORIGINAL: SP self-play, BOTH seats = current policy, both sample + record (record_both=True) -> rec0, rec1
                st_sp = sl(states, 0, n_sp)
                recSP0, recSP1, lSP, aSP, sSP, tSP, rrSP = jax.vmap(
                    lambda s, k: per_env_actions_vs(net, frac, params, params, 0, s, k, greedy_opp, True))(st_sp, keys[0:n_sp])
            if nh:
                st_b, st_c = sl(states, n1, n2), sl(states, n2, n)
                recB, lB, aB, sB, tB, rB = jax.vmap(
                    lambda po, s, k: per_env_actions_vs(net, frac, params, po, 0, s, k, greedy_opp))(opp_b, st_b, keys[n1:n2])
                recC, lC, aC, sC, tC, rC = jax.vmap(
                    lambda po, s, k: per_env_actions_vs(net, frac, params, po, 1, s, k, greedy_opp))(opp_c, st_c, keys[n2:n])
                launch, angle, ships = cat(lSP, lB, lC), cat(aSP, aB, aC), cat(sSP, sB, sC)
                target, arrival = cat(tSP, tB, tC), cat(rrSP, rB, rC)
            else:
                recB = recC = None
                launch, angle, ships = lSP, aSP, sSP
                target, arrival = tSP, rrSP
            states2 = jax.vmap(env_step)(states, launch, angle, ships, target, arrival)
            done = jax.vmap(is_done)(states2)
            # PER-SEAT margin reward: r0 = seat0, r1 = seat1. margin_lam=0 -> pure zero-sum +-1.
            r0, r1 = jax.vmap(lambda s: terminal_rewards(s, lam=margin_lam, D=margin_D))(states2)
            r0 = r0 * done.astype(jnp.float32)
            r1 = r1 * done.astype(jnp.float32)
            idx = jax.random.randint(ridx, (n,), 0, pool_size)
            reset = jax.tree_util.tree_map(lambda p: p[idx], pool)
            states_next = jax.tree_util.tree_map(
                lambda a, b: jnp.where(done.reshape((n,) + (1,) * (a.ndim - 1)), b, a), states2, reset)
            lcnt = launch.sum().astype(jnp.float32)
            fcnt = (launch & (ships == states.p_ships)).sum().astype(jnp.float32)
            etsum = jnp.where(done, states2.step, 0).sum().astype(jnp.float32)
            ecnt = done.sum().astype(jnp.float32)
            stats_t = jnp.stack([lcnt, fcnt, etsum, ecnt])
            ys = ((recSP0, recSP1, recB, recC, done, r0, r1, stats_t) if nh
                  else (recSP0, recSP1, done, r0, r1, stats_t))
            return (states_next, rng), ys

        (states_f, _), ys = jax.lax.scan(one_step, (states, rng), None, length=T)
        if nh:
            rSP0, rSP1, rB, rC, dones, R0, R1, stats = ys
            h0 = tuple(jnp.concatenate([x, y], axis=1) for x, y in zip(rSP0, rB))   # SP.p0 ++ B (learner@p0)
            h1 = tuple(jnp.concatenate([x, y], axis=1) for x, y in zip(rSP1, rC))   # SP.p1 ++ C (learner@p1)
        else:
            rSP0, rSP1, dones, R0, R1, stats = ys
            h0, h1 = rSP0, rSP1
        # h0 = SP.p0 ++ B -> seat0 rewards R0; h1 = SP.p1 ++ C -> seat1 rewards R1.
        # argmax-opp: SP.p0=A0[0:nh_sp), SP.p1=A1[nh_sp:n_sp) (DISJOINT). original: both = SP[0:n_sp) (SP envs in BOTH).
        sp0a, sp0b = (0, nh_sp) if greedy_opp else (0, n_sp)        # SP envs feeding h0 (learner@p0)
        sp1a, sp1b = (nh_sp, n_sp) if greedy_opp else (0, n_sp)     # SP envs feeding h1 (learner@p1)
        d_h0 = jnp.concatenate([dones[:, sp0a:sp0b], dones[:, n1:n2]], axis=1)
        r_h0 = jnp.concatenate([R0[:, sp0a:sp0b], R0[:, n1:n2]], axis=1)
        d_h1 = jnp.concatenate([dones[:, sp1a:sp1b], dones[:, n2:n]], axis=1)
        r_h1 = jnp.concatenate([R1[:, sp1a:sp1b], R1[:, n2:n]], axis=1)
        fv0 = jax.vmap(lambda s: tb._value(net, params, s, 0))(states_f)
        fv1 = jax.vmap(lambda s: tb._value(net, params, s, 1))(states_f)
        fv_h0 = jnp.concatenate([fv0[sp0a:sp0b], fv0[n1:n2]])
        fv_h1 = jnp.concatenate([fv1[sp1a:sp1b], fv1[n2:n]])
        # per-league-lane outcomes for PFSP EMA (learner perspective; tie 0.5).
        dB, r0B = dones[:, n1:n2], R0[:, n1:n2]
        dC, r1C = dones[:, n2:n], R1[:, n2:n]
        f32 = jnp.float32
        games_l = jnp.concatenate([dB.sum(0), dC.sum(0)]).astype(f32)
        wins_l = jnp.concatenate([
            (r0B > 0).sum(0).astype(f32) + 0.5 * (dB & (r0B == 0)).sum(0).astype(f32),
            (r1C > 0).sum(0).astype(f32) + 0.5 * (dC & (r1C == 0)).sum(0).astype(f32)])
        return (h0, h1, d_h0, r_h0, d_h1, r_h1, fv_h0, fv_h1,
                states_f, games_l, wins_l, stats.sum(0))
    return rollout


class LeaguePool:
    """Host-side pool of <= max_slots frozen self-checkpoints with WINRATE-GATED INCREMENTAL ADMISSION
    (AlphaStar/FSP-style). A snapshot is admitted ONLY when the current policy has "mastered" the
    most-recently-admitted REFERENCE: current-vs-reference winrate >= admit_thresh, read FREE off the
    running PFSP-EMA of the reference slot. On trigger we admit the LATEST save-grid checkpoint on disk
    (<= current update) and switch the reference to it; its ema resets to optimistic 0.5 and climbs again
    -> admit_thresh self-paces admission DENSITY. max_admit_interval force-admits on a plateau; the save
    grid is the (non-binding) minimum spacing. Pool full -> FIFO-evict the OLDEST admitted snapshot.
    PFSP P(opp) prop-to (1-ema)^pfsp_p + floor over admitted members. u < min_admit_u (incl. u=0 random
    init / IL base) is NEVER admitted. IDENTICAL to exp023 v9."""

    def __init__(self, scan_dir, max_slots, template_params, pfsp_p, pfsp_floor, np_rng,
                 admit_thresh=0.70, max_admit_interval=10000, admit_min_games=256, save_every=500,
                 min_admit_u=1000):
        self.scan_dir, self.max_slots = scan_dir, max_slots
        self.pfsp_p, self.floor, self.np_rng = pfsp_p, pfsp_floor, np_rng
        self.admit_thresh = admit_thresh
        self.max_admit_interval = max_admit_interval
        self.admit_min_games = admit_min_games
        self.save_every = save_every
        self.min_admit_u = min_admit_u
        self.slot_update = [None] * max_slots            # member ckpt update int, or None = inactive
        self.ema = np.full(max_slots, 0.5, np.float64)   # learner-vs-member winrate estimate
        self.games = np.zeros(max_slots, np.int64)
        self.ref_slot = None                             # slot of the CURRENT reference (most-recent admission)
        self.last_admit_u = None                         # snapshot update of the current reference
        self.last_admit_at = 0                           # TRAINING update at which the last admission happened
        self.member_order = []                           # exp24: every-ever-admitted member ckpt-u, admission order (persistent league_wr columns)
        self.frozen_ema = {}                             # exp24: member_u -> frozen final EMA on FIFO eviction (column KEPT, not updated)
        self.stack = jax.tree_util.tree_map(lambda p: jnp.stack([p] * max_slots), template_params)

    def seed_il_base(self):
        """exp24: admit the warm-start (IL) weights as pool slot 0 at u=0 -> the FIRST reference.
        The stack is already pre-filled with template_params (= the loaded IL base), so this only marks
        slot 0 ACTIVE and makes it the reference. Subsequent grid snapshots (u>=min_admit_u) are then
        admitted only once the learner MASTERS this IL base (AlphaStar-style from IL, not from random)."""
        self.slot_update[0] = 0
        self.ema[0] = 0.5
        self.games[0] = 0
        self.ref_slot, self.last_admit_u, self.last_admit_at = 0, 0, 0
        self.member_order.append(0)                      # IL base u0 = first persistent league_wr column
        print("LEAGUE SEED u0 (IL warm-start base, LOAD-only) -> pool(1): u0:0.500", flush=True)

    def seed_preseed(self, updates):
        """RESUME seed (ported from v13 seed_preseed): load EXISTING grid ckpts (ASCENDING updates) into
        pool slots 0,1,2,... so the LARGEST update lands last -> becomes the reference. Replaces
        seed_il_base on a mid-run resume so the league starts ALREADY POPULATED with a tiered slice of
        this run's own grid (member params loaded from disk; ema resets to 0.5 + re-converges in ~50
        games). Caller sets last_admit_at = start_update after, to anchor the force-admit interval."""
        loaded = []
        for u in updates:                                # ascending -> largest lands last = most-recent reference
            if len(loaded) >= self.max_slots:
                print(f"LEAGUE PRESEED: max_slots={self.max_slots} reached, skipping u{u}+", flush=True); break
            t = self._load_ckpt(u)
            if t is None:
                print(f"LEAGUE PRESEED: ckpt_u{u:05d} not found in {self.scan_dir} -- skipped", flush=True); continue
            i = len(loaded)
            self.stack = jax.tree_util.tree_map(lambda s, p: s.at[i].set(p), self.stack, t)
            self.slot_update[i] = u; self.ema[i] = 0.5; self.games[i] = 0
            self.member_order.append(u); loaded.append(u)
        if loaded:
            self.ref_slot = self.slot_update.index(loaded[-1])
            self.last_admit_u = loaded[-1]
            print(f"LEAGUE PRESEED {len(loaded)} members u{loaded[0]}..u{loaded[-1]} ref=u{loaded[-1]}", flush=True)
        else:
            print("LEAGUE PRESEED 0 members (no ckpts found) -> pool starts EMPTY", flush=True)

    def _latest_grid_u(self, cur_u):
        best = None
        for f in glob.glob(os.path.join(self.scan_dir, "ckpt_u*.msgpack")):
            mm = re.search(r"ckpt_u(\d+)\.msgpack", os.path.basename(f))
            if mm:
                v = int(mm.group(1))
                if self.min_admit_u <= v <= cur_u and (best is None or v > best):
                    best = v
        return best

    def _load_ckpt(self, u):
        fp = os.path.join(self.scan_dir, f"ckpt_u{u:05d}.msgpack")
        if not os.path.exists(fp):
            return None
        with open(fp, "rb") as fh:
            return jax.tree_util.tree_map(jnp.asarray, fser.msgpack_restore(fh.read()))

    def _admit(self, grid_u, cur_u, reason):
        t = self._load_ckpt(grid_u)
        if t is None:
            return False
        free = [i for i, su in enumerate(self.slot_update) if su is None]
        i = free[0] if free else min(range(self.max_slots), key=lambda j: self.slot_update[j])
        if self.slot_update[i] is not None:              # FIFO eviction: FREEZE the outgoing member's last EMA (league_wr column kept, not updated)
            self.frozen_ema[self.slot_update[i]] = float(self.ema[i])
        self.stack = jax.tree_util.tree_map(lambda s, p: s.at[i].set(p), self.stack, t)
        self.slot_update[i] = grid_u
        self.member_order.append(grid_u)                 # new persistent league_wr column
        self.ema[i] = 0.5                                 # optimistic: learner has not mastered this new reference yet
        self.games[i] = 0
        self.ref_slot, self.last_admit_u, self.last_admit_at = i, grid_u, cur_u
        act = [(u, f"{self.ema[k]:.3f}") for k, u in enumerate(self.slot_update) if u is not None]
        print(f"LEAGUE ADMIT u{grid_u} (reason={reason}, ref<-u{grid_u}) -> pool({len(act)}): "
              + " ".join(f"u{u}:{e}" for u, e in act), flush=True)
        return True

    def maybe_admit(self, cur_u):
        grid = self._latest_grid_u(cur_u)
        if grid is None or (self.last_admit_u is not None and grid <= self.last_admit_u):
            return False
        if self.ref_slot is None:                         # FIRST admission: seed the reference
            return self._admit(grid, cur_u, "seed")
        ref_ready = self.games[self.ref_slot] >= self.admit_min_games
        if ref_ready and self.ema[self.ref_slot] >= self.admit_thresh:
            return self._admit(grid, cur_u, f"win{self.ema[self.ref_slot]:.2f}")
        if (cur_u - self.last_admit_at) >= self.max_admit_interval:
            return self._admit(grid, cur_u, f"interval{cur_u - self.last_admit_at}")
        return False

    def n_active(self):
        return sum(1 for x in self.slot_update if x is not None)

    def probs(self):
        act = [i for i, u in enumerate(self.slot_update) if u is not None]
        p = np.zeros(self.max_slots, np.float64)
        if not act:
            return p
        w = (1.0 - self.ema[act]) ** self.pfsp_p
        w = w / w.sum() if w.sum() > 1e-12 else np.full(len(act), 1.0 / len(act))
        p[act] = (1.0 - self.floor) * w + self.floor / len(act)
        return p / p.sum()

    def record_and_resample(self, opp_idx, games_l, wins_l, alpha):
        for lane in np.nonzero(games_l > 0)[0]:
            g, w = float(games_l[lane]), float(wins_l[lane])
            s = opp_idx[lane]
            if self.slot_update[s] is None:
                continue
            keep = (1.0 - alpha) ** g
            self.ema[s] = keep * self.ema[s] + (1.0 - keep) * (w / g)
            self.games[s] += int(g)
        p = self.probs()
        done_lanes = np.nonzero(games_l > 0)[0]
        if len(done_lanes) and p.sum() > 0:
            opp_idx[done_lanes] = self.np_rng.choice(self.max_slots, size=len(done_lanes), p=p)
        return opp_idx

    def mean_wr(self):
        act = [i for i, u in enumerate(self.slot_update) if u is not None]
        return float(self.ema[act].mean()) if act else float("nan")

    def stats_line(self, probs):
        return ";".join(f"u{u}:{self.ema[i]:.3f}:{probs[i]:.3f}:{self.games[i]}"
                        for i, u in enumerate(self.slot_update) if u is not None)

    def slot_ema_ordered(self):
        active = sorted(((self.slot_update[i], self.ema[i]) for i in range(self.max_slots)
                         if self.slot_update[i] is not None), key=lambda x: x[0])
        cells = [f"u{u}:{e:.4f}" for (u, e) in active]
        return cells + [""] * (self.max_slots - len(cells))

    def member_winrates(self):
        """exp24: rolling-EMA winrate for EVERY ever-admitted member, in admission order. Active members
        -> current EMA; FIFO-evicted -> FROZEN last EMA (column KEPT, never updated again). Returns a list
        of (member_u, winrate, is_frozen)."""
        active = {self.slot_update[i]: float(self.ema[i]) for i in range(self.max_slots)
                  if self.slot_update[i] is not None}
        return [(m, active[m] if m in active else self.frozen_ema.get(m, float("nan")), m not in active)
                for m in self.member_order]

    def league_wr_line(self, u):
        """One append-only league_wr.log line: per-member rolling EMA (evicted members tagged [F]=frozen)."""
        cells = [f"u{m}{'[F]' if fr else ''}:{wr:.4f}" for (m, wr, fr) in self.member_winrates()]
        return f"upd {u:>7} | members {len(cells)} | " + " ".join(cells)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_envs", type=int, default=128)
    ap.add_argument("--n_sp", type=int, default=-1)          # self-play lanes; -1 => n_envs//2 (half/half)
    ap.add_argument("--greedy_opp", type=int, default=1)     # A=1 (argmax opp), B=0 (sampled opp)
    ap.add_argument("--board_pool_path", default=None)
    ap.add_argument("--board_pool_n", type=int, default=0)
    ap.add_argument("--T", type=int, default=128)
    ap.add_argument("--updates", type=int, default=550000)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--gamma", type=float, default=0.999)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--ent", type=float, default=0.02)
    ap.add_argument("--vf", type=float, default=0.5)
    ap.add_argument("--margin_lam", type=float, default=0.0)  # 0 = PURE win/loss (margin OFF); 0.4 = tanh ship-diff bonus
    ap.add_argument("--margin_D", type=float, default=300.0)  # ship-diff scale for the tanh margin bonus (only if margin_lam>0)
    ap.add_argument("--E", type=int, default=128)            # v41: E=128
    ap.add_argument("--n_layers", type=int, default=6)       # v41: 6-layer trunk
    ap.add_argument("--n_heads", type=int, default=4)        # v41: 4 heads (d=E/heads=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval_every", type=int, default=0)
    ap.add_argument("--eval_games", type=int, default=64)
    ap.add_argument("--eval_until", type=int, default=0)      # >0: only run the vs-sniper eval while u < this (early debug)
    ap.add_argument("--save_every", type=int, default=500)   # pool fills from these saves
    ap.add_argument("--save_dir", type=str, default=os.path.join(_HERE, "checkpoints"))
    ap.add_argument("--init_from", type=str, default="")     # IL warm-start: a train_il_v5 2-tree {net,frac} msgpack
    ap.add_argument("--start_update", type=int, default=0)
    ap.add_argument("--resume_note", type=str, default="")   # printed in header to document a mid-run config change (e.g. "sigma_max 1.65->1.0")
    ap.add_argument("--preseed", type=str, default="")       # comma-sep grid updates to PRESEED the league at resume (ascending; v13-style). Takes priority over seed_il_base.
    ap.add_argument("--num_minibatches", type=int, default=16)
    ap.add_argument("--lr_schedule", type=str, default="warmflat")
    ap.add_argument("--warmup_updates", type=int, default=1000)
    ap.add_argument("--lr_min", type=float, default=3e-5)         # cosine floor: lr -> lr_min, then hold
    ap.add_argument("--cosine_updates", type=int, default=100000) # cosine anneal horizon (EXPECTED real run length, not --updates max)
    # --- league (winrate-gated incremental admission) ---
    ap.add_argument("--league", type=int, default=1)         # 0 => pure self-play (n_lg=0); 1 => league active from u0
    ap.add_argument("--pool_dir", type=str, default="")      # scan dir for member ckpts ('' = save_dir)
    ap.add_argument("--max_slots", type=int, default=30)     # pool capacity 30 (FIFO-evict oldest when full)
    ap.add_argument("--pfsp_p", type=float, default=1.0)     # LINEAR (1-wr)^1
    ap.add_argument("--pfsp_floor", type=float, default=0.02)
    ap.add_argument("--ema_alpha", type=float, default=0.02)
    ap.add_argument("--admit_thresh", type=float, default=0.70)
    ap.add_argument("--max_admit_interval", type=int, default=20000)
    ap.add_argument("--admit_min_games", type=int, default=1024)
    ap.add_argument("--min_admit_u", type=int, default=1000)  # earliest admittable ckpt update (u<this never enters; u=0 IL base never admitted)
    ap.add_argument("--league_stats_every", type=int, default=10)
    args = ap.parse_args()
    _ckpt_dir = os.path.join(args.save_dir, "ckpts")   # exp25: ckpts in their OWN subdir (separate from logs/scripts)
    os.makedirs(_ckpt_dir, exist_ok=True)
    print("DEVICES:", jax.devices())

    n_sp = (args.n_envs // 2) if args.n_sp < 0 else args.n_sp
    if not args.league:
        n_sp = args.n_envs
    n_lg = args.n_envs - n_sp
    nh = n_lg // 2
    assert n_sp > 0, "need at least one self-play lane"
    assert n_lg >= 0 and n_lg % 2 == 0, f"league lanes ({n_lg}) must be >=0 and even (B/C split)"
    greedy_opp = bool(args.greedy_opp)

    # exp24 2-tree {net, frac}: net = pointer/value econ-CNN trunk; frac = clipped-Gaussian fraction head.
    net = OrbitNet19(E=args.E, n_layers=args.n_layers, n_heads=args.n_heads)
    frac = SimpleFracMLP(E=args.E, n_heads=args.n_heads)   # v41: [emb || emb_tid || gemb] -> (mu,sigma)
    if args.board_pool_path:
        _z = np.load(args.board_pool_path)
        pool = JaxState(**{f: jnp.asarray(_z[f]) for f in JaxState._fields})
        if args.board_pool_n:
            pool = jax.tree_util.tree_map(lambda p: p[:args.board_pool_n], pool)
        print(f"BOARD_POOL loaded {pool.p_owner.shape[0]} boards from {args.board_pool_path}")
    else:
        pool = gen_init_states(max(256, args.n_envs), args.seed)
    pool_size = pool.p_owner.shape[0]
    states = jax.tree_util.tree_map(lambda p: p[:args.n_envs], pool)
    P = pool.p_owner.shape[1]
    rng = jax.random.PRNGKey(args.seed)
    rng, ki, ki2 = jax.random.split(rng, 3)

    single0 = jax.tree_util.tree_map(lambda x: x[0], pool)
    _fc0 = _forecast(single0)
    _R0, _A0, _T0, _Rg0 = reach_solve_static(single0)
    st0, ts0, gl0, m0, econ0 = basic_features(single0, 0, fc=_fc0)   # v36: 5-tuple (+ econ_curves; NO keep_need)
    reach0 = _Rg0 & ((single0.p_owner == 0) & single0.p_mask)[:, None]
    _, _, _, edge0 = tb.edge_features(single0, fc=_fc0, lead=(_R0, _A0, _T0))   # (P,P,6)
    net_params = net.init(ki, st0, ts0, gl0, reach0, m0, edge0, econ0)   # + econ
    # run the init'd net -> emb0/gemb0 to seed the frac.init (v41 SimpleFracMLP needs emb[tid] + gemb)
    tgt0, emb0, gemb0, _b0, _v0 = net.apply(net_params, st0, ts0, gl0, reach0, m0, edge0, econ0)   # v41: slot3 = gemb
    tid0 = jnp.argmax(tgt0, -1)
    emb_tid0 = emb0[tid0]
    frac_params = frac.init(ki2, emb0, emb_tid0, gemb0)   # v41 SimpleFracMLP: (emb, emb_tid, gemb)
    params = {'net': net_params, 'frac': frac_params}
    # exp25: IL warm-start is OPTIONAL (an on/off switch). With --init_from, the u0 pool member is that IL
    # base (LOAD-ONLY, shape-checked). WITHOUT --init_from, u0 = the freshly random-init params (the
    # random-start variant). Both are valid exp25 runs.
    if args.init_from:
        try:
            with open(args.init_from, "rb") as _fh:
                loaded = jax.tree_util.tree_map(jnp.asarray, fser.msgpack_restore(_fh.read()))
        except Exception as _e:
            raise SystemExit(f"FATAL: IL warm-start load FAILED (init_from={args.init_from}): {_e}. "
                             "Self-play ABORTED (u0 pool base is LOAD-ONLY; no random-init fallback).")
        if (jax.tree_util.tree_structure(loaded) != jax.tree_util.tree_structure(params) or
                [x.shape for x in jax.tree_util.tree_leaves(loaded)] !=
                [x.shape for x in jax.tree_util.tree_leaves(params)]):
            raise SystemExit(f"FATAL: IL warm-start shape/structure MISMATCH vs arch "
                             f"(E={args.E} n_layers={args.n_layers} n_heads={args.n_heads}). "
                             "Self-play ABORTED -- relaunch with the arch that trained this IL ckpt.")
        params = loaded
        print(f"WARM-START init_from={args.init_from} start_update={args.start_update} (IL base -> u0 pool seed)")
    else:
        print("RANDOM-INIT start (no --init_from): u0 pool seed = freshly random-init params "
              "(exp25 random-start variant).", flush=True)
    n_params = sum(int(np.prod(p.shape)) for p in jax.tree_util.tree_leaves(params))
    _mode = (f"LEAGUE ON (gated-admit): {n_sp} SP + {nh}+{nh} league lanes, pool<={args.max_slots} (FIFO), "
             f"admit_thresh={args.admit_thresh} min_games={args.admit_min_games} first_admit=u{args.min_admit_u} "
             f"max_interval={args.max_admit_interval}, PFSP p={args.pfsp_p} floor={args.pfsp_floor}" if args.league else
             f"LEAGUE OFF: {n_sp} SP lanes")
    print(f"MODEL_PARAMS {n_params}  (exp24 ilselfplay league {'argmaxOpp' if greedy_opp else 'sampledOpp'} "
          f"(2-tree net+frac econ-CNN, 6-d all-in frac edge, per-seat margin reward), "
          f"E={args.E}, P={P}, gamma={args.gamma} lam={args.lam}, NO-RESIGN, {_mode})", flush=True)
    if args.resume_note:
        print(f"### RESUME @u{args.start_update}: {args.resume_note} ###", flush=True)
    from constants import FLEET_CAP_PER_PLAYER as _FCAP, MAX_FLEETS as _MAXF
    _mb = ((n_sp if greedy_opp else 2 * n_sp) + n_lg) * args.T // args.num_minibatches
    _sched_str = (f"cosine -> {args.lr_min} over {args.cosine_updates}u" if args.lr_schedule == "cosine"
                  else f"{args.lr_schedule}, warmup={args.warmup_updates}")
    print(f"CONFIG | init_from={os.path.basename(args.init_from) or 'RANDOM'} "
          f"board_pool={os.path.basename(args.board_pool_path) if args.board_pool_path else 'gen_init'}"
          f"\n       | arch: E={args.E} n_layers={args.n_layers} n_heads={args.n_heads} ({n_params} params, 2-tree net+frac)"
          f"\n       | rollout: n_envs={args.n_envs} T={args.T} n_sp={n_sp} n_lg={n_lg} ({100*n_sp//args.n_envs}/{100*n_lg//args.n_envs} SP/league)"
          f"\n       | PPO: num_minibatches={args.num_minibatches} mb_size={_mb} epochs={args.epochs} lr={args.lr} ({_sched_str}) clip={args.clip} ent={args.ent} gamma={args.gamma} lam={args.lam} margin_lam={args.margin_lam}"
          f"\n       | env: FLEET_CAP={_FCAP}/side MAX_FLEETS={_MAXF} (train-only) NO-RESIGN greedy_opp={args.greedy_opp}"
          f"\n       | league: max_slots={args.max_slots} admit_thresh={args.admit_thresh} min_admit_u={args.min_admit_u} admit_min_games={args.admit_min_games} max_interval={args.max_admit_interval} PFSP_p={args.pfsp_p} floor={args.pfsp_floor} ema_alpha={args.ema_alpha}"
          f"\n       | run: updates={args.updates} save_every={args.save_every} seed={args.seed}", flush=True)

    np_rng = np.random.default_rng(args.seed + 777)
    if n_lg:
        lp = LeaguePool(args.pool_dir or _ckpt_dir, args.max_slots, params,
                        args.pfsp_p, args.pfsp_floor, np_rng,
                        admit_thresh=args.admit_thresh, max_admit_interval=args.max_admit_interval,
                        admit_min_games=args.admit_min_games, save_every=args.save_every,
                        min_admit_u=args.min_admit_u)
        if args.preseed:                                 # RESUME with v13-style tiered preseed (priority): league starts POPULATED
            lp.seed_preseed([int(x) for x in args.preseed.split(",") if x.strip() != ""])
            lp.last_admit_at = args.start_update         # anchor force-admit interval at the resume point
        elif args.init_from and args.start_update == 0:
            lp.seed_il_base()                            # IL warm-start (start_update==0): seed u0. On RESUME (start_update>0): SKIP -> keep u0-not-pooled (league re-admits from disk ckpts).
        # else (random init): pool starts EMPTY -> first admit = first grid ckpt >= min_admit_u (u1000);
        # league lanes bootstrap vs CURRENT self until then (no useless random-u0 in the pool).
        probs = lp.probs()
        opp_idx = (np_rng.choice(args.max_slots, size=n_lg, p=probs).astype(np.int64)
                   if probs.sum() > 0 else np.zeros(n_lg, np.int64))
    else:
        lp, probs, opp_idx = None, None, np.zeros(0, np.int64)

    total_opt = max(1, args.updates * args.epochs * args.num_minibatches)
    _off = args.start_update * args.epochs * args.num_minibatches
    if args.lr_schedule == "warmflat":
        _ws = max(1, args.warmup_updates * args.epochs * args.num_minibatches)
        lr_sched = lambda s: args.lr * jnp.minimum((s + _off + 1.0) / _ws, 1.0)
    elif args.lr_schedule == "cosine":
        # cosine anneal lr -> lr_min over cosine_updates (the EXPECTED real horizon, NOT --updates max),
        # then HOLD at lr_min. NO warmup (starts at peak lr). s clamped so past the horizon stays at floor.
        _cT = max(1, args.cosine_updates * args.epochs * args.num_minibatches)
        lr_sched = lambda s: args.lr_min + 0.5 * (args.lr - args.lr_min) * (1.0 + jnp.cos(jnp.pi * jnp.minimum((s + _off) / _cT, 1.0)))
    else:
        lr_sched = args.lr
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(lr_sched))
    opt_state = opt.init(params)
    if args.init_from:                                            # WARM-RESTART: restore Adam moments from the .opt sidecar if present (else cold Adam)
        _optf = (args.init_from[:-len(".msgpack")] + ".opt.msgpack") if args.init_from.endswith(".msgpack") else (args.init_from + ".opt.msgpack")
        if os.path.exists(_optf):
            try:
                with open(_optf, "rb") as _ofh:
                    opt_state = fser.from_bytes(opt_state, _ofh.read())
                print(f"WARM-ADAM: restored optimizer state from {_optf}", flush=True)
            except Exception as _oe:
                print(f"opt-state restore FAILED ({_oe}); using cold Adam", flush=True)
        else:
            print(f"COLD-ADAM: no optimizer sidecar at {_optf} (fresh Adam moments this restart)", flush=True)

    rollout = make_league_rollout(net, frac, args.T, pool_size, n_sp, n_lg, greedy_opp,
                                  args.margin_lam, args.margin_D)
    rollout_jit = jax.jit(rollout)
    eval_fn = jax.jit(tb.make_eval(net, frac, EPISODE_STEPS - 2)) if args.eval_every else None
    eval_states = (jax.tree_util.tree_map(lambda p: p[-min(args.eval_games, pool_size):], pool)
                   if (args.eval_every and args.board_pool_path) else
                   (gen_init_states(args.eval_games, seed=10_000) if args.eval_every else None))

    league_active = bool(args.league)

    ar = jnp.arange(P)

    def save_ckpt(u):
        os.makedirs(_ckpt_dir, exist_ok=True)
        with open(os.path.join(_ckpt_dir, f"ckpt_u{u:05d}.msgpack"), "wb") as fh:
            fh.write(fser.msgpack_serialize(jax.device_get(params)))
        try:                                                                            # optimizer (Adam) state sidecar -> warm restarts (best-effort; NEVER crash training)
            with open(os.path.join(_ckpt_dir, f"ckpt_u{u:05d}.opt.msgpack"), "wb") as fh:
                fh.write(fser.to_bytes(jax.device_get(opt_state)))                       # to_bytes handles optax tuple/NamedTuple state (msgpack_serialize cannot)
        except Exception as _se:
            print(f"opt-state save skipped ({_se})", flush=True)
        json.dump({"update": int(u), "n_params": int(n_params), "E": args.E, "P": int(P),
                   "action": "pointer + clipped-Gaussian fraction (coord+edge, 6-d all-in)", "edge_bias": True,
                   "edge_pointer": True, "param_trees": ["net", "frac"], "econ_cnn": True,
                   "reward": f"per-seat margin (lam={args.margin_lam},D={args.margin_D})",
                   "version": ("A-argmaxOpp" if greedy_opp else "B-sampledOpp"),
                   "gamma": args.gamma, "lam": args.lam, "resign": False,
                   "league": {"on": bool(args.league), "n_sp": n_sp, "n_lg": n_lg,
                              "max_slots": args.max_slots, "pool": ([u_ for u_ in lp.slot_update if u_ is not None] if lp else []),
                              "ref_u": (lp.last_admit_u if lp else None),
                              "admit_thresh": args.admit_thresh, "max_admit_interval": args.max_admit_interval,
                              "admit_min_games": args.admit_min_games, "min_admit_u": args.min_admit_u,
                              "pfsp_p": args.pfsp_p, "pfsp_floor": args.pfsp_floor, "ema_alpha": args.ema_alpha,
                              "greedy_opp": greedy_opp},
                   "args": vars(args)},
                  open(os.path.join(args.save_dir, "meta.json"), "w"), default=str)

    def loss_fn(params, batch):                          # exp24 2-tree {net,frac}: pointer + fraction PPO
        # v36: 14 fields (exp23 v9 had 15 w/ edge50_tid; that op-point removed):
        # static, ts, glob, reach, mask, edge, econ, tid, f, acting, is_real, logp_old, adv, ret
        static, ts, glob, reach, mask, edge, econ, tid, f, acting, is_real, logp_old, adv, ret = batch

        def one(s_, t_, gl_, r_, m_, ed_, ec_, ti, fi, ac, rl):
            tgt, emb, gemb, _b, v = net.apply(params['net'], s_, t_, gl_, r_, m_, ed_, ec_)   # v41: slot3 = gemb
            emb_tid = emb[ti]                            # (P,E) chosen-target embedding
            mu, sigma = frac.apply(params['frac'], emb, emb_tid, gemb)   # v41 SimpleFracMLP
            lpt = jax.nn.log_softmax(tgt, -1)[ar, ti]                         # pointer logp
            lpf = tb.cg_logp(fi, mu, sigma)                                   # fraction logp
            lp = jnp.sum(jnp.where(ac, lpt, 0.0)) + jnp.sum(jnp.where(rl, lpf, 0.0))
            p_t = jax.nn.softmax(tgt, -1)
            ent_ptr = -jnp.sum(jnp.where(ac, jnp.sum(p_t * jax.nn.log_softmax(tgt, -1), -1), 0.0)) / jnp.clip(jnp.sum(ac), 1.0)
            ent_frac = jnp.sum(jnp.where(rl, tb.cg_entropy(mu, sigma), 0.0)) / jnp.clip(jnp.sum(rl), 1.0)
            _rlc = jnp.clip(jnp.sum(rl), 1.0)                                  # sigma stats over real-launch rows
            msig = jnp.sum(jnp.where(rl, sigma, 0.0)) / _rlc                   # per-env mean sigma
            msig_sq = jnp.sum(jnp.where(rl, sigma * sigma, 0.0)) / _rlc        # per-env mean sigma^2
            return lp, v, ent_ptr, ent_frac, msig, msig_sq

        lp, v, ent_ptr, ent_frac, msig, msig_sq = jax.vmap(one)(static, ts, glob, reach, mask, edge, econ, tid, f, acting, is_real)
        ent = ent_ptr + ent_frac                          # per-env total entropy (for the bonus)
        ratio = jnp.exp(lp - logp_old)
        adv_n = (adv - adv.mean()) / (adv.std() + 1e-8)
        pg = -jnp.minimum(ratio * adv_n, jnp.clip(ratio, 1 - args.clip, 1 + args.clip) * adv_n).mean()
        vloss = ((v - ret) ** 2).mean()
        entropy = ent.mean()
        loss = pg + args.vf * vloss - args.ent * entropy
        ev = 1.0 - jnp.var(ret - v) / (jnp.var(ret) + 1e-8)
        kl = (logp_old - lp).mean()
        clipfrac = (jnp.abs(ratio - 1.0) > args.clip).mean()
        ent_ptr_m = ent_ptr.mean(); ent_frac_m = ent_frac.mean()             # split-entropy logging (ent = entP + entF)
        mean_sig = msig.mean()                                               # avg sigma over batch
        std_sig = jnp.sqrt(jnp.clip(msig_sq.mean() - mean_sig ** 2, 0.0))    # sigma-of-sigma (spread of sigma)
        return loss, (pg, vloss, entropy, ev, kl, clipfrac, ent_ptr_m, ent_frac_m, mean_sig, std_sig)

    @jax.jit
    def update(params, opt_state, batch):
        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, batch)
        upd, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, upd)
        return params, opt_state, loss, aux

    @jax.jit
    def run_epoch(params, opt_state, b0, b1, perm):
        """Fuse ONE epoch's whole minibatch loop into a single jit'd lax.scan -> 1 GPU dispatch/epoch.
        Sequential param/opt_state threading through the scan is IDENTICAL to the old Python loop."""
        Ns = b0[0].shape[0]; mb = (2 * Ns) // args.num_minibatches
        idxs = perm[:args.num_minibatches * mb].reshape(args.num_minibatches, mb)
        def step(carry, idx):
            p, ostate = carry
            seat = idx >= Ns
            loc = jnp.where(seat, idx - Ns, idx)
            mbatch = tuple(jnp.where(seat.reshape((seat.shape[0],) + (1,) * (x0.ndim - 1)), x1[loc], x0[loc])
                           for x0, x1 in zip(b0, b1))
            p, ostate, loss, aux = update(p, ostate, mbatch)
            return (p, ostate), jnp.stack((loss,) + tuple(aux))
        (params, opt_state), accs = jax.lax.scan(step, (params, opt_state), idxs)
        return params, opt_state, accs                      # accs (num_minibatches, 7)

    os.makedirs(args.save_dir, exist_ok=True)
    # exp24 logging: EXACTLY two logs. TRAINING = stdout (-> sp_train.log via shell redirect); LEAGUE = the
    # member-keyed rolling-EMA winrate log below. (metrics_league / league_stats / slot_ema CSVs dropped.)
    lwf = None
    if n_lg:
        lwf = open(os.path.join(args.save_dir, "sp_league.log"), "a")   # the ONE league log (member-keyed rolling EMA, evicted=[F]frozen)

    _tn = args.T * args.n_envs                            # ENV-TURNS/update = games x T
    _gen = 2 * args.T * args.n_envs                       # GENERATED samples/update = both seats forward-passed
    _gs = ((n_sp if greedy_opp else 2 * n_sp) + n_lg) * args.T   # TRAINING (gradient) samples/update: argmaxOpp SP=1x (A0/A1 learner-only), else SP=2x; + LEAGUE 1x
    _mbsize = max(1, _gs // args.num_minibatches)
    _COLS = [("upd", 6), ("trSamp", 8), ("genSmp", 8), ("SPS", 7), ("effSPS", 7), ("gen", 5), ("tot", 5), ("elapsed", 9),
             ("mem", 5), ("done", 5), ("fsend", 6), ("eturn", 6), ("pWR", 5), ("loss", 8), ("pg", 9),
             ("vloss", 7), ("ent", 6), ("entP", 6), ("entF", 6), ("mSig", 6), ("sSig", 6), ("EV", 6), ("KL", 9), ("clipf", 6), ("eval", 9)]
    _HDR = " | ".join(f"{c:>{w}}" for c, w in _COLS)
    def _h(x):
        return f"{x/1e9:.2f}B" if x >= 1e9 else (f"{x/1e6:.2f}M" if x >= 1e6 else f"{x/1e3:.0f}K")
    def _hms(s):
        s = int(s); return f"{s//3600}:{(s % 3600)//60:02d}:{s % 60:02d}"
    print(f"ROLLOUT: T={args.T} x n_envs={args.n_envs} ({n_sp} SP both-seats[2x] + {nh}+{nh} league) = {_tn:,} ENV-TURNS/update.",
          flush=True)
    print(f"TRAIN: 'trSamp'=cumulative TRAINING samples, 'genSmp'=cumulative GENERATED. per update: "
          f"gen={_gen:,} (2 x T x n_envs), train={_gs:,} = SP 2x + LEAGUE 1x. PPO batch N={_gs:,} | "
          f"num_minibatches={args.num_minibatches} | mb_size={_mbsize:,} | epochs={args.epochs} | "
          f"lr-sched={_sched_str} | opp={'ARGMAX' if greedy_opp else 'SAMPLED'}.", flush=True)
    print(_HDR, flush=True)

    _cum = args.start_update * _gs                        # cumulative TRAINING samples
    _cum_gen = args.start_update * _gen                   # cumulative GENERATED samples
    _t_start = time.time()
    for u in range(args.start_update, args.updates):
        t0 = time.time()
        rng, kr = jax.random.split(rng)
        if n_lg and league_active and lp.n_active() > 0:
            opp_b = jax.tree_util.tree_map(lambda s: s[jnp.asarray(opp_idx[:nh])], lp.stack)
            opp_c = jax.tree_util.tree_map(lambda s: s[jnp.asarray(opp_idx[nh:])], lp.stack)
        elif n_lg:
            # bootstrap (no ckpts yet): league lanes play vs CURRENT self, broadcast to nh lanes. SAME
            # (nh,*param) shape as the gather above -> NO recompile on the empty->active transition.
            opp_b = opp_c = jax.tree_util.tree_map(lambda p: jnp.broadcast_to(p, (nh,) + p.shape), params)
        else:
            opp_b = opp_c = params                        # nh==0 (pure self-play): ignored at trace
        (h0, h1, d_h0, r_h0, d_h1, r_h1, fv_h0, fv_h1,
         states, games_l, wins_l, rstats) = rollout_jit(params, opp_b, opp_c, states, kr, pool)
        jax.block_until_ready(r_h0)
        gen_t = time.time() - t0
        Tn, nl0 = h0[0].shape[0], h0[0].shape[1]
        flat = lambda x: x.reshape((Tn * nl0,) + x.shape[2:])

        def make_batch(recs, rew_side, dn, fv):
            # exp24: rec = 13 fields; flatten 12 (drop value, which feeds GAE) + adv + ret = 14 returned.
            static, ts, glob, reach, mask, edge, econ, tid, f, acting, is_real, logp, value = recs
            adv, ret = tb.gae(value, fv, rew_side, dn, args.gamma, args.lam)
            return [flat(static), flat(ts), flat(glob), flat(reach), flat(mask), flat(edge), flat(econ),
                    flat(tid), flat(f), flat(acting), flat(is_real), flat(logp), flat(adv), flat(ret)]

        b0 = make_batch(h0, r_h0, d_h0, fv_h0)            # learner@p0 (SP.p0 + B), seat0 reward
        b1 = make_batch(h1, r_h1, d_h1, fv_h1)            # learner@p1 (SP.p1 + C), seat1 reward (NO flip: per-seat)
        Ns = b0[0].shape[0]; N = 2 * Ns; accs = []
        for _ in range(args.epochs):
            rng, kp = jax.random.split(rng)
            perm = jax.random.permutation(kp, N)
            params, opt_state, accs_e = run_epoch(params, opt_state, b0, b1, perm)   # 1 jit'd scan/epoch
            accs.append(accs_e)
        (loss, pg, vloss, entropy, ev, kl, clipfrac, ent_ptr_m, ent_frac_m, mean_sig, std_sig) = np.mean(np.asarray(jax.device_get(jnp.concatenate(accs, 0))), axis=0)

        # --- league bookkeeping + gated incremental admission. EMA first (so maybe_admit reads the
        # FRESHEST reference winrate), then admit, then per-update slot-ema log. ---
        if n_lg and league_active:
            opp_idx = lp.record_and_resample(opp_idx, np.asarray(jax.device_get(games_l)),
                                             np.asarray(jax.device_get(wins_l)), args.ema_alpha)   # rolling EMA: EVERY update
            pool_wr = lp.mean_wr()
            lwf.write(lp.league_wr_line(u) + "\n"); lwf.flush()   # the ONE league log (member-keyed rolling EMA, evicted=[F]frozen)
        else:
            pool_wr = float("nan")

        tot_t = time.time() - t0
        sps = _tn / gen_t; eff_sps = _tn / tot_t
        peak_gb = jax.devices()[0].memory_stats().get('peak_bytes_in_use', 0) / 1e9
        rs = np.asarray(jax.device_get(rstats))
        fsend = rs[1] / max(rs[0], 1.0)
        eturn = rs[2] / max(rs[3], 1.0)
        ew = el = None
        last = (u == args.updates - 1)
        _eval_window = (args.eval_until <= 0) or (u < args.eval_until)
        if eval_fn is not None and _eval_window and (u % args.eval_every == 0 or last):
            win, loss_r = eval_fn(params, eval_states); ew, el = float(win), float(loss_r)
        evstr = f"{ew:.2f}/{el:.2f}" if ew is not None else ""
        _cum += _gs
        _cum_gen += _gen
        if (u - args.start_update) % 40 == 0 and u != args.start_update:
            print(_HDR, flush=True)
        print(f"{u:6d} | {_h(_cum):>8} | {_h(_cum_gen):>8} | {sps:7.0f} | {eff_sps:7.0f} | {gen_t:5.2f} | {tot_t:5.2f} | "
              f"{_hms(time.time()-_t_start):>9} | "
              f"{peak_gb:5.1f} | {rs[3]:5.0f} | {fsend:6.3f} | {eturn:6.1f} | {pool_wr:5.3f} | {loss:+8.3f} | {pg:+9.5f} | "
              f"{vloss:7.4f} | {entropy:6.3f} | {ent_ptr_m:6.3f} | {ent_frac_m:6.3f} | {mean_sig:6.4f} | {std_sig:6.4f} | {ev:+6.3f} | {kl:+9.5f} | {clipfrac:6.3f} | {evstr:>9}", flush=True)
        if args.save_every and (u % args.save_every == 0 or last):
            save_ckpt(u)
            # exp25: admission is checked ONLY here, at the save point (the just-saved ckpt is now the disk grid).
            # The rolling EMA winrate is updated every update but READ ONCE here: if the latest reference has been
            # rolled to >= admit_thresh (and >= admit_min_games), admit the just-saved ckpt. No per-update monitor.
            if n_lg and league_active and lp.maybe_admit(u):
                probs = lp.probs()
                if probs.sum() > 0:
                    opp_idx = np_rng.choice(args.max_slots, size=n_lg, p=probs).astype(np.int64)
    if lwf is not None:
        lwf.close()
    print("PIPELINE_OK")


if __name__ == "__main__":
    main()
