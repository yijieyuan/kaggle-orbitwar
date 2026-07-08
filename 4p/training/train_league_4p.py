"""exp24-4p — 4p FFA self-play LEAGUE (warm-started from best combined-IL), v37 econ-CNN, 2-tree {net,frac}.

The 4p analog of the 2p exp24 self-play league. Port = exp022-4p's 4-SEAT self-play machinery
(per-seat canonical view, FFA per-seat reward, seat-block PPO minibatching) wired to the v37 ECON-CNN
model (5-tuple basic_features + 7-arg net.apply(+econ) + 6-d ALL-IN frac edge_tid, exactly like the 2p
exp24 train_league), with the user's 2026-06-18 4p league design:

  ROLLOUT (per update, jit-stable):
    * ANCHOR seat 0 is ALWAYS the CURRENT policy (so every game has >=1 learner seat).
    * Non-anchor seats 1,2,3 are EACH INDEPENDENTLY current (1-league_p, default 0.8) or league
      (league_p=0.2). The assignment `lg` (n_envs, 4) is drawn ONCE per update (static over the T-step
      scan, like 2p's static lanes); seat 0 forced current.
    * THREE INDEPENDENT pool members O1,O2,O3 are PFSP-sampled per update (one per non-anchor seat).
      When seat s in {1,2,3} is a league seat, it is played by O_{s} (so two league seats in one game
      are DIFFERENT checkpoints). Members come from ONE SHARED pool.
    * EVERY current-controlled seat is RECORDED and trained on-policy (anchor + the ~80% current
      non-anchor seats); league seats are MASKED out of the loss via a per-(env,seat) `valid` flag.
    * Reward = per-seat FFA `terminal_rewards(lam=margin_lam)` (lam=0 => pure sign(ships_i-max_others)).
    * NO RESIGN (user: match 2p). done = is_done only. gamma 0.999 lam 0.95.

  LEAGUE POOL (two winrates, user 2026-06-18):
    * ONE shared pool of frozen self-checkpoints (FIFO, max_slots). Seeded with the IL warm-start base
      at u=0 (the first reference), AlphaStar/FSP-from-IL.
    * Per member, track TWO EMAs from the games it plays as a league opponent:
        ema_first  = its 1st-place rate (strict ships winner)  -> PFSP sampling P ~ ema_first^pfsp_p+floor.
        ema_rank   = "current(anchor) out-ranks this member" rate (ships_anchor > ships_member) -> ADMISSION.
    * ADMISSION (== 2p's "mastered the reference 70%", recast for FFA via pairwise RANK): when the
      current REFERENCE's ema_rank >= admit_thresh (0.70) after >= admit_min_games, admit the latest
      save-grid ckpt and make IT the new reference (ema reset). Plateau -> force-admit every
      max_admit_interval. u < min_admit_u (incl u=0 IL base) never admitted.

Warm-start: --init_from <best IL .msgpack> replaces the freshly-init'd {net,frac} (same arch -> same
keys/shapes). Pairs with --start_update for RL resume.

Gradient samples/update = (#current seats) * T ~= (1 + 3*(1-league_p)) * T * n_envs (league seats masked).
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

import train as tb                                    # v37 econ-CNN helpers (cg_*, edge_features, gae, _value)
from state import JaxState                            # noqa: E402
from step import step as env_step                     # noqa: E402
from constants import N_PLAYERS, MAX_FLEETS           # noqa: E402
from env import (gen_init_states, basic_features, ship_totals, terminal_rewards,
                 is_done, EPISODE_STEPS, _forecast)   # noqa: E402
from targeting import reach_solve_static, lead_for_ships   # noqa: E402
from model import OrbitNet19, SimpleFracMLP           # noqa: E402  (v41 frac = SimpleFracMLP[emb||emb_tid||gemb])

# Default None -> procedural gen_init_states(reset(4)) boards (mirrors 2p train_league.py). Pass
# --board_pool_path <boards.npz> to load a saved pool. (The exact deploy pool is not bundled.)
DEFAULT_BOARD_POOL_4P = None


def load_board_pool(path, board_pool_n=0):
    """Load a 4p board-pool npz into a JaxState (F-pads up to MAX_FLEETS if baked smaller)."""
    _z = np.load(path)

    def _pad_f(name, a):
        if not name.startswith("f_") or a.ndim < 2 or a.shape[1] >= MAX_FLEETS:
            return a
        pad = np.zeros((a.shape[0], MAX_FLEETS - a.shape[1]) + a.shape[2:], a.dtype)
        if name in ("f_target", "f_arrival"):
            pad[:] = -1
        return np.concatenate([a, pad], axis=1)
    pool = JaxState(**{f: jnp.asarray(_pad_f(f, np.asarray(_z[f]))) for f in JaxState._fields})
    if board_pool_n:
        pool = jax.tree_util.tree_map(lambda p: p[:board_pool_n], pool)
    return pool


# rec = 14 fields (the 2p exp24 13-field econ rec + a per-(env,seat) `valid` mask):
#   (static, ts, glob, reach, mask, edge, econ, tid, f, acting, is_real, logp, v, valid)
N_REC = 14


def per_env_actions_league_4p(net, frac, cur, opps, is_league, st: JaxState, rng):
    """4-seat action for ONE env. ALL 4 seats run the CURRENT policy (recorded, econ-CNN, sampled);
    non-anchor seats 1,2,3 ALSO run their league opponent O_{s} (opps[s-1]); the seat PLAYS the opp
    action where is_league[s], else the current action. Anchor seat 0 always current. The recorded rec
    is the CURRENT forward with valid = ~is_league[seat] (league-env transitions masked out of the loss).

    opps    = (O1, O2, O3) frozen {net,frac} param trees for seats 1,2,3.
    is_league = (4,) bool, index 0 always False.
    Returns (recs(tuple of 4, each 14 fields), edge, launch, angle, ships_final, target, arrival)."""
    P = st.p_owner.shape[0]; ar = jnp.arange(P); f32 = jnp.float32
    keys = jax.random.split(rng, 2 * N_PLAYERS + 2 * 3)   # 4 current seats + 3 opp seats, (tgt,frac) each
    fc = _forecast(st)
    R, ANG, TURNS, Rg = reach_solve_static(st)
    _, _, _, edge = tb.edge_features(st, fc=fc, lead=(R, ANG, TURNS))
    garrison = st.p_ships

    def side(me, prm, kt, kf, valid, record):
        static, ts, glob, m, econ = basic_features(st, me, fc=fc)   # v37 econ-CNN 5-tuple
        is_mine = (st.p_owner == me) & st.p_mask
        reach_me = Rg & is_mine[:, None]
        tgt, emb, gemb, _board, v = net.apply(prm['net'], static, ts, glob, reach_me, m, edge, econ)   # v41: slot3=gemb
        acting = is_mine & (st.p_ships > 0)
        tid = jax.random.categorical(kt, tgt)            # always sample on-policy; self=hold legal
        is_real = acting & (tid != ar) & R[ar, tid]
        emb_tid = emb[tid]
        mu, sigma = frac.apply(prm['frac'], emb, emb_tid, gemb)   # v41 SimpleFracMLP: [emb || emb_tid || gemb]
        f = tb.cg_sample(kf, mu, sigma)
        ships_i = jnp.clip(jnp.round(f * garrison.astype(f32)).astype(jnp.int32), 0, garrison)
        rec = None
        if record:
            lpt = jax.nn.log_softmax(tgt, -1)[ar, tid]
            lpf = tb.cg_logp(f, mu, sigma)
            logp = jnp.sum(jnp.where(acting, lpt, 0.0)) + jnp.sum(jnp.where(is_real, lpf, 0.0))
            rec = (static, ts, glob, reach_me, m, edge, econ, tid, f, acting, is_real, logp, v, valid)
        return rec, tid, ships_i, is_real

    # current forward for all 4 seats (recorded); valid = this seat is current in this env
    cur_recs = []; cur_tid = []; cur_sh = []; cur_rl = []
    for me in range(N_PLAYERS):
        valid_me = jnp.logical_not(is_league[me])
        rec, t, s, rl = side(me, cur, keys[2 * me], keys[2 * me + 1], valid_me, True)
        cur_recs.append(rec); cur_tid.append(t); cur_sh.append(s); cur_rl.append(rl)

    # opponent forward for non-anchor seats 1,2,3 (action only)
    opp_tid = [None] * N_PLAYERS; opp_sh = [None] * N_PLAYERS; opp_rl = [None] * N_PLAYERS
    base = 2 * N_PLAYERS
    for j, s in enumerate((1, 2, 3)):
        _, t, sh, rl = side(s, opps[j], keys[base + 2 * j], keys[base + 2 * j + 1], False, False)
        opp_tid[s] = t; opp_sh[s] = sh; opp_rl[s] = rl

    # played action per seat: anchor=current; non-anchor = opp where league else current
    played_tid = [cur_tid[0]]; played_sh = [cur_sh[0]]; played_rl = [cur_rl[0]]
    for me in (1, 2, 3):
        lg = is_league[me]
        played_tid.append(jnp.where(lg, opp_tid[me], cur_tid[me]))
        played_sh.append(jnp.where(lg, opp_sh[me], cur_sh[me]))
        played_rl.append(jnp.where(lg, opp_rl[me], cur_rl[me]))

    owner = st.p_owner
    tid = ar
    ships_sent = jnp.zeros((P,), jnp.int32)
    is_real = jnp.zeros((P,), bool)
    for me in range(N_PLAYERS):                          # merge each seat's decision for ITS planets
        sel = (owner == me)
        tid = jnp.where(sel, played_tid[me], tid)
        ships_sent = jnp.where(sel, played_sh[me], ships_sent)
        is_real = jnp.where(sel, played_rl[me], is_real)
    Rx, ANGx, TURNSx = lead_for_ships(st, ships_sent)    # EXECUTED-count lead (partial != all-in speed)
    angle = ANGx[ar, tid]; turns = TURNSx[ar, tid]
    cand = is_real & (ships_sent > 0) & Rx[ar, tid]
    launch = cand & tb.first_hit_gate(st, tid, angle, ships_sent)
    ships_final = jnp.where(launch, ships_sent, 0)
    return tuple(cur_recs), edge, launch, angle, ships_final, tid, st.step + turns


def make_league_rollout_4p(net, frac, T, pool_size, league_p, margin_lam=0.0, margin_D=300.0):
    """Persistent 4p league rollout. Returns:
      recs    tuple of N_PLAYERS per-seat record-stacks (T,n,...) [14 fields, incl `valid`]
      dones   (T,n)
      rews    (T,n,N_PLAYERS)  FFA per-seat reward (terminal_rewards; lam=0 => pure +-1)
      fvs     tuple of N_PLAYERS bootstrap V(s_T) (n,) from the CURRENT policy
      states_f, stats (summed over T):
        [lcnt, fcnt, etsum, ngames, cur_firsts, cur_games, n_cur(sum->n_cur*T = trSamp/update),
         lg_games[1..3], lg_firsts[1..3], lg_ranknum[1..3]]   (7 + 9 = 16)
      cur_firsts/cur_games = CURRENT strict-1st over ALL current-seat participations (-> c1st, seat-unbiased,
      counts non-1st too). lg_firsts -> member ema_first (PFSP); lg_ranknum (frac current out-ranks member) -> ema_rank (admit).
      where for non-anchor seat s: lg_games = league game-ends, lg_firsts = league seat won 1st,
      lg_ahead = league game-ends where anchor(seat0) strictly out-ranks seat s (ships0 > ships_s)."""
    def rollout(cur, o1, o2, o3, states, rng, pool, league_active):
        n = states.p_owner.shape[0]
        rng, klg = jax.random.split(rng)
        # per-update STATIC league assignment: anchor seat 0 always current; seats 1,2,3 ~ Bernoulli(league_p)
        lg = jax.random.bernoulli(klg, league_p, (n, N_PLAYERS))
        lg = lg.at[:, 0].set(False)
        lg = lg & league_active        # empty pool (u<first-admit u1000) -> all-current self-play (match 2p)
        opps = (o1, o2, o3)

        def one_step(carry, _):
            states, rng = carry
            rng, sub, ridx = jax.random.split(rng, 3)
            keys = jax.random.split(sub, n)
            recs, _edge, launch, angle, ships, target, arrival = jax.vmap(
                lambda s, k, il: per_env_actions_league_4p(net, frac, cur, opps, il, s, k))(states, keys, lg)
            states2 = jax.vmap(env_step)(states, launch, angle, ships, target, arrival)
            done = jax.vmap(is_done)(states2)                                        # (n,)
            rew = jax.vmap(lambda s: terminal_rewards(s, lam=margin_lam, D=margin_D))(states2)   # (n,4)
            rew = rew * done[:, None].astype(jnp.float32)
            # ---- league bookkeeping (only at game-ends) ----
            tots = jax.vmap(ship_totals)(states2)                                    # (n,4)
            mx = jnp.max(tots, axis=1, keepdims=True)
            strict = (tots == mx).sum(1) == 1                                        # unique top
            first_oh = ((tots == mx) & strict[:, None]).astype(jnp.float32)          # (n,4) strict 1st
            d = done.astype(jnp.float32)                                             # (n,)
            cur_mask = (~lg).astype(jnp.float32)                                     # (n,4) CURRENT (trained) seats (seat0 always)
            dl = d[:, None] * lg.astype(jnp.float32)                                 # (n,4) league game-ends/seat
            lg_games = dl.sum(0)                                                     # (4,) member participations
            lg_firsts = (dl * first_oh).sum(0)                                       # (4,) member got strict 1st (-> ema_first / PFSP)
            # ema_rank = "CURRENT out-ranks member m": frac of CURRENT seats c with ships_c>ships_m, over ALL
            # current seats (user 2026-06-19: not anchor-only). beat[n,c,m] = seat c out-ranks seat m.
            beat = (tots[:, :, None] > tots[:, None, :]).astype(jnp.float32)         # (n,c,m)
            cur_beat = (beat * cur_mask[:, :, None]).sum(1)                          # (n,m) # current seats beating m
            ncur_env = jnp.clip(cur_mask.sum(1, keepdims=True), 1.0, None)           # (n,1)
            lg_ahead = (dl * (cur_beat / ncur_env)).sum(0)                           # (4,) sum frac-current-beating-m over m's league games
            # CURRENT 1st-place over ALL current-seat participations (1st AND non-1st), NOT anchor-only:
            cur_firsts = (d * (first_oh * cur_mask).sum(1)).sum()                    # games a current seat won strict 1st
            cur_games = (d * cur_mask.sum(1)).sum()                                  # total current-seat participations
            ngames = d.sum()
            lcnt = launch.sum().astype(jnp.float32)
            fcnt = (launch & (ships == states.p_ships)).sum().astype(jnp.float32)
            etsum = jnp.where(done, states2.step, 0).sum().astype(jnp.float32)
            # reset finished envs from the board pool
            idx = jax.random.randint(ridx, (n,), 0, pool_size)
            reset = jax.tree_util.tree_map(lambda p: p[idx], pool)
            states_next = jax.tree_util.tree_map(
                lambda a, b: jnp.where(done.reshape((n,) + (1,) * (a.ndim - 1)), b, a), states2, reset)
            n_cur = (~lg).sum().astype(jnp.float32)                                  # current seat-envs (const/T -> trSamp)
            fo, fm = states2.f_owner, states2.f_mask                                 # (n,F) owner, in-flight mask
            seat_fl = jnp.stack([((fo == k) & fm).sum(1) for k in range(N_PLAYERS)], 1).astype(jnp.float32)  # (n,4) per-seat in-flight count
            mxfl = (seat_fl * cur_mask).max().astype(jnp.float32)                    # peak SINGLE CURRENT-player in-flight fleets this step (<=128 per-player cap; user: current only, not the all-player sum)
            stats = jnp.concatenate([
                jnp.stack([lcnt, fcnt, etsum, ngames, cur_firsts, cur_games, n_cur]),
                lg_games[1:], lg_firsts[1:], lg_ahead[1:], mxfl[None]])              # 7 + 9 + 1(MAX-reduced, idx16) = 17
            return (states_next, rng), (recs, done, rew, stats)

        (states_f, _), (recs, dones, rews, stats) = jax.lax.scan(one_step, (states, rng), None, length=T)
        fvs = tuple(jax.vmap(lambda s: tb._value(net, cur, s, me))(states_f) for me in range(N_PLAYERS))
        return recs, dones, rews, fvs, states_f, jnp.concatenate([stats[:, :16].sum(0), stats[:, 16:].max(0)])  # sum[0:16], MAX mxfl[16]
    return rollout


def make_eval_rollout_4p(net, frac, eval_T):
    """ADMISSION EVAL (user 2026-06-19): candidate @ seat0 vs 3 opponents @ seats 1,2,3 (lg forced
    [F,T,T,T] -> seat0 plays `cand`, seats 1-3 play opps). Runs eval_T steps (full games, reset+continue),
    returns the candidate's strict-1st-place rate over the completed games. Reuses per_env_actions_league_4p
    but DISCARDS recs (no T-scaled memory -> safe at eval_T=500)."""
    def eval_rollout(cand, o1, o2, o3, states, rng, pool):
        n = states.p_owner.shape[0]
        opps = (o1, o2, o3)
        lg = jnp.zeros((n, N_PLAYERS), bool).at[:, 1:].set(True)   # seat0 current=cand; seats 1,2,3 league=opps
        pool_size = pool.p_owner.shape[0]

        def one_step(carry, _):
            states, rng, firsts, ngames = carry
            rng, sub, ridx = jax.random.split(rng, 3)
            keys = jax.random.split(sub, n)
            _recs, _edge, launch, angle, ships, target, arrival = jax.vmap(
                lambda s, k, il: per_env_actions_league_4p(net, frac, cand, opps, il, s, k))(states, keys, lg)
            states2 = jax.vmap(env_step)(states, launch, angle, ships, target, arrival)
            done = jax.vmap(is_done)(states2)
            tots = jax.vmap(ship_totals)(states2)                                    # (n,4)
            mx = jnp.max(tots, axis=1)
            strict = (tots == mx[:, None]).sum(1) == 1
            cand_first = done & strict & (tots[:, 0] == mx)                           # seat0 (cand) is strict 1st
            firsts = firsts + cand_first.sum()
            ngames = ngames + done.sum()
            idx = jax.random.randint(ridx, (n,), 0, pool_size)
            reset = jax.tree_util.tree_map(lambda p: p[idx], pool)
            states_next = jax.tree_util.tree_map(
                lambda a, b: jnp.where(done.reshape((n,) + (1,) * (a.ndim - 1)), b, a), states2, reset)
            return (states_next, rng, firsts, ngames), None

        (sf, _, firsts, ngames), _ = jax.lax.scan(
            one_step, (states, rng, jnp.int32(0), jnp.int32(0)), None, length=eval_T)
        return firsts.astype(jnp.float32) / jnp.clip(ngames.astype(jnp.float32), 1.0), ngames
    return eval_rollout


class LeaguePool4p:
    """ONE shared pool of frozen self-checkpoints with TWO per-member EMAs (exp24-4p, user 2026-06-18):
      ema_first[s] = member's strict-1st-place rate          -> PFSP sampling P ~ ema_first^pfsp_p + floor
      ema_rank[s]  = "current(anchor) out-ranks member" rate -> ADMISSION gate (when s is the reference).
    Admission == 2p's winrate-gated incremental (mastered the reference), recast for FFA via pairwise
    RANK: admit the latest save-grid ckpt when ema_rank[ref] >= admit_thresh after admit_min_games, OR
    force-admit every max_admit_interval. Pool full -> FIFO-evict oldest. u < min_admit_u never admitted."""

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
        self.slot_update = [None] * max_slots             # member ckpt update int, or None = inactive
        self.ema_first = np.full(max_slots, 0.25, np.float64)   # 1st-place rate (4-equal FFA baseline)
        self.ema_rank = np.full(max_slots, 0.5, np.float64)     # current-out-ranks-member rate
        self.games = np.zeros(max_slots, np.int64)
        self.ref_slot = None
        self.last_admit_u = None
        self.last_admit_at = 0
        self.stack = jax.tree_util.tree_map(lambda p: jnp.stack([p] * max_slots), template_params)

    def seed_il_base(self):
        """Admit the IL warm-start base (already pre-filled into the stack as template) as slot 0 at
        u=0 -> the FIRST reference. Subsequent grid snapshots are admitted only once current MASTERS it."""
        self.slot_update[0] = 0
        self.ema_first[0] = 0.25
        self.ema_rank[0] = 0.5
        self.games[0] = 0
        self.ref_slot, self.last_admit_u, self.last_admit_at = 0, 0, 0
        print("LEAGUE SEED u0 (IL warm-start base) -> pool(1): u0", flush=True)

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
        self.stack = jax.tree_util.tree_map(lambda s, p: s.at[i].set(p), self.stack, t)
        self.slot_update[i] = grid_u
        self.ema_first[i] = 0.25                          # optimistic baseline for a fresh member
        self.ema_rank[i] = 0.5                            # current has not yet mastered this new reference
        self.games[i] = 0
        self.ref_slot, self.last_admit_u, self.last_admit_at = i, grid_u, cur_u
        act = [u for u in self.slot_update if u is not None]
        print(f"LEAGUE ADMIT u{grid_u} (reason={reason}, ref<-u{grid_u}) -> pool({len(act)}): "
              + " ".join(f"u{u}" for u in sorted(act)), flush=True)
        return True

    def maybe_admit(self, cur_u):
        grid = self._latest_grid_u(cur_u)
        if grid is None or (self.last_admit_u is not None and grid <= self.last_admit_u):
            return False
        if self.ref_slot is None:                         # FIRST admission seeds the reference
            return self._admit(grid, cur_u, "seed")
        ref_ready = self.games[self.ref_slot] >= self.admit_min_games
        if ref_ready and self.ema_rank[self.ref_slot] >= self.admit_thresh:
            return self._admit(grid, cur_u, f"rank{self.ema_rank[self.ref_slot]:.2f}")
        if (cur_u - self.last_admit_at) >= self.max_admit_interval:
            return self._admit(grid, cur_u, f"interval{cur_u - self.last_admit_at}")
        return False

    def n_active(self):
        return sum(1 for x in self.slot_update if x is not None)

    def probs(self):
        """PFSP over active slots by ema_first (1st-place rate): P ~ ema_first^pfsp_p + uniform floor."""
        act = [i for i, u in enumerate(self.slot_update) if u is not None]
        p = np.zeros(self.max_slots, np.float64)
        if not act:
            return p
        w = np.maximum(self.ema_first[act], 1e-6) ** self.pfsp_p
        w = w / w.sum() if w.sum() > 1e-12 else np.full(len(act), 1.0 / len(act))
        p[act] = (1.0 - self.floor) * w + self.floor / len(act)
        return p / p.sum()

    def sample(self, k):
        """k PFSP draws (with replacement) -> pool slot indices for the k non-anchor seats."""
        p = self.probs()
        if p.sum() <= 0:                                  # no active member (should not happen post-seed)
            act = [i for i, u in enumerate(self.slot_update) if u is not None] or [0]
            return self.np_rng.choice(act, size=k).tolist()
        return self.np_rng.choice(self.max_slots, size=k, p=p).tolist()

    def gather(self, slot):
        return jax.tree_util.tree_map(lambda s: s[slot], self.stack)

    def top_k_slots(self, k):
        """The k admission-eval opponents (user 2026-06-19):
          >= k active members -> the k HIGHEST rolling-1st-rate (ema_first), descending (exactly k -> those k);
          <  k active members -> RANDOM sample k WITH REPLACEMENT from the available (1 member -> [m,m,m];
                                 2 members -> random 111/112/122/222/...)."""
        act = [i for i, u in enumerate(self.slot_update) if u is not None]
        if not act:
            return []
        if len(act) >= k:
            return sorted(act, key=lambda i: self.ema_first[i], reverse=True)[:k]
        return [int(s) for s in self.np_rng.choice(act, size=k, replace=True)]

    def admit_params(self, params, cur_u, admwr):
        """Admit the CANDIDATE's own params (the snapshot the admission eval just measured) as a new pool
        member (FIFO-evict oldest when full). New member = the new reference for refR tracking."""
        free = [i for i, su in enumerate(self.slot_update) if su is None]
        i = free[0] if free else min(range(self.max_slots), key=lambda j: self.slot_update[j])
        self.stack = jax.tree_util.tree_map(lambda s, p: s.at[i].set(p), self.stack, params)
        self.slot_update[i] = cur_u
        self.ema_first[i] = 0.25
        self.ema_rank[i] = 0.5
        self.games[i] = 0
        self.ref_slot, self.last_admit_u, self.last_admit_at = i, cur_u, cur_u
        act = [u for u in self.slot_update if u is not None]
        return i, act

    def update(self, opp_slots, lg_games, lg_firsts, lg_ahead, alpha):
        """opp_slots = [slot1, slot2, slot3] (pool slot for non-anchor seats 1,2,3). lg_* are length-3
        (seats 1,2,3). Aggregate per slot (a slot can occupy >1 seat), then EMA-update both winrates."""
        agg = {}
        for j, slot in enumerate(opp_slots):
            g, fr, ah = float(lg_games[j]), float(lg_firsts[j]), float(lg_ahead[j])
            if g <= 0:
                continue
            a = agg.setdefault(int(slot), [0.0, 0.0, 0.0])
            a[0] += g; a[1] += fr; a[2] += ah
        for slot, (g, fr, ah) in agg.items():
            if self.slot_update[slot] is None:
                continue
            keep = (1.0 - alpha) ** g
            self.ema_first[slot] = keep * self.ema_first[slot] + (1.0 - keep) * (fr / g)
            self.ema_rank[slot] = keep * self.ema_rank[slot] + (1.0 - keep) * (ah / g)
            self.games[slot] += int(g)

    def stats_line(self):
        return ";".join(f"u{u}:f{self.ema_first[i]:.3f}:r{self.ema_rank[i]:.3f}:{self.games[i]}"
                        for i, u in enumerate(self.slot_update) if u is not None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_envs", type=int, default=512)       # 4p (user 2026-06-18); NOTE: 4-seat rollout buffer
    #   (per-seat ts/edge recs) ~40+GB at n=512 -> needs a big-VRAM GPU (H100/H200); 32GB OOMs (probe first)
    ap.add_argument("--league_p", type=float, default=0.2)    # per non-anchor seat: P(league) (current = 1-this)
    ap.add_argument("--board_pool_path", default=DEFAULT_BOARD_POOL_4P)
    ap.add_argument("--board_pool_n", type=int, default=0)
    ap.add_argument("--T", type=int, default=128)
    ap.add_argument("--updates", type=int, default=550000)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)        # align 2p exp25 (RL fine-tune lr, < IL's 3e-4)
    ap.add_argument("--gamma", type=float, default=0.999)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--ent", type=float, default=0.02)       # align 2p exp25
    ap.add_argument("--vf", type=float, default=0.5)
    ap.add_argument("--margin_lam", type=float, default=0.0)  # 0 = pure FFA win/loss; >0 = tanh ship-diff bonus
    ap.add_argument("--margin_D", type=float, default=300.0)
    ap.add_argument("--E", type=int, default=128)        # exp25 v41: E=128 (match IL warm-start ckpt)
    ap.add_argument("--n_layers", type=int, default=6)   # 6-layer trunk
    ap.add_argument("--n_heads", type=int, default=4)    # 4 heads -> d=32
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save_every", type=int, default=200)   # 2p exp25 real run value
    ap.add_argument("--save_dir", type=str, default=os.path.join(_HERE, "checkpoints_sp"))
    ap.add_argument("--init_from", type=str, default="")     # best IL ckpt (warm-start) or RL resume
    ap.add_argument("--start_update", type=int, default=0)
    ap.add_argument("--num_minibatches", type=int, default=64)   # user 2026-06-18: mb=N/64 (n512/4seat/T128 -> 4096)
    ap.add_argument("--lr_schedule", type=str, default="cosine")   # align 2p exp25 (cosine anneal -> lr_min, hold)
    ap.add_argument("--warmup_updates", type=int, default=1000)
    ap.add_argument("--lr_min", type=float, default=3e-5)           # cosine floor (align 2p exp25)
    ap.add_argument("--cosine_updates", type=int, default=100000)   # cosine anneal horizon (EXPECTED real run len, not --updates max)
    # --- league pool ---
    ap.add_argument("--max_slots", type=int, default=30)     # pool capacity (FIFO-evict oldest)
    ap.add_argument("--pfsp_p", type=float, default=1.0)     # LINEAR: P ~ ema_first^1 + floor (matches 2p)
    ap.add_argument("--pfsp_floor", type=float, default=0.02)
    ap.add_argument("--ema_alpha", type=float, default=0.02)
    ap.add_argument("--admit_thresh", type=float, default=0.70)  # admit when current out-ranks reference >= this
    ap.add_argument("--max_admit_interval", type=int, default=5000)    # 2p exp25 real run value (force-admit on plateau)
    ap.add_argument("--admit_min_games", type=int, default=1024)       # align 2p exp25
    ap.add_argument("--min_admit_u", type=int, default=1000)
    # NEW admission (user 2026-06-19): after each ckpt, run --eval_envs controlled games (candidate @ seat0 vs the
    # 3 highest rolling-1st-rate pool members @ seats 1-3); admit the candidate if its 1st-rate > --admit_winrate.
    ap.add_argument("--admit_winrate", type=float, default=0.35)
    ap.add_argument("--first_league_u", type=int, default=200)   # SEED the first league member = current snapshot at
    #                                                              this update (pool empty before -> pure self-play)
    ap.add_argument("--eval_T", type=int, default=500)           # admission-eval rollout length (full 4p games)
    ap.add_argument("--eval_envs", type=int, default=512)        # admission-eval game count (boards from the pool)
    ap.add_argument("--league_stats_every", type=int, default=20)
    args = ap.parse_args()
    print("DEVICES:", jax.devices(), flush=True)

    net = OrbitNet19(E=args.E, n_layers=args.n_layers, n_heads=args.n_heads)
    frac = SimpleFracMLP(E=args.E, n_heads=args.n_heads)   # v41: [emb || emb_tid || gemb] -> (mu,sigma)
    if args.board_pool_path and os.path.exists(args.board_pool_path):
        pool = load_board_pool(args.board_pool_path, args.board_pool_n)
        print(f"BOARD_POOL loaded {pool.p_owner.shape[0]} boards from {args.board_pool_path}", flush=True)
    else:
        if args.board_pool_path:
            print(f"BOARD_POOL '{args.board_pool_path}' not found -> gen_init_states(reset(4))", flush=True)
        pool = gen_init_states(max(256, args.n_envs), args.seed)
    pool_size = pool.p_owner.shape[0]
    states = jax.tree_util.tree_map(lambda p: p[:args.n_envs], pool)
    P = pool.p_owner.shape[1]
    rng = jax.random.PRNGKey(args.seed)
    rng, ki, ki2 = jax.random.split(rng, 3)

    # econ-CNN init (7-arg net) — identical v41 seeding to train_il_v5 -> IL warm-start compatible
    single0 = jax.tree_util.tree_map(lambda x: x[0], pool)
    fc0 = _forecast(single0)
    R0, A0, T0, Rg0 = reach_solve_static(single0)
    st0, ts0, gl0, m0, econ0 = basic_features(single0, 0, fc=fc0)
    reach0 = Rg0 & ((single0.p_owner == 0) & single0.p_mask)[:, None]
    _, _, _, edge0 = tb.edge_features(single0, fc=fc0, lead=(R0, A0, T0))
    net_params = net.init(ki, st0, ts0, gl0, reach0, m0, edge0, econ0)
    ar0 = jnp.arange(P)
    tgt0, emb0, gemb0, _b0, _v0 = net.apply(net_params, st0, ts0, gl0, reach0, m0, edge0, econ0)   # v41: slot3=gemb
    tid0 = jnp.argmax(tgt0, -1)
    # v41: frac = SimpleFracMLP([emb[s] || emb_tid || gemb]); NO edge.
    frac_params = frac.init(ki2, emb0, emb0[tid0], gemb0)
    params = {'net': net_params, 'frac': frac_params}
    if args.init_from:
        with open(args.init_from, "rb") as _fh:
            params = jax.tree_util.tree_map(jnp.asarray, fser.msgpack_restore(_fh.read()))
        print(f"WARM-START / RESUME init_from={args.init_from} start_update={args.start_update}", flush=True)
    n_params = sum(int(np.prod(p.shape)) for p in jax.tree_util.tree_leaves(params))
    print(f"MODEL_PARAMS {n_params}  (exp24-4p v37 econ-CNN 2-tree {{net,frac}}, 4p FFA LEAGUE: anchor "
          f"seat0 + 3 seats x {1.0-args.league_p:.0%}/{args.league_p:.0%} current/league, 3 indep PFSP "
          f"members/update, E={args.E} L={args.n_layers} H={args.n_heads} P={P}, gamma={args.gamma} "
          f"lam={args.lam}, NO-RESIGN)", flush=True)

    np_rng = np.random.default_rng(args.seed + 777)
    lp = LeaguePool4p(args.save_dir, args.max_slots, params, args.pfsp_p, args.pfsp_floor, np_rng,
                      admit_thresh=args.admit_thresh, max_admit_interval=args.max_admit_interval,
                      admit_min_games=args.admit_min_games, save_every=args.save_every,
                      min_admit_u=args.min_admit_u)
    # Pool starts EMPTY (user 2026-06-19): the FIRST league member is SEEDED with the current snapshot at
    # u == first_league_u (default 200), NOT at u0. Before that -> pure self-play (league_active gate off).
    # Subsequent members enter via the 512-game admission eval (cand vs top-3, 1st-rate > admit_winrate).

    total_opt = max(1, args.updates * args.epochs * args.num_minibatches)
    _off = args.start_update * args.epochs * args.num_minibatches
    if args.lr_schedule == "warmflat":
        _ws = max(1, args.warmup_updates * args.epochs * args.num_minibatches)
        lr_sched = lambda s: args.lr * jnp.minimum((s + _off + 1.0) / _ws, 1.0)
    elif args.lr_schedule == "cosine":
        # align 2p exp25: cosine anneal lr -> lr_min over cosine_updates (EXPECTED real horizon, NOT
        # --updates max), then HOLD at lr_min. NO warmup (starts at peak lr); clamp past horizon to floor.
        _cT = max(1, args.cosine_updates * args.epochs * args.num_minibatches)
        lr_sched = lambda s: args.lr_min + 0.5 * (args.lr - args.lr_min) * (1.0 + jnp.cos(jnp.pi * jnp.minimum((s + _off) / _cT, 1.0)))
    else:
        lr_sched = args.lr
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(lr_sched))
    opt_state = opt.init(params)

    rollout = make_league_rollout_4p(net, frac, args.T, pool_size, args.league_p, args.margin_lam, args.margin_D)
    rollout_jit = jax.jit(rollout)
    eval_rollout = make_eval_rollout_4p(net, frac, args.eval_T)   # 512-game admission eval (candidate vs top-3)
    eval_rollout_jit = jax.jit(eval_rollout)
    eval_states = jax.tree_util.tree_map(lambda p: p[:args.eval_envs], pool)   # fixed boards -> comparable admWR across ckpts
    ar = jnp.arange(P)

    def save_ckpt(u):
        os.makedirs(args.save_dir, exist_ok=True)
        with open(os.path.join(args.save_dir, f"ckpt_u{u:05d}.msgpack"), "wb") as fh:
            fh.write(fser.msgpack_serialize(jax.device_get(params)))
        json.dump({"update": int(u), "n_params": int(n_params), "E": args.E, "n_layers": args.n_layers,
                   "n_heads": args.n_heads, "P": int(P), "n_players": int(N_PLAYERS),
                   "arch": "exp24-4p v37 econ-CNN pointer self=hold + 6-d ALL-IN clipped-Gaussian frac (2-tree)",
                   "trainer": "train_league_4p", "canonical": "C4-rotation", "resign": False,
                   "reward": f"FFA per-seat (margin_lam={args.margin_lam})", "league_p": args.league_p,
                   "param_trees": ["net", "frac"], "args": vars(args)},
                  open(os.path.join(args.save_dir, "meta.json"), "w"), default=str)

    def loss_fn(params, batch):                          # econ-CNN 2-tree PPO + per-(env,seat) valid mask
        (static, ts, glob, reach, mask, edge, econ, tid, f, acting, is_real,
         logp_old, adv, ret, valid) = batch

        def one(s_, t_, gl_, r_, m_, ed_, ec_, ti, fi, ac, rl):
            tgt, emb, gemb, _b, v = net.apply(params['net'], s_, t_, gl_, r_, m_, ed_, ec_)   # v41: slot3=gemb
            emb_tid = emb[ti]
            mu, sigma = frac.apply(params['frac'], emb, emb_tid, gemb)   # v41 SimpleFracMLP: [emb||emb_tid||gemb]
            lpt = jax.nn.log_softmax(tgt, -1)[ar, ti]
            lpf = tb.cg_logp(fi, mu, sigma)
            lp_ = jnp.sum(jnp.where(ac, lpt, 0.0)) + jnp.sum(jnp.where(rl, lpf, 0.0))
            p_t = jax.nn.softmax(tgt, -1)
            ent_ptr = -jnp.sum(jnp.where(ac, jnp.sum(p_t * jax.nn.log_softmax(tgt, -1), -1), 0.0)) / jnp.clip(jnp.sum(ac), 1.0)
            ent_frac = jnp.sum(jnp.where(rl, tb.cg_entropy(mu, sigma), 0.0)) / jnp.clip(jnp.sum(rl), 1.0)
            _rlc = jnp.clip(jnp.sum(rl), 1.0)                                  # sigma stats over real-launch rows
            msig = jnp.sum(jnp.where(rl, sigma, 0.0)) / _rlc                   # per-env mean sigma
            msig_sq = jnp.sum(jnp.where(rl, sigma * sigma, 0.0)) / _rlc        # per-env mean sigma^2
            return lp_, v, ent_ptr, ent_frac, msig, msig_sq

        lp, v, ent_ptr, ent_frac, msig, msig_sq = jax.vmap(one)(static, ts, glob, reach, mask, edge, econ, tid, f, acting, is_real)
        ent = ent_ptr + ent_frac
        vd = valid.astype(jnp.float32)
        denom = jnp.clip(vd.sum(), 1.0)
        ratio = jnp.exp(lp - logp_old)
        adv_mean = (adv * vd).sum() / denom
        adv_var = ((adv - adv_mean) ** 2 * vd).sum() / denom
        adv_n = (adv - adv_mean) / (jnp.sqrt(adv_var) + 1e-8)
        pg = -(jnp.minimum(ratio * adv_n, jnp.clip(ratio, 1 - args.clip, 1 + args.clip) * adv_n) * vd).sum() / denom
        vloss = (((v - ret) ** 2) * vd).sum() / denom
        entropy = (ent * vd).sum() / denom
        loss = pg + args.vf * vloss - args.ent * entropy
        ret_mean = (ret * vd).sum() / denom
        ss_res = (((ret - v) ** 2) * vd).sum() / denom
        ss_tot = (((ret - ret_mean) ** 2) * vd).sum() / denom
        ev = 1.0 - ss_res / jnp.clip(ss_tot, 1e-8)
        kl = ((logp_old - lp) * vd).sum() / denom
        clipfrac = ((jnp.abs(ratio - 1.0) > args.clip).astype(jnp.float32) * vd).sum() / denom
        ent_ptr_m = (ent_ptr * vd).sum() / denom                             # split-entropy logging (ent = entP + entF)
        ent_frac_m = (ent_frac * vd).sum() / denom
        mean_sig = (msig * vd).sum() / denom                                 # avg sigma over (valid) batch
        std_sig = jnp.sqrt(jnp.clip((msig_sq * vd).sum() / denom - mean_sig ** 2, 0.0))   # sigma-of-sigma
        return loss, (pg, vloss, entropy, ev, kl, clipfrac, ent_ptr_m, ent_frac_m, mean_sig, std_sig)

    @jax.jit
    def update(params, opt_state, batch):
        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, batch)
        upd, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, upd)
        return params, opt_state, loss, aux

    @jax.jit
    def run_epoch(params, opt_state, bs, perm):
        """One epoch's whole minibatch loop fused into a jit'd lax.scan (seat-block gather + update).
        bs = tuple of N_PLAYERS per-seat batches (each a list of fields). idx//Ns picks the seat block,
        idx%Ns the in-block row."""
        NP = len(bs)
        Ns = bs[0][0].shape[0]; mb = max(1, (NP * Ns) // args.num_minibatches)   # max(1,..) floor (defensive; v12 parity)
        idxs = perm[:args.num_minibatches * mb].reshape(args.num_minibatches, mb)

        def step(carry, idx):
            p, ostate = carry
            seat = idx // Ns
            loc = idx % Ns

            def sel_field(fi):
                out = bs[NP - 1][fi][loc]
                for me in range(NP - 1):
                    sm = (seat == me).reshape((seat.shape[0],) + (1,) * (out.ndim - 1))
                    out = jnp.where(sm, bs[me][fi][loc], out)
                return out
            mbatch = tuple(sel_field(fi) for fi in range(len(bs[0])))
            p, ostate, loss, aux = update(p, ostate, mbatch)
            return (p, ostate), jnp.stack((loss,) + tuple(aux))
        (params, opt_state), accs = jax.lax.scan(step, (params, opt_state), idxs)
        return params, opt_state, accs

    os.makedirs(args.save_dir, exist_ok=True)
    # exp25 logging (user 2026-06-19): TRAIN log -> stdout (train_sp.log); LEAGUE log -> SEPARATE sp_league.log.
    lwf = open(os.path.join(args.save_dir, "sp_league.log"), "a")

    _tn = args.T * args.n_envs                            # ENV-TURNS/update = games x T
    _gen = N_PLAYERS * args.T * args.n_envs
    _gs_exp = int((1.0 + 3.0 * (1.0 - args.league_p)) * args.T * args.n_envs)   # EXPECTED training samples (actual varies)
    _mbsize = max(1, (N_PLAYERS * args.T * args.n_envs) // args.num_minibatches)
    # cleaned cols (user 2026-06-19): dropped genSmp/entP/entF/eval/pWR/pool. admWR = candidate 1st-rate vs top-3
    # (512-game admission eval, per ckpt); refR = current rank-wr vs last-admitted ckpt; c1st = current 1st-rate (train).
    _COLS = [("upd", 6), ("trSamp", 8), ("SPS", 7), ("effSPS", 7), ("gen", 5), ("tot", 5), ("elapsed", 9),
             ("mem", 5), ("done", 5), ("fsend", 6), ("lpt", 6), ("eturn", 6), ("loss", 8), ("pg", 9),
             ("vloss", 7), ("ent", 6), ("mSig", 6), ("sSig", 6), ("EV", 6), ("KL", 9), ("clipf", 6),
             ("c1st", 5), ("admWR", 6), ("refR", 5), ("mxfl", 6)]
    _HDR = " | ".join(f"{c:>{w}}" for c, w in _COLS)
    def _h(x):
        return f"{x/1e9:.2f}B" if x >= 1e9 else (f"{x/1e6:.2f}M" if x >= 1e6 else f"{x/1e3:.0f}K")
    def _hms(s):
        s = int(s); return f"{s//3600}:{(s % 3600)//60:02d}:{s % 60:02d}"
    print(f"ROLLOUT: T={args.T} x n_envs={args.n_envs} = {_tn:,} ENV-TURNS/update; all {N_PLAYERS} seats "
          f"forward-passed+recorded (league seats masked in loss). anchor seat0 always current; seats 1-3 each "
          f"{args.league_p:.0%} league (3 indep PFSP/update). NO-RESIGN.", flush=True)
    print(f"TRAIN: 'trSamp'=cumulative TRAINING samples (ACTUAL current seats). per update: train~={_gs_exp:,} "
          f"(=(1+3x(1-league_p)) x T x n_envs) | num_minibatches={args.num_minibatches} | mb_size={_mbsize:,} "
          f"| epochs={args.epochs} | lr-sched={args.lr_schedule}. ADMISSION: every ckpt, {args.eval_envs}-game eval "
          f"(cand vs top-3 by rolling 1st-rate); admit if cand 1st-rate > {args.admit_winrate}.", flush=True)
    print(_HDR, flush=True)

    _cum = args.start_update * _gs_exp                    # cumulative TRAINING samples (actual current transitions)
    _cum_gen = args.start_update * _gen                   # cumulative GENERATED samples
    _t_start = time.time()
    admwr = float("nan")                                  # last admission-eval 1st-rate (cand vs top-3), per ckpt
    for u in range(args.start_update, args.updates):
        t0 = time.time()
        rng, kr = jax.random.split(rng)
        opp_slots = lp.sample(3)                          # 3 independent PFSP draws (shared pool)
        o1, o2, o3 = (lp.gather(s) for s in opp_slots)
        league_active = jnp.asarray(lp.n_active() > 0)    # traced (no recompile on empty->active flip @ u1000)
        recs, dones, rews, fvs, states, rstats = rollout_jit(params, o1, o2, o3, states, kr, pool, league_active)
        jax.block_until_ready(rews)
        gen_t = time.time() - t0
        Tn, nl0 = recs[0][0].shape[0], recs[0][0].shape[1]
        flat = lambda x: x.reshape((Tn * nl0,) + x.shape[2:])

        def make_batch(rec, rew_side, fv):
            (static, ts, glob, reach, mask, edge, econ, tid, f, acting, is_real, logp, value, valid) = rec
            adv, ret = tb.gae(value, fv, rew_side, dones, args.gamma, args.lam)
            return [flat(static), flat(ts), flat(glob), flat(reach), flat(mask), flat(edge), flat(econ),
                    flat(tid), flat(f), flat(acting), flat(is_real), flat(logp), flat(adv), flat(ret), flat(valid)]

        bs = tuple(make_batch(recs[me], rews[..., me], fvs[me]) for me in range(N_PLAYERS))
        Ns = bs[0][0].shape[0]; N = N_PLAYERS * Ns; accs = []
        for _ in range(args.epochs):
            rng, kp = jax.random.split(rng)
            perm = jax.random.permutation(kp, N)
            params, opt_state, accs_e = run_epoch(params, opt_state, bs, perm)
            accs.append(accs_e)
        (loss, pg, vloss, entropy, ev, kl, clipfrac, ent_ptr_m, ent_frac_m, mean_sig, std_sig) = np.mean(
            np.asarray(jax.device_get(jnp.concatenate(accs, 0))), axis=0)

        # ---- parse rollout stats + update the league pool EMAs ----
        rs = np.asarray(jax.device_get(rstats))
        lcnt, fcnt, etsum, ngames, cur_firsts, cur_games, n_cur_T = rs[0], rs[1], rs[2], rs[3], rs[4], rs[5], rs[6]
        lg_games, lg_firsts, lg_ranknum = rs[7:10], rs[10:13], rs[13:16]   # seats 1,2,3
        mxfl = rs[16]                                                       # peak SINGLE CURRENT-player in-flight fleets (<=128 cap), MAX over rollout
        lp.update(opp_slots, lg_games, lg_firsts, lg_ranknum, args.ema_alpha)   # rolling EMA every update (lg_ranknum -> ema_rank)

        tot_t = time.time() - t0
        sps = _tn / gen_t; eff_sps = _tn / tot_t
        fsend = fcnt / max(lcnt, 1.0)
        lpt = lcnt / max(_tn, 1.0)        # launches per env-turn (ALL seats) -> 0.00 == nobody launching (collapse signal)
        eturn = etsum / max(ngames, 1.0)
        c1st = cur_firsts / max(cur_games, 1.0)            # CURRENT 1st-place rate over ALL current seats (seat-unbiased; counts non-1st)
        ref_rank = (lp.ema_rank[lp.ref_slot] if lp.ref_slot is not None else float("nan"))
        _act = [i for i, uu in enumerate(lp.slot_update) if uu is not None]
        pool_wr = (float(np.mean([lp.ema_rank[i] for i in _act])) if _act else float("nan"))   # 4p analog of 2p mean_wr
        peak_gb = jax.devices()[0].memory_stats().get('peak_bytes_in_use', 0) / 1e9
        _cum += n_cur_T                                   # ACTUAL current(trained) transitions this update (all current seats)
        _cum_gen += _gen
        last = (u == args.updates - 1)
        evstr = ""                                        # no in-training eval (4p): eval column blank (2p column-parity)
        if (u - args.start_update) % 40 == 0 and u != args.start_update:
            print(_HDR, flush=True)
        print(f"{u:6d} | {_h(_cum):>8} | {sps:7.0f} | {eff_sps:7.0f} | {gen_t:5.2f} | {tot_t:5.2f} | "
              f"{_hms(time.time()-_t_start):>9} | "
              f"{peak_gb:5.1f} | {ngames:5.0f} | {fsend:6.3f} | {lpt:6.2f} | {eturn:6.1f} | {loss:+8.3f} | {pg:+9.5f} | "
              f"{vloss:7.4f} | {entropy:6.3f} | {mean_sig:6.4f} | {std_sig:6.4f} | "
              f"{ev:+6.3f} | {kl:+9.5f} | {clipfrac:6.3f} | "
              f"{c1st:5.2f} | {admwr:6.3f} | {ref_rank:5.2f} | {mxfl:6.0f}", flush=True)
        if args.save_every and (u % args.save_every == 0 or last):
            save_ckpt(u)
            if lp.n_active() == 0:
                # SEED the FIRST league member = the current snapshot at first_league_u (user 2026-06-19).
                if u >= args.first_league_u:
                    lp.admit_params(params, u, float("nan"))
                    lwf.write(f"u{u} SEED first-league-member u{u} -> pool(1)\n"); lwf.flush()
                    print(f"LEAGUE SEED first member u{u} -> pool(1)", flush=True)
            else:
                # admission: eval_envs controlled games (candidate @ seat0 vs top-3 pool members by rolling
                # 1st-rate @ seats 1-3); admit the candidate's own params if its 1st-rate > admit_winrate.
                top3 = lp.top_k_slots(3)
                eo1, eo2, eo3 = (lp.gather(s) for s in top3)
                rng, ke = jax.random.split(rng)
                _wr, _ng = eval_rollout_jit(params, eo1, eo2, eo3, eval_states, ke, pool)
                admwr = float(_wr)
                admitted = admwr > args.admit_winrate
                if admitted:
                    lp.admit_params(params, u, admwr)
                lwf.write(f"u{u} admWR={admwr:.3f} ({int(_ng)}g) vs top3="
                          + ",".join(f"u{lp.slot_update[s]}" for s in top3)
                          + (f" -> ADMIT u{u}" if admitted else " -> no")
                          + f" | pool({lp.n_active()}): {lp.stats_line()}\n")
                lwf.flush()
                if admitted:
                    print(f"LEAGUE ADMIT u{u} (admWR={admwr:.3f} > {args.admit_winrate}) -> pool({lp.n_active()})", flush=True)
    lwf.close()
    print("PIPELINE_OK", flush=True)


if __name__ == "__main__":
    main()
