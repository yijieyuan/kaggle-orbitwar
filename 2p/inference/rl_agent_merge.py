"""2p RL agent — MERGE of two adjacent exp26-simplefrac checkpoints (A=u55000 MAIN, B=u53000),
pure-numpy deploy port of merge_eval2.py's `merged()` (2p/eval_cross/merge_eval2.py lines 327-351).
Rollout horizon is ADAPTIVE to the remaining overage bank (see agent()): >15s->H3, >5s->H2, <=5s->greedy A.

Per turn at seat `me`:
  1. aA = decode(arr, W_A, me) (greedy A); aB = decode(arr, W_B, me) (greedy B).
  2. agree = for every MINE planet, aA & aB match on (launch, target_id, ships).
        agree  -> output aA (DONE, no rollout).
  3. disagree -> H-turn rollout each config, pick by UNIFIED value (mean of A's and B's value head):
        vj = rollout(first=aA, pMe=W_A, pOpp=W_B)   (me plays A, opp plays B)
        vy = rollout(first=aB, pMe=W_B, pOpp=W_A)   (me plays B, opp plays A)
        output aA if vj >= vy else aB.

     rollout(first, pMe, pOpp):
       opp_action0 = decode(arr, pOpp, opp=1-me)
       step engine applying (first @ owner==me, opp_action0 @ owner==opp) -> st1
       then H-1 more steps each applying (decode(st,pMe,me) @ me, decode(st,pOpp,opp) @ opp)
       at final state return 0.5*(value_of(stf, W_A, me) + value_of(stf, W_B, me))   (ALWAYS A&B unified).

The action-merge-by-owner + engine step reuse the search combo's sim_lookahead (forward_sim swept-pair
step + from_kaggle_obs + state_to_obs). Decode/value_of are byte-identical to the greedy exp26 deploy port.

Knobs (env): ORBIT25_2P_WEIGHTS_A (npz, default weights_2p_A_u55000.npz=MAIN),
ORBIT25_2P_WEIGHTS_B (npz, default weights_2p_B_u53000.npz), ORBIT_MERGE_H (deep horizon, 3),
ORBIT_MERGE_H_MID (mid horizon, 2), ORBIT_MERGE_H3_REMAIN (20s), ORBIT_MERGE_H2_REMAIN (10s).
Overtime safety: each turn reads the OFFICIAL 60s bank obs["remainingOverageTime"] AND tallies our own
wall-clock; using the MAX-spent of the two, `remaining = 60 - used` picks the horizon (>15s->H3, >5s->H2,
else greedy A=u55000). A hard floor also forces greedy if used + worst-turn (ORBIT_MERGE_WORST_TURN ~2s,
observed ~0.5s) could pass 59s -> a low bank, a missing/wrong env field, or a single long turn can never TLE.
"""
import os
import sys
import math
import time
import copy
import numpy as np

# UNCONDITIONAL insert (Kaggle pops exec_dir after exec; the `if not in` guard can skip — see memory)
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import rl_infer as R
from engine import AgentState
from sim_lookahead import from_kaggle_obs, OrbitSimulator, state_to_obs

_WA = R.load_weights(os.environ.get("ORBIT25_2P_WEIGHTS_A", os.path.join(_HERE, "weights", "weights_2p_u55000.npz")))  # main
_WB = R.load_weights(os.environ.get("ORBIT25_2P_WEIGHTS_B", os.path.join(_HERE, "weights", "weights_2p_u53000.npz")))
_STATE = AgentState()
_SIM = OrbitSimulator()
_L = 64  # MAX_COMET_PATH

# obs tuple indices
P_ID, P_OWNER, P_X, P_Y, P_R, P_SHIPS, P_PROD = range(7)
F_ID, F_OWNER, F_SHIPS = 0, 1, 6

# ── merge config ──
# ADAPTIVE rollout horizon by remaining 60s overage bank: plenty -> deep, moderate -> shallow,
# low -> greedy. remaining = 60 - used (used = max(60-overage, self-timed), conservative).
_H_DEEP = int(os.environ.get("ORBIT_MERGE_H", "3"))            # H when remaining bank > _H3_REMAIN
_H_MID = int(os.environ.get("ORBIT_MERGE_H_MID", "2"))         # H when remaining bank > _H2_REMAIN
_H3_REMAIN = float(os.environ.get("ORBIT_MERGE_H3_REMAIN", "20"))  # > this s left -> H_DEEP
_H2_REMAIN = float(os.environ.get("ORBIT_MERGE_H2_REMAIN", "10"))   # > this s left -> H_MID ; else greedy A
_GUARD_S = float(os.environ.get("ORBIT_MERGE_GUARD_S", "10"))   # (legacy, unused) env-overage floor
# OVERTIME BANK guard: Kaggle gives 1s/turn + a 60s cumulative overage bank, exposed per-turn as
# obs["remainingOverageTime"] (starts 60, decremented by each turn's seconds over 1s). We STOP the rollout
# and play the MAIN ckpt's greedy (A=u55000) once remainingOverageTime < _BANK_REMAIN (10s left = 50s used).
# Fallback: if the env omits the field, accrue our own wall-clock over 1s into _OVERTIME_USED and stop at _BANK_S.
_BANK_REMAIN = float(os.environ.get("ORBIT_MERGE_BANK_REMAIN", "10"))
_BANK_S = float(os.environ.get("ORBIT_MERGE_BANK_S", "50"))
_WORST_TURN = float(os.environ.get("ORBIT_MERGE_WORST_TURN", "2.0"))  # conservative cap on ONE rollout turn's overage (observed max ~0.5s)
_OVERTIME_USED = 0.0

# Rollout AgentState seeding: a fleet's first-hit is FIXED once launched (only a new comet changes it,
# which update() re-detects), so SEED each rollout AgentState with the live agent's incremental cache
# (_STATE.fleet_hit) -> the persisting in-flight fleets are LOOKED UP (not re-walked); update() then only
# walks the few NEW (rollout-hypothetical) launches. EXACT same result as a fresh re-walk (the cached
# absolute arrival-turn is invariant to the fleet's position), just O(fleets) dict-copy + O(new x walk)
# instead of O(all x walk) -> kills the fleet-count slowdown. Toggle off (=0) to reproduce the old behavior.
_SEED = int(os.environ.get("ORBIT_MERGE_SEEDCACHE", "1"))
# Replay-eval only: skip the per-turn wall-clock accrual into _OVERTIME_USED so the bank-guard is driven
# PURELY by obs["remainingOverageTime"] (the replay's actual Kaggle bank). Lets a local replay reproduce
# the deployed guard timing (the local machine's speed must NOT change which turns roll out). Default 0.
_NOSELFTIME = int(os.environ.get("ORBIT_MERGE_NOSELFTIME", "0"))


def _rs():
    """Rollout AgentState that REUSES the live agent's deterministic caches. A fresh AgentState's first
    update() triggers a full REBUILD (recomputes planet_traj over step..500 ~= 190ms) AND recomputes every
    fleet's first-hit (known_fleets empty). Instead shallow-copy _STATE: planet_traj/game_key/planet_base
    are carried (so update() SLICES, no rebuild), and the MUTABLE fleet/comet caches are copied (so the
    rollout's hypothetical launches don't pollute the live _STATE). update() then only slices + walks the
    few NEW launches. EXACT same result, O(slice + new) instead of O(full rebuild + all fleets)."""
    if not _SEED:
        return AgentState()
    s = copy.copy(_STATE)                       # carries planet_traj/comet_traj (read-only) + game_key/av/base
    s.fleet_hit = dict(_STATE.fleet_hit)        # copy mutable caches: rollout launches must NOT touch _STATE
    s.known_fleets = set(_STATE.known_fleets)
    s.fleet_traj = dict(_STATE.fleet_traj)
    s.comet_traj = dict(_STATE.comet_traj)
    s.known_comets = set(_STATE.known_comets)
    return s


def _get(o, key, default=None):
    return o.get(key, default) if isinstance(o, dict) else getattr(o, key, default)


def _obs_to_arr(obs, state):
    """kaggle obs -> rl_infer arr dict (P = active planets, obs order). f_target/f_arrival
    come from state.fleet_hit (first_hit_from, kind=='planet'). Byte-identical to the greedy agent."""
    planets = list(_get(obs, "planets", []) or [])
    fleets = list(_get(obs, "fleets", []) or [])
    av = float(_get(obs, "angular_velocity", 0.0) or 0.0)
    step = int(_get(obs, "step", 0) or 0)
    comet_pids = set(_get(obs, "comet_planet_ids", []) or [])
    initial = {ip[0]: ip for ip in (_get(obs, "initial_planets", planets) or planets)}
    cpath = {}
    for grp in (_get(obs, "comets", []) or []):
        ids = _get(grp, "planet_ids", []) or []
        paths = _get(grp, "paths", []) or []
        idx = _get(grp, "path_index", 0) or 0
        for pid, path in zip(ids, paths):
            cpath[pid] = (path, idx)

    P = len(planets)
    p_id = np.array([p[P_ID] for p in planets], np.int32)
    p_owner = np.array([p[P_OWNER] for p in planets], np.int32)
    p_x = np.array([p[P_X] for p in planets], np.float32)
    p_y = np.array([p[P_Y] for p in planets], np.float32)
    p_radius = np.array([p[P_R] for p in planets], np.float32)
    p_ships = np.array([p[P_SHIPS] for p in planets], np.int32)
    p_prod = np.array([p[P_PROD] for p in planets], np.int32)
    p_mask = np.ones(P, bool)
    p_is_comet = np.array([p[P_ID] in comet_pids for p in planets], bool)
    p_is_orbiting = np.zeros(P, bool)
    p_orbital_r = np.zeros(P, np.float32); p_orbital_a = np.zeros(P, np.float32)
    p_cpx = np.zeros((P, _L), np.float32); p_cpy = np.zeros((P, _L), np.float32)
    p_cidx = np.zeros(P, np.int32)
    p_clen = np.zeros(P, np.int32)
    for i, p in enumerate(planets):
        pid = p[P_ID]
        if not p_is_comet[i]:
            init = initial.get(pid)
            if init is not None:
                orb_r = math.hypot(init[2] - R.SUN_X, init[3] - R.SUN_Y)
                p_orbital_r[i] = orb_r
                if orb_r + init[4] < R.ROTATION_LIMIT:
                    p_is_orbiting[i] = True
                    p_orbital_a[i] = math.atan2(p_y[i] - R.SUN_Y, p_x[i] - R.SUN_X)
        elif pid in cpath:
            path, idx = cpath[pid]
            n = min(len(path), _L)
            for j in range(n):
                p_cpx[i, j] = path[j][0]; p_cpy[i, j] = path[j][1]
            p_cidx[i] = idx
            p_clen[i] = n

    id2slot = {int(p_id[i]): i for i in range(P)}
    F = len(fleets)
    f_owner = np.array([f[F_OWNER] for f in fleets], np.int32) if F else np.zeros(0, np.int32)
    f_ships = np.array([f[F_SHIPS] for f in fleets], np.int32) if F else np.zeros(0, np.int32)
    f_mask = np.ones(F, bool)
    f_target = np.full(F, -1, np.int32); f_arrival = np.full(F, -1, np.int32)
    for i, f in enumerate(fleets):
        h = state.fleet_hit.get(f[F_ID])
        if h and h["kind"] == "planet" and h["planet"] is not None:
            slot = id2slot.get(h["planet"])
            if slot is not None and h["turn"] is not None:
                f_target[i] = slot; f_arrival[i] = h["turn"]

    arr = dict(p_owner=p_owner, p_x=p_x, p_y=p_y, p_radius=p_radius, p_ships=p_ships,
               p_prod=p_prod, p_mask=p_mask, p_is_comet=p_is_comet, p_is_orbiting=p_is_orbiting,
               p_orbital_r=p_orbital_r, p_orbital_a=p_orbital_a, p_comet_path_x=p_cpx,
               p_comet_path_y=p_cpy, p_comet_idx=p_cidx, p_comet_len=p_clen,
               f_owner=f_owner, f_ships=f_ships,
               f_target=f_target, f_arrival=f_arrival, f_mask=f_mask, step=step, av=av)
    return arr, p_id


def _encode(dec, p_id):
    """decode tuple (launch, tid, angle, ships) -> kaggle action list [[from_id, angle, ships], ...]."""
    launch, tid, angle, ships = dec
    return [[int(p_id[i]), float(angle[i]), int(ships[i])]
            for i in range(len(p_id)) if launch[i] and ships[i] > 0]


def _agree(arr, aA, aB, me):
    """True iff for every MINE planet, aA & aB match on (launch, target_id, ships).
    Mirrors merge_eval2.merged eq = (launch eq)&(tid eq)&(ships eq), all over is_mine."""
    lA, tA, _angA, sA = aA
    lB, tB, _angB, sB = aB
    is_mine = (arr["p_owner"] == int(me)) & arr["p_mask"]
    eq = (lA == lB) & (tA == tB) & (sA == sB)
    return bool(np.all(np.where(is_mine, eq, True)))


def _rollout(base_obs, initial, me, opp, first_act, W_me, W_opp, h):
    """h-turn rollout from base_obs. first_act = the chosen me-action this turn (kaggle list).
    Step 1: opp greedy via W_opp, apply (first_act @ me, opp0 @ opp). Then H-1 steps each applying
    (decode(W_me)@me, decode(W_opp)@opp). Return unified value 0.5*(V_A + V_B) for `me` at final state."""
    st = from_kaggle_obs(base_obs)

    # --- turn 0 (the disagree turn): me plays `first_act`, opp plays its greedy under W_opp ---
    fresh = _rs(); o = dict(base_obs); o["player"] = opp; fresh.update(o)
    arr_opp, pid_opp = _obs_to_arr(o, fresh)
    opp0 = _encode(R.decode(arr_opp, W_opp, opp), pid_opp) if arr_opp["p_x"].shape[0] else []
    st = _SIM.step(st, {me: first_act, opp: opp0})

    # --- h-1 more turns: both sides greedy under their pMe/pOpp ---
    for _ in range(max(0, h - 1)):
        obs_me = dict(state_to_obs(st, me, initial)); obs_me["player"] = me
        s_me = _rs(); s_me.update(obs_me)
        arr_me, pid_me = _obs_to_arr(obs_me, s_me)
        if arr_me["p_x"].shape[0] == 0:
            break
        a_me = _encode(R.decode(arr_me, W_me, me), pid_me)

        obs_op = dict(state_to_obs(st, opp, initial)); obs_op["player"] = opp
        s_op = _rs(); s_op.update(obs_op)
        arr_op, pid_op = _obs_to_arr(obs_op, s_op)
        a_op = _encode(R.decode(arr_op, W_opp, opp), pid_op) if arr_op["p_x"].shape[0] else []

        st = _SIM.step(st, {me: a_me, opp: a_op})

    # --- unified value at final state: ALWAYS mean of A's and B's value head, for `me` ---
    obs_f = dict(state_to_obs(st, me, initial)); obs_f["player"] = me
    s_f = _rs(); s_f.update(obs_f)
    arr_f, _ = _obs_to_arr(obs_f, s_f)
    if arr_f["p_x"].shape[0] == 0:
        return 0.0
    return 0.5 * (float(R.value_of(arr_f, _WA, me)) + float(R.value_of(arr_f, _WB, me)))


def agent(obs, config=None):
    global _OVERTIME_USED
    _t0 = time.perf_counter()
    me = int(_get(obs, "player", 0) or 0); opp = 1 - me
    _STATE.update(obs)
    if int(_get(obs, "step", 0) or 0) <= 1:
        _OVERTIME_USED = 0.0                                  # reset the overtime bank at game start

    arr, p_id = _obs_to_arr(obs, _STATE)
    P = arr["p_x"].shape[0]
    if P == 0:
        return []

    aA = R.decode(arr, _WA, me)
    aB = R.decode(arr, _WB, me)

    # ── overtime safety (double guard, conservative) ──
    # Read the OFFICIAL 60s overage bank: obs["remainingOverageTime"] (starts 60, the env decrements it by
    # each turn's seconds over the 1s actTimeout). `used` = seconds of the bank already spent, taken as the
    # MAX of the official reading and our own per-turn wall-clock tally (_OVERTIME_USED) — trust whichever
    # says MORE spent, so a missing/wrong env field can never make us over-spend. Skip the rollout and play
    # the MAIN ckpt's greedy action (A = u55000) if EITHER:
    #   (a) used >= _BANK_S (50s soft cap, 10s margin) — stop merging well before the bank runs out, OR
    #   (b) used + _WORST_TURN > 59s — even one more worst-case rollout turn can't push us past the 59s hard
    #       limit -> guarantees no single long turn ever times out, even if we haven't hit the 50s cap.
    overage = _get(obs, "remainingOverageTime", None)
    used = _OVERTIME_USED if overage is None else max(60.0 - float(overage), _OVERTIME_USED)
    remaining = 60.0 - used                                    # overage bank left (conservative)
    # ADAPTIVE rollout horizon by remaining bank: deep when plentiful, shallow when moderate, greedy when low.
    if remaining > _H3_REMAIN:
        h = _H_DEEP                                            # > 15s left -> H=3
    elif remaining > _H2_REMAIN:
        h = _H_MID                                             # > 5s left  -> H=2
    else:
        h = 0                                                 # <= 5s left -> greedy A (main ckpt)
    if used + _WORST_TURN > 59.0:
        h = 0                                                 # hard TLE floor: one more worst turn can't pass 59s
    if os.environ.get("EVAL_FORCE_MERGE"):
        h = _H_DEEP                                           # local eval: ignore the overage bank -> always full merge
    if _agree(arr, aA, aB, me) or h == 0:
        act = _encode(aA, p_id)
    else:
        # DISAGREE -> h-turn rollout each config, unified value, pick higher
        initial = _get(obs, "initial_planets", _get(obs, "planets", [])) or []
        first_A = _encode(aA, p_id)
        first_B = _encode(aB, p_id)
        vj = _rollout(obs, initial, me, opp, first_A, _WA, _WB, h)   # me plays A, opp plays B
        vy = _rollout(obs, initial, me, opp, first_B, _WB, _WA, h)   # me plays B, opp plays A
        act = first_A if vj >= vy else first_B

    dt = time.perf_counter() - _t0
    if dt > 1.0 and not _NOSELFTIME:
        _OVERTIME_USED += (dt - 1.0)                          # accrue wall-clock over 1s/turn into the bank
    return act


# Pin the entrypoint (Kaggle picks the LAST callable in main.py — see memory).
__kaggle_entrypoint__ = agent
