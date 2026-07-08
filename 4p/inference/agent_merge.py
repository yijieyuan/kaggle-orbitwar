"""4p value-ensemble MERGE deploy agent (pure-numpy) — FLAT 4p inference.

Value-ensemble 1-ply lookahead over TWO exp25-simplefrac 4p checkpoints (A=u44000 main, B=u39000):

  a0 = decode(arr, W_A, me)          # #1 (A=u44000) proposed action
  a1 = decode(arr, W_B, me)          # #2 (B=u39000) proposed action
  # scenario A: I play a0, the OTHER 3 seats play W_B greedy -> 1 forward_sim step -> stA
  #             then _advance(H-1 extra steps: me=W_A greedy, others=W_B greedy)
  # scenario B: symmetric (me=a1, others=W_A; advance me=W_B, others=W_A)
  vA = 0.5*(value_of(W_A, stA, me) + value_of(W_B, stA, me))   # both value heads, averaged
  vB = 0.5*(value_of(W_A, stB, me) + value_of(W_B, stB, me))
  pick a0 if vA >= vB else a1                                  # higher H-step value

Mirrors 4p/eval_cross/merge_deploy/agent_merge.py (verified: H=2 seed-cache fidelity 444/444 vs
full-recompute). Imports rl_infer/engine/forward_sim/sim_runner as FLAT top-level modules; these SHARE
names with the 2p side, so a single process must not import both tracks. Pure-numpy (no torch/jax).

ADAPTIVE-H driven by the official 60s overage bank: each turn reads obs["remainingOverageTime"].
When the bank has > 20s margin -> H=2 merge (1 forward_sim step + 1 extra _advance step, value-ensemble
pick). Otherwise -> GREEDY (just decode(W_A), no rollout). Hard floor: if a worst-turn estimate would
breach the bank -> greedy. This caps used-bank well below 60s -> never TLE.

Weights via env (set by make_agent / the submission entry, absolute):
  ORBIT4P_MERGE_A = .../4p/inference/weights/weights_4p_u44000.npz   (A = #1 = main)
  ORBIT4P_MERGE_B = .../4p/inference/weights/weights_4p_u39000.npz   (B = #2)
"""
import os
import sys
import math
import copy
import time
import numpy as np

# Flat 4p inference: import siblings top-level. These exp25 4p rl_infer/engine/forward_sim/sim_runner
# SHARE names with the 2p side, so a single process must not import both tracks.
_HERE = os.path.dirname(os.path.abspath(__file__))

import rl_infer as R
from engine import AgentState
from forward_sim import from_kaggle_obs, OrbitSimulator
from sim_runner import state_to_obs

_WA = R.load_weights(os.environ.get("ORBIT4P_MERGE_A", os.path.join(_HERE, "weights", "weights_4p_u44000.npz")))  # A = #1 (u44000)
_WB = R.load_weights(os.environ.get("ORBIT4P_MERGE_B", os.path.join(_HERE, "weights", "weights_4p_u39000.npz")))  # B = #2 (u39000)
_STATE = AgentState()
_SIM = OrbitSimulator()
_L = 64  # MAX_COMET_PATH

# --- ADAPTIVE-H bank accounting -----------------------------------------------------------------
# The official env grants a 60s overage bank on top of the 1s/turn budget. We estimate how much of
# that bank we've consumed (used) and only do the H=2 merge while there's comfortable margin left.
_BANK = 60.0          # total overage bank (s)
_PER_TURN_BUDGET = 1.0
_H_MERGE = 2          # merge lookahead depth when the bank allows it
_WORST_TURN_EST = 3.0 # conservative upper bound on a single merge turn's overage (s)
_MARGIN = 20.0        # require remaining_bank > this (s) before doing the merge
_FLOOR = 59.0         # if used + worst-turn-est would exceed this -> greedy (hard floor)
_self_tally = 0.0     # our own accrued overage (sum of per-turn (wall - 1s), floored at 0)
_last_step = -1


def _rs():
    """Lookahead AgentState reusing the live _STATE caches (carry planet_traj/game_key -> update()
    SLICES, no ~190ms rebuild; copy mutable fleet/comet caches so the lookahead's launches don't
    pollute _STATE). EXACT replication of agent_merge.py::_rs (seed-cache, fidelity 444/444)."""
    s = copy.copy(_STATE)
    s.fleet_hit = dict(_STATE.fleet_hit)
    s.known_fleets = set(_STATE.known_fleets)
    s.fleet_traj = dict(_STATE.fleet_traj)
    s.comet_traj = dict(_STATE.comet_traj)
    s.known_comets = set(_STATE.known_comets)
    return s


P_ID, P_OWNER, P_X, P_Y, P_R, P_SHIPS, P_PROD = range(7)
F_ID, F_OWNER, F_SHIPS = 0, 1, 6


def _get(o, key, default=None):
    return o.get(key, default) if isinstance(o, dict) else getattr(o, key, default)


def _obs_to_arr(obs, state):
    """kaggle obs -> rl_infer arr dict (copy of exp25-4p agent._obs_to_arr)."""
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
    p_cidx = np.zeros(P, np.int32); p_clen = np.zeros(P, np.int32)
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
            p_cidx[i] = idx; p_clen[i] = n

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
               f_owner=f_owner, f_ships=f_ships, f_target=f_target, f_arrival=f_arrival,
               f_mask=f_mask, step=step, av=av)
    return arr, p_id


def _encode(dec, p_id):
    launch, tid, angle, ships = dec
    return [[int(p_id[i]), float(angle[i]), int(ships[i])]
            for i in range(len(p_id)) if launch[i] and ships[i] > 0]


def _value_at(st, initial, W_list, me):
    """mean value head(s) of `me`'s seat at lookahead state st. REUSE the main agent's INCREMENTAL
    fleet_hit cache (_STATE) via the seed-cache: a fleet's first-hit is FIXED once launched, so the
    lookahead's persisting in-flight fleets are LOOKED UP (no O(fleets x walk) re-prediction); new
    launches just left -> miss the cache -> f_target=-1, negligible for an H-step value and symmetric
    across stA/stB. Same as agent_merge.py::_value_at."""
    o = state_to_obs(st, me, initial); o["player"] = me
    s = _rs(); s.update(o)            # seeded: slice planet_traj + walk only NEW launches (EXACT + fast)
    arr, _ = _obs_to_arr(o, s)
    if arr["p_x"].shape[0] == 0:
        return 0.0
    return float(np.mean([R.value_of(arr, W, me) for W in W_list]))


def _advance(st, W_me, W_opp, me, others, initial, n):
    """n extra rollout steps after the first: seat `me` plays W_me greedy, the other 3 seats play
    W_opp greedy. H=2 -> n=1 (one extra step). Same as agent_merge.py::_advance."""
    for _ in range(n):
        o = state_to_obs(st, 0, initial)
        s = _rs(); s.update(o)
        arr, pid = _obs_to_arr(o, s)
        if arr["p_x"].shape[0] == 0:
            break
        act = {me: _encode(R.decode(arr, W_me, me), pid)}
        for k in others:
            act[k] = _encode(R.decode(arr, W_opp, k), pid)
        st = _SIM.step(st, act)
    return st


def _greedy(arr, p_id, me):
    """Plain single-checkpoint greedy decode with W_A (the main, #1 checkpoint). No rollout."""
    return _encode(R.decode(arr, _WA, me), p_id)


def _do_merge(obs, arr, p_id, me, H):
    """Value-ensemble pick between a0 (W_A) and a1 (W_B) via H-step lookahead. EXACT replication of
    agent_merge.py::agent merge body (generalized to depth H)."""
    a0 = R.decode(arr, _WA, me)        # (launch, tid, angle, ships)
    a1 = R.decode(arr, _WB, me)
    initial = _get(obs, "initial_planets", _get(obs, "planets", [])) or []
    state = from_kaggle_obs(obs, n_players=4)
    others = [k for k in range(4) if k != me]
    # scenario A: me=a0, others = W_B greedy
    actA = {me: _encode(a0, p_id)}
    for k in others:
        actA[k] = _encode(R.decode(arr, _WB, k), p_id)
    stA = _SIM.step(state, actA)
    stA = _advance(stA, _WA, _WB, me, others, initial, H - 1)   # H-1 extra steps (me=W_A, opp=W_B)
    vA = 0.5 * (_value_at(stA, initial, [_WA], me) + _value_at(stA, initial, [_WB], me))
    # scenario B: me=a1, others = W_A greedy
    actB = {me: _encode(a1, p_id)}
    for k in others:
        actB[k] = _encode(R.decode(arr, _WA, k), p_id)
    stB = _SIM.step(state, actB)
    stB = _advance(stB, _WB, _WA, me, others, initial, H - 1)   # H-1 extra steps (me=W_B, opp=W_A)
    vB = 0.5 * (_value_at(stB, initial, [_WA], me) + _value_at(stB, initial, [_WB], me))
    pick = a0 if vA >= vB else a1
    return _encode(pick, p_id)


def agent(obs, config=None):
    global _self_tally, _last_step
    t0 = time.perf_counter()
    me = int(_get(obs, "player", 0) or 0)
    step = int(_get(obs, "step", 0) or 0)

    # Reset the self-overage tally at game start (step 0/1).
    if step <= 1 or step < _last_step:
        _self_tally = 0.0
    _last_step = step

    # --- ADAPTIVE-H: estimate bank used, decide merge vs greedy ---------------------------------
    remaining = _get(obs, "remainingOverageTime", None)
    if remaining is not None:
        # The official bank report is authoritative; combine with our own tally as a safety lower
        # bound (in case the env under-reports or this is a fresh game with no report yet).
        used = max(_BANK - float(remaining), _self_tally)
    else:
        used = _self_tally
    remaining_bank = _BANK - used
    # Do the merge only with comfortable margin AND a hard floor against the worst single turn.
    do_merge = (remaining_bank > _MARGIN) and (used + _WORST_TURN_EST <= _FLOOR)
    if os.environ.get("EVAL_FORCE_MERGE"):
        do_merge = True                                    # local eval: ignore the Kaggle overage bank -> always full merge

    _STATE.update(obs)
    arr, p_id = _obs_to_arr(obs, _STATE)
    P = arr["p_x"].shape[0]
    if P == 0:
        return []

    if do_merge:
        actions = _do_merge(obs, arr, p_id, me, _H_MERGE)
    else:
        actions = _greedy(arr, p_id, me)

    # Accrue our own overage (wall-clock beyond the 1s/turn budget, floored at 0).
    elapsed = time.perf_counter() - t0
    _self_tally += max(0.0, elapsed - _PER_TURN_BUDGET)
    return actions


# Pin the entrypoint (Kaggle picks the LAST callable in main.py — see memory).
__kaggle_entrypoint__ = agent
