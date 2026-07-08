"""Kaggle orbit_wars agent: the trained OrbitNet RL policy, pure numpy (NO jax/torch).

Per turn: parse obs -> AgentState (deterministic-physics cache; derives each fleet's
first-hit target+arrival via shared.physics.first_hit_from) -> build the (P,FP) features +
(P,P) reachability mask -> numpy OrbitNet forward -> reach-masked greedy decode -> emit
[src_planet_id, abs_angle_radians, ships]. Feature byte-parity with the JAX trainer verified
in feature_parity.py (numpy basic_features == jax env.basic_features). exp13: NO emit-verify —
the firing judgment is the model's (see `agent` below). Weights npz shipped alongside this file.
"""
import os
import sys
import math
import numpy as np

# UNCONDITIONAL insert (Kaggle pops exec_dir after exec; the `if not in` guard can skip — see memory)
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import rl_infer as R
from engine import AgentState

_W = R.load_weights(os.environ.get("ORBIT25_2P_WEIGHTS", os.path.join(_HERE, "weights", "weights_2p_u55000.npz")))
_STATE = AgentState()
_L = 64  # MAX_COMET_PATH

# obs tuple indices
P_ID, P_OWNER, P_X, P_Y, P_R, P_SHIPS, P_PROD = range(7)
F_ID, F_OWNER, F_SHIPS = 0, 1, 6


def _get(o, key, default=None):
    return o.get(key, default) if isinstance(o, dict) else getattr(o, key, default)


def _obs_to_arr(obs, state):
    """kaggle obs -> rl_infer arr dict (P = active planets, obs order). f_target/f_arrival
    come from `state.fleet_hit` (first_hit_from, kind=='planet')."""
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
    p_clen = np.zeros(P, np.int32)          # v2: comet path length (for comet-lifetime reachability)
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



def agent(obs, config=None):
    me = int(_get(obs, "player", 0) or 0)
    _STATE.update(obs)
    arr, p_id = _obs_to_arr(obs, _STATE)
    P = arr["p_x"].shape[0]
    if P == 0:
        return []
    launch, tid, angle, ships = R.decode(arr, _W, me)
    # exp13: NO emit-verify. The firing JUDGMENT is the MODEL's, not the wrapper's. The jax
    # trainer lets imperfect shots fly (the env stamps each fleet's REAL first-hit into f_target,
    # and the reward teaches the policy to avoid wasteful/own-planet shots); deploy mirrors train,
    # so we emit decode's launches as-is. The only gate is the reach+conv mask INSIDE decode,
    # which is identical on the train side. (agent = thin I/O wrapper.)
    actions = []
    for i in range(P):
        if launch[i] and ships[i] > 0:
            actions.append([int(p_id[i]), float(angle[i]), int(ships[i])])
    return actions


# Pin the entrypoint (Kaggle picks the LAST callable in main.py — see memory).
__kaggle_entrypoint__ = agent
