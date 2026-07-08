"""obs_state.py — weight-free kaggle-obs -> rl_infer `arr` dict reconstruction (exp23 4p IL build stack).

VERBATIM copy of exp22-4p agent._obs_to_arr (4p deploy reconstruction), pulled into its own module so
build_dataset_4p.py can import it WITHOUT triggering agent.py's module-level weight load (mirrors the 2p
exp20 obs_state.py layout). f_target/f_arrival come from engine.AgentState.fleet_hit (geometric first-hit,
owner-agnostic) — the SAME f_target/f_arrival the RL features see. agent.py keeps its own identical copy so
the deploy path stays byte-identical to the proven exp22-4p agent. If you change one, change both.
"""
import math
import numpy as np

import rl_infer as R                    # numpy forward lib; load_weights is NOT called at import (weight-free)

_L = 64  # MAX_COMET_PATH

# obs tuple indices (debug.md): planet [id,owner,x,y,radius,ships,prod]; fleet [id,owner,x,y,angle,from,ships]
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
