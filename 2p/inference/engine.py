"""experiment-040-skeleton — class-based agent skeleton on the NEW pure-Python physics.

The point of this file is the ENGINE, not the strategy: an `AgentState` that caches
all DETERMINISTIC future info so it's computed once and only sliced each turn —

  * planet (non-comet) future positions  : fixed for the whole game -> computed ONCE.
  * comet future positions                : fixed once it spawns       -> computed when it appears.
  * each fleet's straight-line trajectory  : fixed once launched        -> computed when first seen.
  * each fleet's first_hit (planet, turn)  : fixed once launched        -> computed when first seen,
                                             re-computed only at comet-spawn boundaries (hidden new comets).

From `fleet_hit` we derive `incoming[planet] = [(fid, abs_turn, owner, ships), ...]` for free.
`first_hit` is fed the sliced planet table from the cache, so it never recomputes orbital motion.

The actual decision (`AgentState.decide`) is a deliberately SIMPLE placeholder — real
strategy plugs in there, on top of the cached physics.
"""
import os
import sys
import math

import numpy as np

# --- make `shared` importable — vendored at the bundle root next to this file (see memory). ---
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from shared.physics import (
    future_positions, aim, first_hit, first_hit_from, fleet_speed, dist,
    LAST_TURN,
    P_ID, P_OWNER, P_X, P_Y, P_R, P_SHIPS, P_PROD,
    F_ID, F_OWNER, F_X, F_Y, F_ANGLE, F_FROM, F_SHIPS,
)

COMET_SPAWN_STEPS = {50, 150, 250, 350, 450}


def _get(o, key, default=None):
    if isinstance(o, dict):
        return o.get(key, default)
    return getattr(o, key, default)


class AgentState:
    """Persistent across turns within ONE episode. Holds the deterministic-future
    caches; resets automatically when a new game is detected."""

    def __init__(self):
        self._reset()

    # ----- lifecycle -----
    def _reset(self):
        self.game_key = None
        self.av = 0.0
        self.planet_base = 0                 # abs turn at which planet_traj was computed
        self.planet_traj = {}                # pid -> ndarray (n,2) xy, index k = abs turn planet_base+k
        self.comet_traj = {}                 # pid -> (ndarray (n,2), start_abs_turn)
        self.fleet_traj = {}                 # fid -> (ndarray (m,2), start_abs_turn)
        self.fleet_hit = {}                  # fid -> dict(planet,turn,kind,owner,ships)
        self.known_fleets = set()
        self.known_comets = set()
        self.last_step = -1
        self.n_obj_computes = 0              # diagnostic: object-trajectories computed this game

    # ----- per-turn update (the caching core) -----
    def update(self, obs):
        step = int(_get(obs, "step", 0) or 0)
        av = float(_get(obs, "angular_velocity", 0.0) or 0.0)
        planets = list(_get(obs, "planets", []) or [])
        fleets = list(_get(obs, "fleets", []) or [])
        comet_pids = set(_get(obs, "comet_planet_ids", []) or [])
        comets = list(_get(obs, "comets", []) or [])

        noncomet = [p for p in planets if p[P_ID] not in comet_pids]
        game_key = (round(av, 9), frozenset(p[P_ID] for p in noncomet))
        if game_key != self.game_key:                       # new game -> rebuild
            self._reset()
            self.game_key = game_key
            self.av = av
            self.planet_base = step
            fp = future_positions(noncomet, [], av, start_turn=step, end_turn=LAST_TURN)
            for pid, rows in fp["planets"].items():
                self.planet_traj[pid] = rows[:, 1:3]
                self.n_obj_computes += 1

        # planet table for this turn (sliced cache; index k = pos k turns from `step`)
        ptable = self._planet_table(step)

        # --- comets -> (re)cache. NOTE: comet ids are REUSED across spawn batches
        # (a 50-batch comet expires, then the 150-batch reuses the same id), so we
        # can't dedup by id. Treat a comet as new whenever the cache doesn't already
        # predict its CURRENT position -> re-cache it and flag a spawn. ---
        new_comet = False
        cur_by_id = {p[P_ID]: p for p in planets}
        for grp in comets:
            ids = _get(grp, "planet_ids", []) or []
            paths = _get(grp, "paths", []) or []
            pidx = _get(grp, "path_index", 0) or 0
            for cid, path in zip(ids, paths):
                cur = cur_by_id.get(cid)
                if cur is None:
                    continue
                cached = self.comet_pos(cid, step)
                if cached is not None and abs(cached[0] - cur[P_X]) < 1e-6 and abs(cached[1] - cur[P_Y]) < 1e-6:
                    continue                                   # same comet, already cached
                fp = future_positions([cur], [], av,
                                      comets=[{"planet_ids": [cid], "paths": [path], "path_index": pidx}],
                                      comet_planet_ids=[cid], start_turn=step, end_turn=LAST_TURN)
                self.comet_traj[cid] = (fp["planets"][cid][:, 1:3], step)
                self.known_comets.add(cid)
                new_comet = True
                self.n_obj_computes += 1

        # --- new fleets -> cache trajectory + first_hit ---
        cur_fids = set(f[F_ID] for f in fleets)
        for f in fleets:
            fid = f[F_ID]
            if fid in self.known_fleets:
                continue
            self.known_fleets.add(fid)
            fp = future_positions([], [f], av, start_turn=step, end_turn=LAST_TURN)
            self.fleet_traj[fid] = (fp["fleets"][fid][:, 1:3], step)
            self.fleet_hit[fid] = self._compute_fleet_hit(f, planets, comets, comet_pids, step, ptable)
            self.n_obj_computes += 1

        # --- a NEW comet just spawned -> still-flying fleets' first_hit may change ---
        if new_comet:
            for f in fleets:
                self.fleet_hit[f[F_ID]] = self._compute_fleet_hit(f, planets, comets, comet_pids, step, ptable)

        # --- drop vanished fleets (hit something / left board) ---
        for fid in list(self.known_fleets):
            if fid not in cur_fids:
                self.known_fleets.discard(fid)
                self.fleet_traj.pop(fid, None)
                self.fleet_hit.pop(fid, None)

        self.last_step = step

    def _compute_fleet_hit(self, f, planets, comets, comet_pids, step, ptable):
        pid, rel, kind = first_hit_from(
            f[F_X], f[F_Y], f[F_ANGLE], f[F_SHIPS], planets, self.av,
            comets=comets, comet_planet_ids=comet_pids,
            start_turn=step, end_turn=LAST_TURN, planet_table=ptable)
        return {"planet": pid, "turn": (step + rel) if rel is not None else None,
                "kind": kind, "owner": f[F_OWNER], "ships": f[F_SHIPS]}

    # ----- queries (all served from cache, no recompute) -----
    def _planet_table(self, now):
        off = now - self.planet_base
        return {pid: traj[off:] for pid, traj in self.planet_traj.items()}

    def planet_pos(self, pid, abs_turn):
        traj = self.planet_traj.get(pid)
        if traj is None:
            return None
        k = abs_turn - self.planet_base
        return tuple(traj[k]) if 0 <= k < len(traj) else None

    def comet_pos(self, cid, abs_turn):
        ct = self.comet_traj.get(cid)
        if ct is None:
            return None
        arr, base = ct
        k = abs_turn - base
        return tuple(arr[k]) if 0 <= k < len(arr) else None

    def fleet_pos(self, fid, abs_turn):
        ft = self.fleet_traj.get(fid)
        if ft is None:
            return None
        arr, base = ft
        k = abs_turn - base
        return tuple(arr[k]) if 0 <= k < len(arr) else None

    def incoming(self):
        """{planet_id: [(fid, abs_turn, owner, ships), ...] sorted by arrival turn}."""
        out = {}
        for fid, h in self.fleet_hit.items():
            if h["kind"] == "planet" and h["planet"] is not None:
                out.setdefault(h["planet"], []).append((fid, h["turn"], h["owner"], h["ships"]))
        for pid in out:
            out[pid].sort(key=lambda r: (r[1] if r[1] is not None else 1 << 30))
        return out

    # ----- placeholder decision (real strategy plugs in here) -----
    def decide(self, obs):
        step = int(_get(obs, "step", 0) or 0)
        me = int(_get(obs, "player", 0) or 0)
        planets = list(_get(obs, "planets", []) or [])
        comet_pids = set(_get(obs, "comet_planet_ids", []) or [])
        comets = list(_get(obs, "comets", []) or [])
        ptable = self._planet_table(step)
        mine = [p for p in planets if p[P_OWNER] == me and p[P_SHIPS] >= 15]
        targets = [p for p in planets if p[P_OWNER] != me]
        moves = []
        for src in mine:
            best = None
            for tgt in targets:
                res = aim(src, tgt, src[P_SHIPS] // 2, self.av, start_turn=step, end_turn=LAST_TURN)
                if not (res["reachable"] and res["converged"]):
                    continue
                hit = first_hit(src, src[P_SHIPS] // 2, res["angle"], planets, self.av,
                                comets=comets, comet_planet_ids=comet_pids,
                                start_turn=step, end_turn=LAST_TURN, planet_table=ptable)
                if hit[0] == tgt[P_ID] and hit[2] == "planet":      # the shot really lands on tgt
                    if best is None or res["turns"] < best[1]:
                        best = (res["angle"], res["turns"], tgt[P_ID])
            if best is not None:
                moves.append([src[P_ID], float(best[0]), int(src[P_SHIPS] // 2)])
        return moves


# Kaggle agent = a function; persist the cache in a module-level singleton across turns.
_STATE = AgentState()


def agent(observation, configuration=None):
    _STATE.update(observation)
    return _STATE.decide(observation)


# Pin the entrypoint (Kaggle picks the LAST callable; see memory feedback_kaggle_last_callable).
__kaggle_entrypoint__ = agent
