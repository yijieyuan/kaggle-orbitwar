"""Object motion + FUNCTION 1: future_positions.

Given the current planets / fleets (+ comet paths if any), compute every
object's coordinates at every turn from `start_turn` to `end_turn` (=499 by
default = the last state the game cares about). A fleet that leaves the board
is dropped from that turn on (we don't keep simulating it). Collisions are NOT
applied here — this is pure kinematics (where things WOULD be); the first
collision is §3 `first_hit`.

Motion rules (match kaggle 1.30.1):
  - static planet (orbital_r + radius >= 50): never moves.
  - 公转 planet (orbital_r + radius < 50): rotates about the sun at
    `angular_velocity` rad/turn (position = sun + r*(cos,sin)(angle0 + av*k)).
  - comet: follows its precomputed path; once the path runs out it expires
    (its sequence simply stops).
  - fleet: straight line at fleet_speed(ships) along its angle; stops at border.

Tuple layout (as given by the kaggle observation):
  planet = [id, owner, x, y, radius, ships, production]
  fleet  = [id, owner, x, y, angle, from_planet_id, ships]
"""
import math

import numpy as np

from .constants import SUN_X, SUN_Y, ROTATION_LIMIT, BOARD, LAST_TURN
from .geom import fleet_speed

# planet tuple indices
P_ID, P_OWNER, P_X, P_Y, P_R, P_SHIPS, P_PROD = range(7)
# fleet tuple indices
F_ID, F_OWNER, F_X, F_Y, F_ANGLE, F_FROM, F_SHIPS = range(7)


# ----------------------------- scalar helpers (used by §3) -----------------------------
def orbit_params(px, py, radius):
    """(is_orbiting, orbital_r, base_angle) derived from a planet's CURRENT position."""
    dx = px - SUN_X
    dy = py - SUN_Y
    r = math.hypot(dx, dy)
    orbiting = (r + radius) < ROTATION_LIMIT
    return orbiting, r, math.atan2(dy, dx)


def planet_pos_after(planet, angular_velocity, k, comet_path=None, comet_path_index=0):
    """Position of a planet/comet k turns from its CURRENT state.
    If `comet_path` is given the object follows it (returns None once the path
    runs out = comet expired). Otherwise orbital/static rotation about the sun."""
    if comet_path is not None:
        idx = comet_path_index + k
        if 0 <= idx < len(comet_path):
            cx, cy = comet_path[idx]
            return (float(cx), float(cy))
        return None
    px, py, radius = planet[P_X], planet[P_Y], planet[P_R]
    orbiting, r, base = orbit_params(px, py, radius)
    if not orbiting:
        return (px, py)
    a = base + angular_velocity * k
    return (SUN_X + r * math.cos(a), SUN_Y + r * math.sin(a))


def fleet_pos_after(fleet, k):
    """Straight-line position of a fleet k turns from its CURRENT state."""
    sp = fleet_speed(fleet[F_SHIPS])
    ang = fleet[F_ANGLE]
    return (fleet[F_X] + k * sp * math.cos(ang), fleet[F_Y] + k * sp * math.sin(ang))


def comet_lookup(comets, comet_planet_ids=None):
    """Build {pid: (path[list of (x,y)], path_index)} from kaggle obs.comets groups."""
    out = {}
    if not comets:
        return out
    for group in comets:
        if isinstance(group, dict):
            ids = group.get("planet_ids", [])
            paths = group.get("paths", [])
            idx = group.get("path_index", 0)
        else:
            ids = getattr(group, "planet_ids", [])
            paths = getattr(group, "paths", [])
            idx = getattr(group, "path_index", 0)
        for pid, path in zip(ids, paths):
            out[pid] = ([(float(p[0]), float(p[1])) for p in path], idx)
    return out


# ----------------------------- FUNCTION 1 -----------------------------
def future_positions(planets, fleets, angular_velocity, comets=None,
                     comet_planet_ids=None, start_turn=0, end_turn=LAST_TURN):
    """每一个 turn 的坐标，从现在到 turn 499（end_turn）。

    Returns a dict:
      {
        'planets': {pid: ndarray (n,3) of [turn, x, y]},   # comets stop when path ends
        'fleets':  {fid: ndarray (m,3) of [turn, x, y]},   # stops once it leaves the board
      }
    Row 0 is the CURRENT state (turn == start_turn). Positions are vectorized
    over the turn horizon with numpy.
    """
    comet_map = comet_lookup(comets, comet_planet_ids)
    horizon = int(end_turn) - int(start_turn)
    if horizon < 0:
        horizon = 0
    k = np.arange(0, horizon + 1)               # 0..horizon
    turns = start_turn + k

    out_p = {}
    for p in planets:
        pid = p[P_ID]
        if pid in comet_map:
            path, base_idx = comet_map[pid]
            n = len(path)
            valid_k = k[(base_idx + k >= 0) & (base_idx + k < n)]   # turns the comet still exists
            if valid_k.size == 0:
                out_p[pid] = np.empty((0, 3))
                continue
            idxs = base_idx + valid_k
            xy = np.array([path[i] for i in idxs], dtype=float)     # (v,2)
            rows = np.column_stack([start_turn + valid_k, xy[:, 0], xy[:, 1]])
            out_p[pid] = rows
            continue
        px, py, radius = p[P_X], p[P_Y], p[P_R]
        orbiting, r, base = orbit_params(px, py, radius)
        if not orbiting:
            xs = np.full(k.shape, float(px))
            ys = np.full(k.shape, float(py))
        else:
            # The env's rotation COUNT at obs.step=t is max(0, t-1): obs.step 0 and 1
            # are the same (initial) layout, then +av per step. `base` is the angle at
            # `start_turn` (rotation count rc0). Position at absolute turn t advances by
            # the rotation-count DELTA, not by k — correct for any start_turn incl. 0.
            rc0 = max(0, start_turn - 1)
            rc = np.maximum(0, turns - 1)                 # turns == start_turn + k
            ang = base + angular_velocity * (rc - rc0)
            xs = SUN_X + r * np.cos(ang)
            ys = SUN_Y + r * np.sin(ang)
        out_p[pid] = np.column_stack([turns, xs, ys])

    out_f = {}
    for f in fleets:
        sp = fleet_speed(f[F_SHIPS])
        ang = f[F_ANGLE]
        xs = f[F_X] + k * sp * math.cos(ang)
        ys = f[F_Y] + k * sp * math.sin(ang)
        inb = (xs >= 0.0) & (xs <= BOARD) & (ys >= 0.0) & (ys <= BOARD)
        if not inb[0] and inb.size:
            # already off-board at start_turn (shouldn't happen for a live fleet)
            cut = 0
        else:
            # keep the contiguous in-board prefix
            off = np.argmax(~inb) if (~inb).any() else inb.size
            cut = off
        out_f[f[F_ID]] = np.column_stack([turns[:cut], xs[:cut], ys[:cut]])
    return {"planets": out_p, "fleets": out_f}
