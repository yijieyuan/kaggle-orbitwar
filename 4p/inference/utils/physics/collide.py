"""FUNCTION 3: first_hit — 按当前 ships 和 angle，第一个撞到的 planet.

Walk a fleet turn by turn and report the FIRST event using the kaggle 1.30.1
swept-pair collision (fleet segment A->B vs each planet segment P0->P1 moving
together). Within a turn: planet > out-of-board > sun; first planet in list
order wins. No source exclusion (matches the env). Scalar / early-exit.

Two entry points:
  first_hit(src, ...)        — a NEW launch (fleet spawns at src.radius+0.1 outside src)
  first_hit_from(x, y, ...)  — a fleet ALREADY at (x,y) (e.g. an in-flight fleet from obs)

Both accept an optional `planet_table` = {pid: seq of (x,y) indexed by RELATIVE
turn k (k=0 == now)} so callers (the agent cache) can feed precomputed planet
positions instead of recomputing the orbital motion every call. Comets are
handled via `comets`/`comet_planet_ids` regardless of the table.

Returns (planet_id, turn, kind):  kind ∈ {'planet','sun','oob','none'}; turn is
relative (1 = first move). turn is None only for 'none'.
"""
import math

from .constants import SUN_X, SUN_Y, SUN_R, BOARD, LAUNCH_CLEARANCE, LAST_TURN
from .geom import fleet_speed, swept_pair_hit, point_seg_dist2
from .motion import planet_pos_after, comet_lookup, P_ID, P_X, P_Y, P_R


def _first_event(lx, ly, vx, vy, planets, angular_velocity, comet_map,
                 start_turn, end_turn, planet_table):
    sun_r2 = SUN_R * SUN_R
    T = int(end_turn) - int(start_turn)
    for t in range(1, T + 1):
        ax = lx + (t - 1) * vx
        ay = ly + (t - 1) * vy
        bx = lx + t * vx
        by = ly + t * vy

        for p in planets:
            pid = p[P_ID]
            if planet_table is not None and pid in planet_table:
                tab = planet_table[pid]
                if t >= len(tab):
                    continue                       # no cached pos this far -> treat as absent
                p0x, p0y = tab[t - 1]
                p1x, p1y = tab[t]
            elif pid in comet_map:
                cpath, cidx = comet_map[pid]
                p0 = planet_pos_after(p, angular_velocity, t - 1, comet_path=cpath, comet_path_index=cidx)
                if p0 is None:
                    continue                       # comet already gone before this turn
                p1 = planet_pos_after(p, angular_velocity, t, comet_path=cpath, comet_path_index=cidx)
                if p1 is None:
                    p1 = p0                         # expires this turn -> stays put (matches env)
                p0x, p0y = p0
                p1x, p1y = p1
            else:
                p0x, p0y = planet_pos_after(p, angular_velocity, t - 1)
                p1x, p1y = planet_pos_after(p, angular_velocity, t)
            if swept_pair_hit(ax, ay, bx, by, p0x, p0y, p1x, p1y, p[P_R]):
                return (pid, t, "planet")

        if not (0.0 <= bx <= BOARD and 0.0 <= by <= BOARD):
            return (None, t, "oob")
        if point_seg_dist2(SUN_X, SUN_Y, ax, ay, bx, by) < sun_r2:
            return (None, t, "sun")

    return (None, None, "none")


def first_hit_from(start_x, start_y, angle, ships, planets, angular_velocity,
                   comets=None, comet_planet_ids=None, start_turn=0, end_turn=LAST_TURN,
                   planet_table=None):
    """First event for a fleet ALREADY at (start_x, start_y) — e.g. an in-flight
    fleet read from the observation (do NOT add the launch clearance here)."""
    sp = fleet_speed(ships)
    vx = math.cos(angle) * sp
    vy = math.sin(angle) * sp
    comet_map = comet_lookup(comets, comet_planet_ids)
    return _first_event(start_x, start_y, vx, vy, planets, angular_velocity,
                        comet_map, start_turn, end_turn, planet_table)


def first_hit(src, ships, angle, planets, angular_velocity, comets=None,
              comet_planet_ids=None, start_turn=0, end_turn=LAST_TURN, planet_table=None):
    """First event for a NEW launch from `src` (fleet spawns at src.radius+0.1
    outside the source planet, then flies along `angle`)."""
    lx = src[P_X] + math.cos(angle) * (src[P_R] + LAUNCH_CLEARANCE)
    ly = src[P_Y] + math.sin(angle) * (src[P_R] + LAUNCH_CLEARANCE)
    return first_hit_from(lx, ly, angle, ships, planets, angular_velocity, comets=comets,
                          comet_planet_ids=comet_planet_ids, start_turn=start_turn,
                          end_turn=end_turn, planet_table=planet_table)
