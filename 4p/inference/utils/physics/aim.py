"""FUNCTION 2: aim — 朝哪个方向打 + 能不能打得到.

Given a hypothetical fleet of `ships` launched from `src` at a (possibly
moving) `target`, return the firing angle, the arrival turn, and whether it is
actually reachable: enough turns left before the game ends (turn 499) AND the
launch->impact path is not blocked by the sun.

The target moves (公转/comet), so we solve a fixed-point lead: aim where the
target IS now, compute travel turns, re-aim where it WILL BE after those turns,
repeat until the lead point stops moving.

Pure Python (math only) — it's a tiny iterative solve, no win from vectorizing.
"""
import math

from .constants import LAUNCH_CLEARANCE, LAST_TURN
from .geom import fleet_speed, seg_hits_sun
from .motion import planet_pos_after, P_X, P_Y, P_R

_CONVERGE_TOL = 0.3   # lead point must settle within this (units) in x and y


def aim(src, target, ships, angular_velocity, start_turn=0, end_turn=LAST_TURN,
        max_iter=6, target_comet_path=None, target_comet_index=0):
    """Returns dict:
      angle        : float    — firing angle (radians)
      turns        : int      — arrival turns (relative; >=1)
      arrival_turn : int      — absolute = start_turn + turns
      reachable    : bool     — not sun-blocked AND arrives by end_turn
      sun_blocked  : bool
      too_late     : bool     — arrival_turn > end_turn
      converged    : bool     — the lead fixed-point settled (False => fast/close target, lead approximate)
      reason       : str      — 'ok' | 'sun_blocked' | 'too_late'
    """
    sx, sy, sr = src[P_X], src[P_Y], src[P_R]
    tr = target[P_R]
    lc = sr + LAUNCH_CLEARANCE

    def estimate(tx, ty):
        """Direct geometry to a FIXED point (tx,ty): firing angle, arrival turns,
        launch point, and impact point (target's outer edge)."""
        ang = math.atan2(ty - sy, tx - sx)
        lx = sx + math.cos(ang) * lc
        ly = sy + math.sin(ang) * lc
        center_d = math.hypot(tx - sx, ty - sy)
        hit_d = center_d - lc - tr
        if hit_d < 0.0:
            hit_d = 0.0
        turns = max(1, math.ceil(hit_d / fleet_speed(ships)))
        ex = lx + math.cos(ang) * hit_d
        ey = ly + math.sin(ang) * hit_d
        return ang, turns, lx, ly, ex, ey

    def target_after(turns):
        if target_comet_path is not None:
            idx = target_comet_index + turns
            if 0 <= idx < len(target_comet_path):
                return target_comet_path[idx]
            return (target[P_X], target[P_Y])
        return planet_pos_after(target, angular_velocity, turns)

    tx, ty = target[P_X], target[P_Y]
    ang, turns, lx, ly, ex, ey = estimate(tx, ty)
    converged = False
    for _ in range(max_iter):
        ptx, pty = target_after(turns)
        n_ang, n_turns, lx, ly, ex, ey = estimate(ptx, pty)
        if abs(ptx - tx) < _CONVERGE_TOL and abs(pty - ty) < _CONVERGE_TOL and abs(n_turns - turns) <= 1:
            ang, turns = n_ang, n_turns
            converged = True
            break
        tx, ty = ptx, pty
        ang, turns = n_ang, n_turns

    sun_blocked = seg_hits_sun(lx, ly, ex, ey)         # uses SUN_R + SUN_SAFETY (planning margin)
    arrival_turn = start_turn + turns
    too_late = arrival_turn > end_turn
    reachable = (not sun_blocked) and (not too_late)
    reason = "ok" if reachable else ("sun_blocked" if sun_blocked else "too_late")
    return {
        "angle": ang, "turns": turns, "arrival_turn": arrival_turn,
        "reachable": reachable, "sun_blocked": sun_blocked, "too_late": too_late,
        "converged": converged, "reason": reason,
    }
