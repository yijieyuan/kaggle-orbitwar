"""Low-level geometry primitives — pure Python (math only), scalar.

These are the per-element building blocks used by the aim (§2) and first-hit
(§3) functions. They mirror the kaggle 1.30.1 env exactly.
"""
import math

from .constants import SUN_X, SUN_Y, SUN_R, SUN_SAFETY, MAX_SPEED, BOARD

_LOG1000 = math.log(1000.0)


def fleet_speed(ships) -> float:
    """Log speed curve: 1 ship -> 1.0, 1000+ ships -> MAX_SPEED (6.0). Matches env."""
    n = int(ships)
    if n <= 1:
        return 1.0
    if n > 1000:
        n = 1000
    return 1.0 + (MAX_SPEED - 1.0) * (math.log(n) / _LOG1000) ** 1.5


def dist(ax, ay, bx, by) -> float:
    return math.hypot(ax - bx, ay - by)


def point_seg_dist2(cx, cy, x0, y0, x1, y1) -> float:
    """Squared distance from point (cx,cy) to segment (x0,y0)-(x1,y1)."""
    dx = x1 - x0
    dy = y1 - y0
    L2 = dx * dx + dy * dy
    if L2 < 1e-12:
        return (cx - x0) ** 2 + (cy - y0) ** 2
    t = ((cx - x0) * dx + (cy - y0) * dy) / L2
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    fx = x0 + t * dx
    fy = y0 + t * dy
    return (cx - fx) ** 2 + (cy - fy) ** 2


def seg_hits_sun(x0, y0, x1, y1, margin: float = SUN_SAFETY) -> bool:
    """Planning sun-block test: does the path come within (SUN_R + margin) of the sun?
    aim (§2) uses margin=SUN_SAFETY (11.5). The env's actual fleet-kill uses raw
    SUN_R (10.0) — pass margin=0.0 for that (first_hit does its own raw check)."""
    eff = SUN_R + margin
    return point_seg_dist2(SUN_X, SUN_Y, x0, y0, x1, y1) < eff * eff


def swept_pair_hit(ax, ay, bx, by, p0x, p0y, p1x, p1y, r) -> bool:
    """kaggle 1.30.1 continuous swept-pair collision. True iff a fleet moving
    (ax,ay)->(bx,by) and a planet moving (p0x,p0y)->(p1x,p1y) come within r at
    some t in [0,1]. Byte-identical to orbit_wars.py::swept_pair_hit."""
    d0x = ax - p0x
    d0y = ay - p0y
    dvx = (bx - ax) - (p1x - p0x)
    dvy = (by - ay) - (p1y - p0y)
    a = dvx * dvx + dvy * dvy
    b = 2.0 * (d0x * dvx + d0y * dvy)
    c = d0x * d0x + d0y * d0y - r * r
    if a < 1e-12:
        return c <= 0.0
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return False
    sq = math.sqrt(disc)
    t1 = (-b - sq) / (2.0 * a)
    t2 = (-b + sq) / (2.0 * a)
    return t2 >= 0.0 and t1 <= 1.0


def in_board(x, y) -> bool:
    return 0.0 <= x <= BOARD and 0.0 <= y <= BOARD
