"""Pure-Python physics (rewrite, 2026-06-02).

No numba / njit / jax. numpy used only to vectorize §1's per-turn position
table. Three public functions, all matching kaggle_environments 1.30.1 and
verified against shared/sim/forward_sim (see verify.py):

  §1 future_positions(planets, fleets, angular_velocity, ...) -> per-turn coords
       of every planet / comet / fleet from now to turn 499 (fleets stop at border).
  §2 aim(src, target, ships, angular_velocity, ...) -> firing angle + 能不能打到
       (arrival turns within budget, not sun-blocked).
  §3 first_hit(src, ships, angle, planets, angular_velocity, ...) -> first planet
       the shot hits (swept-pair), or sun / oob / none.

The previous physics (v1 + v2) is archived under shared/physics_archive/.
Tuple layout:
  planet = [id, owner, x, y, radius, ships, production]
  fleet  = [id, owner, x, y, angle, from_planet_id, ships]
"""
from .constants import (
    BOARD, SUN_X, SUN_Y, SUN_R, SUN_SAFETY, ROTATION_LIMIT, MAX_SPEED,
    COMET_RADIUS, EPISODE_STEPS, LAST_TURN, LAUNCH_CLEARANCE,
)
from .geom import (
    fleet_speed, dist, point_seg_dist2, seg_hits_sun, swept_pair_hit, in_board,
)
from .motion import (
    future_positions, planet_pos_after, fleet_pos_after, comet_lookup,
    P_ID, P_OWNER, P_X, P_Y, P_R, P_SHIPS, P_PROD,
    F_ID, F_OWNER, F_X, F_Y, F_ANGLE, F_FROM, F_SHIPS,
)
from .aim import aim
from .collide import first_hit, first_hit_from

__all__ = [
    "future_positions", "aim", "first_hit", "first_hit_from",
    "fleet_speed", "dist", "point_seg_dist2", "seg_hits_sun", "swept_pair_hit", "in_board",
    "planet_pos_after", "fleet_pos_after", "comet_lookup",
    "BOARD", "SUN_X", "SUN_Y", "SUN_R", "SUN_SAFETY", "ROTATION_LIMIT", "MAX_SPEED",
    "COMET_RADIUS", "EPISODE_STEPS", "LAST_TURN", "LAUNCH_CLEARANCE",
]
