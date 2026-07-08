"""Board / sun / speed constants for the pure-Python physics rewrite.

Single source of truth; values match kaggle_environments orbit_wars (1.30.1).
"""

BOARD = 100.0
SUN_X = 50.0
SUN_Y = 50.0
SUN_R = 10.0                 # env's fleet-kill radius (first_hit uses this, raw)
SUN_SAFETY = 1.5            # OUR planning margin; aim uses SUN_R + SUN_SAFETY = 11.5
ROTATION_LIMIT = 50.0       # orbital_r + radius < 50 -> planet 公转(绕 sun); else static
MAX_SPEED = 6.0             # fleet speed cap (log curve, 1..1000 ships -> 1.0..6.0)
COMET_RADIUS = 1.0
EPISODE_STEPS = 500
LAST_TURN = EPISODE_STEPS - 1   # 499: last state the game cares about (see about_the_game.md §10)
LAUNCH_CLEARANCE = 0.1     # fleet spawns at src.radius + 0.1 outside the source planet
