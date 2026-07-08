"""Capacity and physics constants for the game (must match the engine)."""

# ---------------------------------------------------------------------------
# Capacity — tune based on GPU memory and profiling
# ---------------------------------------------------------------------------
B_DEFAULT: int = 1024   # default games per batch
P_MAX: int = 64         # planet slots per game  (real games have 24-52 planets)
F_MAX: int = 256        # fleet slots per game
A: int = 2              # players per game

# ---------------------------------------------------------------------------
# Physics (must match the game engine)
# ---------------------------------------------------------------------------
BOARD_SIZE: float = 100.0
CENTER: float = 50.0
SUN_RADIUS: float = 10.0
MAX_SHIP_SPEED: float = 6.0
ROT_RADIUS_LIMIT: float = 50.0  # planets with orbital_radius + radius < this orbit

# ---------------------------------------------------------------------------
# Observation — relative ownership encoding
# ---------------------------------------------------------------------------
OWN: int = 0      # slot belongs to the observing player
ENEMY: int = 1    # slot belongs to an opponent
NEUTRAL: int = 2  # slot is unclaimed
DEAD: int = 3     # slot is empty (alive_mask=False)

# ---------------------------------------------------------------------------
# Library
# ---------------------------------------------------------------------------
LIBRARY_K_DEFAULT: int = 100_000  # number of starting states to pre-generate

# ---------------------------------------------------------------------------
# Comets (optional, gated by comets_enabled)
# ---------------------------------------------------------------------------
COMET_EVENTS: int = 5
COMETS_PER_EVENT: int = 4
COMET_PATH_MAX: int = 40
COMET_SPAWN_STEPS: tuple[int, ...] = (50, 150, 250, 350, 450)
COMET_RADIUS: float = 1.0
COMET_PRODUCTION: float = 1.0

# ---------------------------------------------------------------------------
# Early termination — call the game when one player dominates the leaderboard
#
# Calibrated on 535 Kaggle replays (scripts/analyze_early_termination.py).
# 2p: 100% accuracy over 242 games, saves ~48 turns / game (~30%).
# 4p: 100% accuracy on triggered (286/293) games, saves ~59 turns / game.
# ---------------------------------------------------------------------------
EARLY_TERM_MARGIN: float = 2.0          # leader_score >= MARGIN * runner_up_score
EARLY_TERM_STREAK_2P: int = 5           # consecutive turns the lead must hold
EARLY_TERM_STREAK_4P: int = 20
EARLY_TERM_PROD_WEIGHT_2P: float = 5.0  # score = 5 * production + 1 * (planet + fleet ships)
EARLY_TERM_SHIP_WEIGHT_2P: float = 1.0
EARLY_TERM_PROD_WEIGHT_4P: float = 1.0  # 4p uses production alone
EARLY_TERM_SHIP_WEIGHT_4P: float = 0.0

# ---------------------------------------------------------------------------
# Episode length (default number of game steps)
# ---------------------------------------------------------------------------
DEFAULT_EPISODE_STEPS: int = 500
