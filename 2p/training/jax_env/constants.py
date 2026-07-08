"""JAX 2p forward-sim constants — kaggle_environments 1.30.1 (swept-pair collision).
Ported from shared/sim/forward_sim.py (the verified 1.30.1 oracle). Used for self-play
training-data generation; parity-tested vs forward_sim @1.30.1 (see parity_test.py).
"""
import math

# --- game constants (must match forward_sim @1.30.1) ---
BOARD = 100.0
SUN_X = 50.0
SUN_Y = 50.0
SUN_R = 10.0
MAX_SPEED = 6.0
COMET_RADIUS = 1.0
COMET_PRODUCTION = 1
ROTATION_LIMIT = 50.0          # orbital_radius + planet_radius < 50 => orbiting
EPISODE_STEPS = 500
LAST_TURN = 499
LAUNCH_CLEARANCE = 0.1         # fleet spawns at planet.radius + 0.1 (must match kaggle exactly)
COMET_SPAWN_STEPS = (50, 150, 250, 350, 450)

# --- fixed array sizes (the env is fixed-shape for jit/vmap; masks handle the rest) ---
MAX_PLANETS = 48               # real ≤40 + ≤4 comets = ≤44; 48 leaves headroom
FLEET_CAP_PER_PLAYER = 128     # exp24: per-player IN-FLIGHT cap (TRAIN-ONLY regularizer; eval/deploy uncapped)
MAX_FLEETS = 2 * FLEET_CAP_PER_PLAYER   # = 256 (per-player cap bounds total in-flight; was 512 uncapped)
MAX_COMET_PATH = 64            # a comet's trajectory length (visible 5–40 + entry/exit)
N_COMET_SLOTS = 4              # comets spawn in batches of 4

_LOG1000 = math.log(1000.0)
