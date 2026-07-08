"""Bundled subset of shared.sim for the standalone solution:
comet_gen (comet path precompute, used by the jax engine) + forward_sim (numpy reference oracle,
used by the parity tests). The full repo's kaggle_wrapper/runner are NOT needed here."""
from .forward_sim import OrbitSimulator, GameState, Planet, Fleet, fleet_speed  # noqa: F401
