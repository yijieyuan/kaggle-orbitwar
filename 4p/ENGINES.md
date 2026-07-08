# 4p game engines + parity tests

Three interchangeable implementations of the Orbit-Wars 4p game dynamics:

| engine  | where | role |
|---------|-------|------|
| official | `kaggle_environments.make("orbit_wars")` (pip, v1.30.1) | ground truth; what Kaggle scores |
| jax      | `training/jax_env/` | vectorized/jit engine used for self-play PPO training |
| numpy    | `inference/engine.py` | pure-numpy deploy engine (also drives the merge rollout) |

Parity tests (they assert the three agree; run from a checkout with the bundled `shared/` under
`training/shared/`):

- `training/decode_parity.py`      -- numpy port vs the jax model, action/feature match to f32
- `training/jax_env/parity_test.py` -- jax engine step vs the numpy reference physics

The deploy port passed these at submission time (numpy == jax bit-exact on the bundled weights).
