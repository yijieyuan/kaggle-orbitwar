"""Comet spawn for the JAX env (kaggle 1.30.1).

- gen_schedule / attach_schedule: build the per-game comet schedule (5 spawn batches ×
  4 paths) from shared/sim/comet_gen.precompute_all_comets and store it in the JaxState.
- spawn_inject: jit-able; at turns 50/150/250/350/450 it writes the batch's 4 comets into
  the reserved comet slots [MAX_PLANETS-4 .. MAX_PLANETS) with idx=-1 / x=y=-99
  (placed on the next move, collision disabled on first placement) — matching forward_sim.
"""
import os, sys
import numpy as np
import jax.numpy as jnp

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))  # 2p/training (bundled shared/ lives here; up 1)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from constants import (MAX_PLANETS, MAX_COMET_PATH, N_COMET_SLOTS, COMET_SPAWN_STEPS,
                       COMET_RADIUS, COMET_PRODUCTION)
from state import NUM_SPAWN

_CS = MAX_PLANETS - N_COMET_SLOTS    # first reserved comet slot index
_SPAWN = jnp.asarray(COMET_SPAWN_STEPS, dtype=jnp.int32)


def gen_schedule(initial_planets, angular_velocity, seed, L=MAX_COMET_PATH):
    """Host-side: precompute the comet schedule for one game. Returns np arrays
    (sched_px, sched_py, sched_len, sched_ships) padded to L."""
    from shared.sim.comet_gen import precompute_all_comets
    pc = precompute_all_comets(initial_planets, float(angular_velocity), int(seed))
    NS, NC = NUM_SPAWN, N_COMET_SLOTS
    px = np.zeros((NS, NC, L), np.float32); py = np.zeros((NS, NC, L), np.float32)
    ln = np.zeros((NS, NC), np.int32); sh = np.zeros((NS,), np.int32)
    for b in range(NS):
        sh[b] = int(pc["ships_per_spawn"][b])
        paths = pc["paths_per_spawn"][b]
        if paths is None:
            continue
        for c in range(min(len(paths), NC)):
            path = paths[c]; n = min(len(path), L)
            for t in range(n):
                px[b, c, t] = float(path[t][0]); py[b, c, t] = float(path[t][1])
            ln[b, c] = n
    return dict(sched_px=px, sched_py=py, sched_len=ln, sched_ships=sh)


def attach_schedule(state, sched):
    """Return state with the comet schedule arrays attached."""
    return state._replace(
        sched_px=jnp.asarray(sched["sched_px"]), sched_py=jnp.asarray(sched["sched_py"]),
        sched_len=jnp.asarray(sched["sched_len"]), sched_ships=jnp.asarray(sched["sched_ships"]),
    )


def spawn_inject(state):
    """If state.step+1 is a comet-spawn turn, write the batch's comets into the
    reserved comet slots. Pure jnp / jit-able. Called at the top of step()."""
    nxt = state.step + 1
    is_spawn = jnp.any(nxt == _SPAWN)
    sidx = jnp.argmax(nxt == _SPAWN)                       # 0..4 (valid iff is_spawn)
    blen = state.sched_len[sidx]                           # (NC,)
    bsh = state.sched_ships[sidx]                          # ()
    bpx = state.sched_px[sidx]                             # (NC, L)
    bpy = state.sched_py[sidx]
    v = is_spawn & (blen > 0)                              # (NC,) which slots activate
    ids = state.comet_base_id + jnp.arange(N_COMET_SLOTS, dtype=jnp.int32)
    sl = slice(_CS, _CS + N_COMET_SLOTS)

    def s(arr, new):                                       # set scalar-per-slot field
        return arr.at[sl].set(jnp.where(v, new, arr[sl]))

    return state._replace(
        p_id=s(state.p_id, ids),
        p_owner=s(state.p_owner, jnp.int32(-1)),
        p_x=s(state.p_x, jnp.float32(-99.0)),
        p_y=s(state.p_y, jnp.float32(-99.0)),
        p_radius=s(state.p_radius, jnp.float32(COMET_RADIUS)),
        p_ships=s(state.p_ships, bsh),
        p_prod=s(state.p_prod, jnp.int32(COMET_PRODUCTION)),
        p_mask=s(state.p_mask, True),
        p_is_comet=s(state.p_is_comet, True),
        p_is_orbiting=s(state.p_is_orbiting, False),
        p_orbital_r=s(state.p_orbital_r, jnp.float32(0.0)),
        p_orbital_a=s(state.p_orbital_a, jnp.float32(0.0)),
        p_comet_idx=s(state.p_comet_idx, jnp.int32(-1)),
        p_comet_len=s(state.p_comet_len, blen),
        p_comet_path_x=state.p_comet_path_x.at[sl].set(jnp.where(v[:, None], bpx, state.p_comet_path_x[sl])),
        p_comet_path_y=state.p_comet_path_y.at[sl].set(jnp.where(v[:, None], bpy, state.p_comet_path_y[sl])),
    )
