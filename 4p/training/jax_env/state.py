"""JaxState — fixed-shape, vmap/scan-able game state for the JAX 2p forward sim.

A NamedTuple of jnp arrays (so it is a JAX pytree: jit/vmap/scan work directly).
`from_kaggle_obs` builds one host-side from a kaggle observation (mirrors
shared/sim/forward_sim.from_kaggle_obs) for reset + parity. Slots are filled in obs
order; inactive slots are masked. Ints are int32, positions float32.
"""
from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np
import jax.numpy as jnp

from constants import (
    SUN_X, SUN_Y, ROTATION_LIMIT, MAX_SPEED, MAX_PLANETS, MAX_FLEETS,
    MAX_COMET_PATH, N_COMET_SLOTS, COMET_SPAWN_STEPS, _LOG1000,
)

NUM_SPAWN = len(COMET_SPAWN_STEPS)   # 5 batches (turns 50/150/250/350/450)


class JaxState(NamedTuple):
    # --- planets (incl. comets); index = fixed slot, p_mask marks active ---
    p_id: jnp.ndarray           # (P,) int32  (planet id; for action decode + parity match)
    p_owner: jnp.ndarray        # (P,) int32  (-1 neutral, 0/1 players)
    p_x: jnp.ndarray            # (P,) f32
    p_y: jnp.ndarray            # (P,) f32
    p_radius: jnp.ndarray       # (P,) f32
    p_ships: jnp.ndarray        # (P,) int32
    p_prod: jnp.ndarray         # (P,) int32
    p_mask: jnp.ndarray         # (P,) bool  (active planet slot)
    p_is_comet: jnp.ndarray     # (P,) bool
    p_is_orbiting: jnp.ndarray  # (P,) bool
    p_orbital_r: jnp.ndarray    # (P,) f32
    p_orbital_a: jnp.ndarray    # (P,) f32  (current angle)
    p_comet_path_x: jnp.ndarray # (P, MAX_COMET_PATH) f32
    p_comet_path_y: jnp.ndarray # (P, MAX_COMET_PATH) f32
    p_comet_idx: jnp.ndarray    # (P,) int32  (current index into the path)
    p_comet_len: jnp.ndarray    # (P,) int32  (path length; expire when idx >= len)
    # --- fleets ---
    f_owner: jnp.ndarray        # (F,) int32
    f_x: jnp.ndarray            # (F,) f32
    f_y: jnp.ndarray            # (F,) f32
    f_angle: jnp.ndarray        # (F,) f32
    f_ships: jnp.ndarray        # (F,) int32
    f_from: jnp.ndarray         # (F,) int32
    f_speed: jnp.ndarray        # (F,) f32
    f_mask: jnp.ndarray         # (F,) bool
    f_target: jnp.ndarray       # (F,) int32  (target planet SLOT index; -1 = unknown)
    f_arrival: jnp.ndarray      # (F,) int32  (absolute arrival turn; -1 = unknown)
    # NOTE: f_target/f_arrival are projection metadata, valid ONLY where f_mask is True
    # (stale in dead/consumed slots). Always AND with f_mask when reading (see env._forecast).
    # --- comet spawn schedule (per game; injected at turns 50/150/250/350/450) ---
    sched_px: jnp.ndarray       # (NUM_SPAWN, N_COMET_SLOTS, MAX_COMET_PATH) f32
    sched_py: jnp.ndarray       # (NUM_SPAWN, N_COMET_SLOTS, MAX_COMET_PATH) f32
    sched_len: jnp.ndarray      # (NUM_SPAWN, N_COMET_SLOTS) int32  (0 => no comet)
    sched_ships: jnp.ndarray    # (NUM_SPAWN,) int32
    comet_base_id: jnp.ndarray  # () int32  (spawned comet ids = base + 0..3, reused each spawn)
    # --- scalars ---
    step: jnp.ndarray           # () int32
    av: jnp.ndarray             # () f32  (angular velocity, rad/turn)
    next_fleet_id: jnp.ndarray  # () int32


def fleet_speed_np(ships):
    """Log speed curve (matches geom.fleet_speed / forward_sim)."""
    n = int(ships)
    if n <= 1:
        return 1.0
    if n > 1000:
        n = 1000
    return 1.0 + (MAX_SPEED - 1.0) * (math.log(n) / _LOG1000) ** 1.5


def fleet_speed_jax(ships):
    """Vectorized log speed curve for jnp arrays."""
    n = jnp.clip(ships.astype(jnp.float32), 1.0, 1000.0)
    sp = 1.0 + (MAX_SPEED - 1.0) * jnp.power(jnp.log(n) / _LOG1000, 1.5)
    return jnp.where(ships <= 1, jnp.float32(1.0), sp)


def _get(o, key, default=None):
    if isinstance(o, dict):
        return o.get(key, default)
    return getattr(o, key, default)


def from_kaggle_obs(obs, P=MAX_PLANETS, F=MAX_FLEETS, L=MAX_COMET_PATH) -> JaxState:
    """Build a JaxState from a kaggle observation (host-side, numpy → jnp).

    Mirrors forward_sim.from_kaggle_obs: orbital params derived from initial_planets;
    comet paths from obs.comets; non-comet planets that orbit get is_orbiting.
    """
    planets = list(_get(obs, "planets", []) or [])
    fleets = list(_get(obs, "fleets", []) or [])
    av = float(_get(obs, "angular_velocity", 0.0) or 0.0)
    step = int(_get(obs, "step", 0) or 0)
    comet_ids = set(_get(obs, "comet_planet_ids", []) or [])
    initial = _get(obs, "initial_planets", planets) or planets
    init_by_id = {ip[0]: ip for ip in initial}
    comets_info = _get(obs, "comets", []) or []
    nfid = _get(obs, "next_fleet_id", None)
    if nfid is None:
        raise ValueError("from_kaggle_obs: obs missing next_fleet_id (kaggle always sets it)")

    # comet path lookup: pid -> (path[list of (x,y)], idx)
    cpath = {}
    for grp in comets_info:
        ids = grp.get("planet_ids", []) if isinstance(grp, dict) else _get(grp, "planet_ids", [])
        paths = grp.get("paths", []) if isinstance(grp, dict) else _get(grp, "paths", [])
        idx = grp.get("path_index", 0) if isinstance(grp, dict) else _get(grp, "path_index", 0)
        for pid, path in zip(ids, paths):
            cpath[pid] = (path, idx)

    # planet arrays
    p_id = np.full(P, -1, np.int32)
    p_owner = np.full(P, -1, np.int32); p_x = np.zeros(P, np.float32); p_y = np.zeros(P, np.float32)
    p_radius = np.zeros(P, np.float32); p_ships = np.zeros(P, np.int32); p_prod = np.zeros(P, np.int32)
    p_mask = np.zeros(P, bool); p_is_comet = np.zeros(P, bool); p_is_orbiting = np.zeros(P, bool)
    p_orbital_r = np.zeros(P, np.float32); p_orbital_a = np.zeros(P, np.float32)
    p_cpx = np.zeros((P, L), np.float32); p_cpy = np.zeros((P, L), np.float32)
    p_cidx = np.zeros(P, np.int32); p_clen = np.zeros(P, np.int32)

    for i, pt in enumerate(planets):
        if i >= P:
            raise ValueError(f"more planets ({len(planets)}) than MAX_PLANETS={P}")
        pid, owner, x, y, radius, ships, production = pt[:7]
        is_comet = pid in comet_ids
        p_id[i] = int(pid)
        p_owner[i] = int(owner); p_x[i] = float(x); p_y[i] = float(y)
        p_radius[i] = float(radius); p_ships[i] = int(ships); p_prod[i] = int(production)
        p_mask[i] = True; p_is_comet[i] = is_comet
        init = init_by_id.get(pid)
        if init is not None and not is_comet:
            ix, iy, iradius = init[2], init[3], init[4]
            orb_r = math.hypot(ix - SUN_X, iy - SUN_Y)
            p_orbital_r[i] = orb_r
            if orb_r + iradius < ROTATION_LIMIT:
                p_is_orbiting[i] = True
                p_orbital_a[i] = math.atan2(float(y) - SUN_Y, float(x) - SUN_X)
        if is_comet and pid in cpath:
            path, idx = cpath[pid]
            n = min(len(path), L)
            for j in range(n):
                p_cpx[i, j] = float(path[j][0]); p_cpy[i, j] = float(path[j][1])
            p_cidx[i] = int(idx); p_clen[i] = len(path)

    # fleet arrays
    f_owner = np.zeros(F, np.int32); f_x = np.zeros(F, np.float32); f_y = np.zeros(F, np.float32)
    f_angle = np.zeros(F, np.float32); f_ships = np.zeros(F, np.int32); f_from = np.full(F, -1, np.int32)
    f_speed = np.zeros(F, np.float32); f_mask = np.zeros(F, bool)
    f_target = np.full(F, -1, np.int32); f_arrival = np.full(F, -1, np.int32)
    for i, ft in enumerate(fleets):
        if i >= F:
            raise ValueError(f"more fleets ({len(fleets)}) than MAX_FLEETS={F}")
        fid, owner, x, y, angle, from_id, ships = ft[:7]
        f_owner[i] = int(owner); f_x[i] = float(x); f_y[i] = float(y); f_angle[i] = float(angle)
        f_ships[i] = int(ships); f_from[i] = int(from_id); f_speed[i] = fleet_speed_np(int(ships))
        f_mask[i] = True

    # comet spawn schedule: empty by default; attach via comet.gen_schedule for training
    sched_px = np.zeros((NUM_SPAWN, N_COMET_SLOTS, L), np.float32)
    sched_py = np.zeros((NUM_SPAWN, N_COMET_SLOTS, L), np.float32)
    sched_len = np.zeros((NUM_SPAWN, N_COMET_SLOTS), np.int32)
    sched_ships = np.zeros((NUM_SPAWN,), np.int32)
    real_ids = [int(p_id[i]) for i in range(P) if p_mask[i] and not p_is_comet[i]]
    comet_base = (max(real_ids) + 1) if real_ids else 0

    j = jnp.asarray
    return JaxState(
        p_id=j(p_id),
        p_owner=j(p_owner), p_x=j(p_x), p_y=j(p_y), p_radius=j(p_radius), p_ships=j(p_ships),
        p_prod=j(p_prod), p_mask=j(p_mask), p_is_comet=j(p_is_comet), p_is_orbiting=j(p_is_orbiting),
        p_orbital_r=j(p_orbital_r), p_orbital_a=j(p_orbital_a), p_comet_path_x=j(p_cpx),
        p_comet_path_y=j(p_cpy), p_comet_idx=j(p_cidx), p_comet_len=j(p_clen),
        f_owner=j(f_owner), f_x=j(f_x), f_y=j(f_y), f_angle=j(f_angle), f_ships=j(f_ships),
        f_from=j(f_from), f_speed=j(f_speed), f_mask=j(f_mask),
        f_target=j(f_target), f_arrival=j(f_arrival),
        sched_px=j(sched_px), sched_py=j(sched_py), sched_len=j(sched_len),
        sched_ships=j(sched_ships), comet_base_id=jnp.int32(comet_base),
        step=jnp.int32(step), av=jnp.float32(av), next_fleet_id=jnp.int32(int(nfid)),
    )
