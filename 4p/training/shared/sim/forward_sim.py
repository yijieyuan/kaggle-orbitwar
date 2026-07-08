"""
Orbit Wars forward simulator — Python implementation matching kaggle_environments.

Rules reference: orbit-wars/README.md

Turn order (per README):
    1. Comet expiration (remove comets that left board)
    2. Comet spawning (new groups at steps 50/150/250/350/450)
    3. Fleet launch (process player actions)
    4. Production (owned planets generate ships)
    5. Fleet movement (along headings, check OOB/sun/planet collision)
    6. Planet rotation + comet movement (fleets swept into combat)
    7. Combat resolution

Notes:
- This sim does NOT generate new comet spawn paths (unknown without the
  real env). It only advances existing comets. Pass new comets in from
  the real observation when crossing a spawn boundary.
- Handles 2 or 4 player games via the ``n_players`` parameter.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Tuple

import numpy as np

# ----------------------------- Constants (from README) -----------------------------
BOARD = 100.0
SUN_X, SUN_Y = 50.0, 50.0
SUN_R = 10.0
MAX_SPEED = 6.0
SPEED_LOG_REF = math.log(1000.0)  # speed formula denominator
COMET_RADIUS = 1.0
COMET_PRODUCTION = 1
COMET_SPAWN_STEPS = {50, 150, 250, 350, 450}
ROTATION_LIMIT = 50.0  # orbital_radius + planet_radius < 50 => rotates
EPISODE_STEPS = 500


# ----------------------------- Dataclasses -----------------------------
@dataclass
class Planet:
    id: int
    owner: int          # 0..n_players-1, or -1 for neutral
    x: float
    y: float
    radius: float
    ships: int
    production: int
    # Rotation state — for orbiting planets we compute current (x,y) from
    # this, and for static planets these are unused.
    orbital_radius: float = 0.0     # distance from sun center
    orbital_angle: float = 0.0      # current angle in radians
    is_orbiting: bool = False
    # Comet-specific
    is_comet: bool = False
    comet_path: Optional[List[Tuple[float, float]]] = None   # full trajectory
    comet_path_index: int = 0


@dataclass
class Fleet:
    id: int
    owner: int
    x: float
    y: float
    angle: float        # radians; direction of travel
    ships: int
    from_planet_id: int
    # Derived / convenience
    speed: float = 0.0


# v18: fast manual clone — bypasses dataclass.replace's reflection overhead
# (~5-10x faster). Used at the top of OrbitSimulator.step where we need a
# defensive copy of the input state.
def _clone_planet(p):
    new = Planet.__new__(Planet)
    new.id = p.id
    new.owner = p.owner
    new.x = p.x
    new.y = p.y
    new.radius = p.radius
    new.ships = p.ships
    new.production = p.production
    new.orbital_radius = p.orbital_radius
    new.orbital_angle = p.orbital_angle
    new.is_orbiting = p.is_orbiting
    new.is_comet = p.is_comet
    new.comet_path = p.comet_path
    new.comet_path_index = p.comet_path_index
    return new


def _clone_fleet(f):
    new = Fleet.__new__(Fleet)
    new.id = f.id
    new.owner = f.owner
    new.x = f.x
    new.y = f.y
    new.angle = f.angle
    new.ships = f.ships
    new.from_planet_id = f.from_planet_id
    new.speed = f.speed
    return new


@dataclass
class GameState:
    planets: List[Planet]            # includes comets
    fleets: List[Fleet]
    step: int
    angular_velocity: float          # rad/turn for orbiting planets
    n_players: int
    next_fleet_id: int = 0
    # Game-over tracking
    terminated: bool = False
    reason: str = ""

    # Convenience: owners currently alive (have any planet or fleet)
    def alive_players(self) -> List[int]:
        alive = set()
        for p in self.planets:
            if p.owner >= 0:
                alive.add(p.owner)
        for f in self.fleets:
            if f.owner >= 0:
                alive.add(f.owner)
        return sorted(alive)


# ----------------------------- Physics helpers -----------------------------
def fleet_speed(ships: int) -> float:
    """Fleet speed formula from README."""
    if ships <= 1:
        return 1.0
    ratio = math.log(min(ships, 1000)) / SPEED_LOG_REF
    return 1.0 + (MAX_SPEED - 1.0) * (ratio ** 1.5)


try:
    from numba import njit as _njit  # type: ignore
    _HAS_NUMBA = True
except Exception:
    _HAS_NUMBA = False
    def _njit(*a, **kw):
        def _wrap(f):
            return f
        return _wrap


_BOARD = float(BOARD)
_SUN_X_F = float(SUN_X)
_SUN_Y_F = float(SUN_Y)
_SUN_R_F = float(SUN_R)


# Module-level flags for sim optimization tiers:
USE_NJIT_FLEET_MOVEMENT = True   # v15 Tier D  (legacy 1.28.0 path only)
USE_NJIT_SWEEP = True            # v17 Tier F  (legacy 1.28.0 path only)
USE_FAST_CLONE = True            # v18 dataclass.replace bypass

# Collision model selector (added when kaggle_environments moved 1.28.0 -> 1.30.1):
#   "1.30.1" -> swept-pair continuous collision. Fleet AND planet move
#               simultaneously within the tick (single relative-motion
#               quadratic, see _swept_pair_hit). Matches kaggle_environments
#               >= ~1.29 (orbit_wars.py::swept_pair_hit). Pure-Python.
#   "1.28.0" -> legacy two-phase one-sided model: fleet-movement-segment vs
#               STATIONARY planet, then planet-rotation-sweep vs STATIONARY
#               fleet. Uses the native-C / numba hot loops. Matches
#               kaggle_environments <= 1.28.
# The locally-installed env is 1.30.1, so that is the default. "1.28.0" is
# kept selectable in case the live competition still scores on the old engine.
COLLISION_MODEL = "1.30.1"


@_njit(cache=True, nogil=True)
def _seg_circle_scalar(x0, y0, x1, y1, cx, cy, radius):
    dx = x1 - x0
    dy = y1 - y0
    seg_len_sq = dx * dx + dy * dy
    if seg_len_sq < 1e-12:
        return (x0 - cx) * (x0 - cx) + (y0 - cy) * (y0 - cy) <= radius * radius
    t = ((cx - x0) * dx + (cy - y0) * dy) / seg_len_sq
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    closest_x = x0 + t * dx
    closest_y = y0 + t * dy
    ddx = closest_x - cx
    ddy = closest_y - cy
    return ddx * ddx + ddy * ddy <= radius * radius


@_njit(cache=True, nogil=True)
def _swept_pair_hit(ax, ay, bx, by, p0x, p0y, p1x, p1y, r):
    """Continuous swept-pair collision (kaggle_environments 1.30.1 model).

    True iff a fleet moving (ax,ay)->(bx,by) and a planet moving
    (p0x,p0y)->(p1x,p1y) come within `r` of each other for some t in [0, 1].
    Both segments are treated as linear over the tick (planet rotation
    linearised to its chord). Byte-identical to orbit_wars.py::swept_pair_hit.
    """
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


def _fleet_movement_njit(
    fx_arr, fy_arr, fspeed_arr, fangle_arr,
    px_arr, py_arr, pr_arr,
):
    """v9 lib (C): vectorized fleet movement, replaces numba njit version.

    Same return contract: (new_fx, new_fy, hit_idx).
        hit_idx[i] == -1 → survived; new_fx/new_fy valid
        hit_idx[i] == -2 → destroyed (OOB or sun)
        hit_idx[i] >= 0  → hit planets[hit_idx[i]]
    """
    import ctypes as _ct
    from shared.agent_lib.v9._binding import _lib as _v9_lib
    n = fx_arr.shape[0]
    new_fx = np.empty(n, dtype=np.float64)
    new_fy = np.empty(n, dtype=np.float64)
    hit_idx = np.full(n, -1, dtype=np.int64)
    if n == 0:
        return new_fx, new_fy, hit_idx
    _v9_lib.compute_fleet_movement_v5(
        fx_arr.ctypes.data_as(_ct.POINTER(_ct.c_double)),
        fy_arr.ctypes.data_as(_ct.POINTER(_ct.c_double)),
        fspeed_arr.ctypes.data_as(_ct.POINTER(_ct.c_double)),
        fangle_arr.ctypes.data_as(_ct.POINTER(_ct.c_double)),
        _ct.c_int(n),
        px_arr.ctypes.data_as(_ct.POINTER(_ct.c_double)),
        py_arr.ctypes.data_as(_ct.POINTER(_ct.c_double)),
        pr_arr.ctypes.data_as(_ct.POINTER(_ct.c_double)),
        _ct.c_int(px_arr.shape[0]),
        _ct.c_double(_BOARD), _ct.c_double(_SUN_X_F),
        _ct.c_double(_SUN_Y_F), _ct.c_double(_SUN_R_F),
        new_fx.ctypes.data_as(_ct.POINTER(_ct.c_double)),
        new_fy.ctypes.data_as(_ct.POINTER(_ct.c_double)),
        hit_idx.ctypes.data_as(_ct.POINTER(_ct.c_int64)),
    )
    return new_fx, new_fy, hit_idx


@_njit(cache=True, nogil=True)
def _fleet_movement_njit_LEGACY_NUMBA(
    fx_arr, fy_arr, fspeed_arr, fangle_arr,
    px_arr, py_arr, pr_arr,
):
    """OBSOLETE: kept for parity testing only. Use _fleet_movement_njit (C)."""
    n = fx_arr.shape[0]
    np_count = px_arr.shape[0]
    new_fx = np.empty(n, dtype=np.float64)
    new_fy = np.empty(n, dtype=np.float64)
    hit_idx = np.full(n, -1, dtype=np.int64)
    for i in range(n):
        fx = fx_arr[i]
        fy = fy_arr[i]
        sp = fspeed_arr[i]
        ang = fangle_arr[i]
        nx = fx + sp * math.cos(ang)
        ny = fy + sp * math.sin(ang)
        # Planet collision FIRST (matches kaggle env order)
        hit = -1
        for j in range(np_count):
            # Inlined _seg_circle_scalar for speed
            cx = px_arr[j]
            cy = py_arr[j]
            r = pr_arr[j]
            dx = nx - fx
            dy = ny - fy
            seg_len_sq = dx * dx + dy * dy
            if seg_len_sq < 1e-12:
                d2 = (fx - cx) * (fx - cx) + (fy - cy) * (fy - cy)
                if d2 <= r * r:
                    hit = j
                    break
                continue
            t = ((cx - fx) * dx + (cy - fy) * dy) / seg_len_sq
            if t < 0.0:
                t = 0.0
            elif t > 1.0:
                t = 1.0
            cxf = fx + t * dx
            cyf = fy + t * dy
            ddx = cxf - cx
            ddy = cyf - cy
            if ddx * ddx + ddy * ddy <= r * r:
                hit = j
                break
        if hit >= 0:
            hit_idx[i] = hit
            continue
        # OOB
        if not (0.0 <= nx <= _BOARD and 0.0 <= ny <= _BOARD):
            hit_idx[i] = -2
            continue
        # Sun
        dx = nx - fx
        dy = ny - fy
        seg_len_sq = dx * dx + dy * dy
        sun_hit = False
        if seg_len_sq < 1e-12:
            d2 = (fx - _SUN_X_F) ** 2 + (fy - _SUN_Y_F) ** 2
            sun_hit = d2 <= _SUN_R_F * _SUN_R_F
        else:
            t = ((_SUN_X_F - fx) * dx + (_SUN_Y_F - fy) * dy) / seg_len_sq
            if t < 0.0:
                t = 0.0
            elif t > 1.0:
                t = 1.0
            cxf = fx + t * dx
            cyf = fy + t * dy
            ddx = cxf - _SUN_X_F
            ddy = cyf - _SUN_Y_F
            sun_hit = ddx * ddx + ddy * ddy <= _SUN_R_F * _SUN_R_F
        if sun_hit:
            hit_idx[i] = -2
            continue
        new_fx[i] = nx
        new_fy[i] = ny
    return new_fx, new_fy, hit_idx


def _sweep_collisions_njit(
    fx_arr, fy_arr,
    p_old_x, p_old_y, p_new_x, p_new_y, p_radius, p_active,
):
    """v9 lib (C): vectorized sweep, replaces numba njit version."""
    import ctypes as _ct
    from shared.agent_lib.v9._binding import _lib as _v9_lib
    n = fx_arr.shape[0]
    hit_planet = np.full(n, -1, dtype=np.int64)
    if n == 0:
        return hit_planet
    p_active_i32 = np.asarray(p_active, dtype=np.int32)
    _v9_lib.compute_sweep_collisions_v5(
        fx_arr.ctypes.data_as(_ct.POINTER(_ct.c_double)),
        fy_arr.ctypes.data_as(_ct.POINTER(_ct.c_double)),
        _ct.c_int(n),
        p_old_x.ctypes.data_as(_ct.POINTER(_ct.c_double)),
        p_old_y.ctypes.data_as(_ct.POINTER(_ct.c_double)),
        p_new_x.ctypes.data_as(_ct.POINTER(_ct.c_double)),
        p_new_y.ctypes.data_as(_ct.POINTER(_ct.c_double)),
        p_radius.ctypes.data_as(_ct.POINTER(_ct.c_double)),
        p_active_i32.ctypes.data_as(_ct.POINTER(_ct.c_int)),
        _ct.c_int(p_old_x.shape[0]),
        hit_planet.ctypes.data_as(_ct.POINTER(_ct.c_int64)),
    )
    return hit_planet


@_njit(cache=True, nogil=True)
def _sweep_collisions_njit_LEGACY_NUMBA(
    fx_arr, fy_arr,
    p_old_x, p_old_y, p_new_x, p_new_y, p_radius, p_active,
):
    """OBSOLETE: kept for parity testing only."""
    n_fleets = fx_arr.shape[0]
    n_planets = p_old_x.shape[0]
    hit_planet = np.full(n_fleets, -1, dtype=np.int64)
    for j in range(n_planets):
        if not p_active[j]:
            continue
        ox = p_old_x[j]
        oy = p_old_y[j]
        nx = p_new_x[j]
        ny = p_new_y[j]
        # Skip planets that didn't move
        if ox == nx and oy == ny:
            continue
        r = p_radius[j]
        r2 = r * r
        for i in range(n_fleets):
            if hit_planet[i] >= 0:
                continue
            fx = fx_arr[i]
            fy = fy_arr[i]
            # Check 1: degenerate seg (fx,fy)-(fx,fy) vs circle (nx,ny,r)
            # i.e., is fleet's end-of-turn position within planet's NEW position?
            ddx = fx - nx
            ddy = fy - ny
            if ddx * ddx + ddy * ddy <= r2:
                hit_planet[i] = j
                continue
            # Check 2: planet rotation segment (ox,oy)→(nx,ny) vs point fleet
            sdx = nx - ox
            sdy = ny - oy
            seg_sq = sdx * sdx + sdy * sdy
            if seg_sq < 1e-12:
                continue
            t = ((fx - ox) * sdx + (fy - oy) * sdy) / seg_sq
            if t < 0.0:
                t = 0.0
            elif t > 1.0:
                t = 1.0
            cxf = ox + t * sdx
            cyf = oy + t * sdy
            ddx = fx - cxf
            ddy = fy - cyf
            if ddx * ddx + ddy * ddy <= r2:
                hit_planet[i] = j
    return hit_planet


def segment_crosses_circle(
    p0: Tuple[float, float],
    p1: Tuple[float, float],
    center: Tuple[float, float],
    radius: float,
) -> bool:
    """Continuous collision: does the segment [p0, p1] come within `radius` of `center`?
    Tuple-arg compat wrapper for callers; the numba-compiled scalar form
    `_seg_circle_scalar` is used directly in hot loops within this module.
    """
    return _seg_circle_scalar(
        p0[0], p0[1], p1[0], p1[1], center[0], center[1], radius)


def planet_position_at(planet: Planet, angular_velocity: float, step_offset: int = 0) -> Tuple[float, float]:
    """Predict a planet's (x, y) after `step_offset` more turns of rotation."""
    if not planet.is_orbiting:
        return planet.x, planet.y
    new_angle = planet.orbital_angle + angular_velocity * step_offset
    x = SUN_X + planet.orbital_radius * math.cos(new_angle)
    y = SUN_Y + planet.orbital_radius * math.sin(new_angle)
    return x, y


# ----------------------------- Core Simulator -----------------------------
class OrbitSimulator:
    """Pure-Python forward simulator. Each step returns a new GameState.

    To support comet spawn at turns 50/150/250/350/450, pass `precomputed_comets`
    (output of `shared.sim.comet_gen.precompute_all_comets`) to the constructor.
    """

    def __init__(self, precomputed_comets: dict = None):
        self.precomputed_comets = precomputed_comets

    def step(
        self,
        state: GameState,
        actions_by_player: Dict[int, List[List]],
    ) -> GameState:
        """
        actions_by_player: {player_id: [[from_planet_id, angle, ships], ...]}
        Returns a NEW GameState (input is not mutated).
        """
        # Deep-copy to avoid mutation side effects
        s = GameState(
            planets=([_clone_planet(p) for p in state.planets] if USE_FAST_CLONE
                     else [replace(p) for p in state.planets]),
            fleets=([_clone_fleet(f) for f in state.fleets] if USE_FAST_CLONE
                    else [replace(f) for f in state.fleets]),
            step=state.step,
            angular_velocity=state.angular_velocity,
            n_players=state.n_players,
            next_fleet_id=state.next_fleet_id,
        )

        next_step = s.step + 1

        # --- 1. Comet expiration (ones that left board already) ---
        s.planets = [p for p in s.planets if not self._comet_expired(p)]

        # --- 2. Comet spawning ---
        # If precomputed_comets provided, spawn at turn boundaries.
        if self.precomputed_comets is not None and next_step in COMET_SPAWN_STEPS:
            spawn_idx = list(COMET_SPAWN_STEPS).index(next_step) if next_step in {50,150,250,350,450} else None
            # Use sorted order [50,150,250,350,450] to map next_step → index
            steps_sorted = [50, 150, 250, 350, 450]
            if next_step in steps_sorted:
                spawn_idx = steps_sorted.index(next_step)
                paths = self.precomputed_comets["paths_per_spawn"][spawn_idx]
                ships = self.precomputed_comets["ships_per_spawn"][spawn_idx]
                if paths is not None:
                    next_id = max((p.id for p in s.planets), default=-1) + 1
                    for i, path in enumerate(paths):
                        comet = Planet(
                            id=next_id + i,
                            owner=-1,            # neutral
                            x=-99.0,             # off-board until first advance
                            y=-99.0,
                            radius=COMET_RADIUS,
                            ships=int(ships),
                            production=COMET_PRODUCTION,
                            orbital_radius=0.0,
                            orbital_angle=0.0,
                            is_orbiting=False,
                            is_comet=True,
                            comet_path=[(float(x), float(y)) for x, y in path],
                            comet_path_index=-1,  # advance will increment to 0
                        )
                        s.planets.append(comet)

        # --- 3. Fleet launch (process actions) ---
        planet_by_id = {p.id: p for p in s.planets}
        for player_id, action_list in actions_by_player.items():
            if action_list is None:
                continue
            for move in action_list:
                if len(move) < 3:
                    continue
                from_id, angle, ships = move[0], float(move[1]), int(move[2])
                if from_id not in planet_by_id:
                    continue  # stale id
                p = planet_by_id[from_id]
                if p.owner != player_id:
                    continue  # not your planet
                if ships <= 0 or ships > p.ships:
                    continue  # invalid
                # Subtract from source
                p.ships -= ships
                # Spawn fleet just outside planet radius
                # IMPORTANT: kaggle_environments uses +0.1 (not +1e-3); must match
                # exactly or fleet positions diverge from kaggle.
                spawn_dist = p.radius + 0.1
                fx = p.x + spawn_dist * math.cos(angle)
                fy = p.y + spawn_dist * math.sin(angle)
                new_fleet = Fleet(
                    id=s.next_fleet_id,
                    owner=player_id,
                    x=fx,
                    y=fy,
                    angle=angle,
                    ships=ships,
                    from_planet_id=from_id,
                    speed=fleet_speed(ships),
                )
                s.next_fleet_id += 1
                s.fleets.append(new_fleet)

        # --- 4. Production (owned planets add ships) ---
        for p in s.planets:
            if p.owner >= 0:
                p.ships += p.production

        # --- 5+6. Fleet movement + planet collision (model-selectable) ---
        # 1.30.1 (default): one swept-pair continuous check (fleet & planet
        #   move simultaneously), pure-Python — see _move_collide_swept.
        # 1.28.0 (legacy): two-phase one-sided, native-C/numba — see
        #   _move_collide_two_phase.
        arrivals: Dict[int, List[Fleet]] = {}
        expired_comet_ids = set()
        # 1.30.1 swept-pair is the ONLY model now (the installed env is 1.30.1). The legacy
        # 1.28.0 two-phase path + its native-C/numba helpers are retired (unreachable) — that
        # is what drops forward_sim's shared/native + shared/agent_lib.v9 dependency.
        self._move_collide_swept(s, arrivals, expired_comet_ids)

        # Remove expired comets immediately (matches kaggle line 612-626)
        if expired_comet_ids:
            s.planets = [p for p in s.planets if p.id not in expired_comet_ids]

        # --- 7. Combat resolution ---
        for planet_id, fleets_hitting in arrivals.items():
            p = planet_by_id.get(planet_id)
            if p is None:
                continue
            self._resolve_combat(p, fleets_hitting)

        # Advance step + check termination.
        # Match the kaggle env: it terminates when the interpreter's INCOMING
        # step >= episodeSteps - 2, and the core labels that DONE state with
        # step = episodeSteps - 1 (e.g. obs.step 499 is the DONE state for the
        # default episodeSteps=500). Our s.step here IS that output step, so
        # the equivalent threshold is EPISODE_STEPS - 1 (499), not 500.
        s.step = next_step
        if s.step >= EPISODE_STEPS - 1:
            s.terminated = True
            s.reason = "step limit"
        else:
            alive = s.alive_players()
            if len(alive) <= 1:
                s.terminated = True
                s.reason = f"only {len(alive)} player(s) left"

        return s

    # --- Movement + collision: 1.30.1 swept-pair (default) ---
    def _move_collide_swept(self, s, arrivals, expired_comet_ids):
        """kaggle_environments 1.30.1 collision model (pure-Python).

        Fleet and planet move SIMULTANEOUSLY within the tick: a single
        relative-motion check (_swept_pair_hit) decides collision. Planet
        end-of-tick positions are computed UP FRONT (so every fleet sees the
        same planet old->new motion) and APPLIED only AFTER collisions are
        resolved — matching orbit_wars.py step 2/3/4 ordering exactly.
        Mutates s.fleets / s.planets and fills arrivals / expired_comet_ids.
        """
        # (a) each planet's start/end position this tick: (old_x, old_y,
        #     new_x, new_y, radius, check_collision). Not applied yet.
        planet_paths = []
        for p in s.planets:
            ox, oy = p.x, p.y
            nx, ny = ox, oy
            check = True
            if p.is_orbiting:
                na = p.orbital_angle + s.angular_velocity
                nx = SUN_X + p.orbital_radius * math.cos(na)
                ny = SUN_Y + p.orbital_radius * math.sin(na)
            elif p.is_comet and p.comet_path is not None:
                nidx = p.comet_path_index + 1
                if nidx >= len(p.comet_path):
                    # Comet's path ran out: stays put this tick, removed after
                    # combat. (new == old -> tested as a stationary comet.)
                    expired_comet_ids.add(p.id)
                else:
                    nx, ny = p.comet_path[nidx]
                    # First on-board placement (old is the -99 off-board
                    # placeholder): kaggle disables the collision check.
                    check = (ox >= 0.0)
            planet_paths.append((ox, oy, nx, ny, p.radius, check))

        # (b) move each fleet; planet (swept-pair) check FIRST, then OOB, then
        #     sun — first planet in list order wins (matches env break-on-first).
        surviving_fleets: List[Fleet] = []
        n_planets = len(s.planets)
        for f in s.fleets:
            ox, oy = f.x, f.y
            nx = ox + f.speed * math.cos(f.angle)
            ny = oy + f.speed * math.sin(f.angle)
            hit_id = None
            for j in range(n_planets):
                pox, poy, pnx, pny, pr, check = planet_paths[j]
                if not check:
                    continue
                if _swept_pair_hit(ox, oy, nx, ny, pox, poy, pnx, pny, pr):
                    hit_id = s.planets[j].id
                    break
            if hit_id is not None:
                arrivals.setdefault(hit_id, []).append(f)
                continue
            if not (0.0 <= nx <= BOARD and 0.0 <= ny <= BOARD):
                continue
            if _seg_circle_scalar(ox, oy, nx, ny, SUN_X, SUN_Y, SUN_R):
                continue
            f.x, f.y = nx, ny
            surviving_fleets.append(f)
        s.fleets = surviving_fleets

        # (c) apply planet movement now that collisions are resolved.
        for j, p in enumerate(s.planets):
            _, _, pnx, pny, _, _ = planet_paths[j]
            if p.is_orbiting:
                p.orbital_angle += s.angular_velocity
            elif p.is_comet and p.comet_path is not None:
                p.comet_path_index += 1
            p.x, p.y = pnx, pny

    # --- Movement + collision: LEGACY 1.28.0 two-phase one-sided ---
    def _move_collide_two_phase(self, s, arrivals, expired_comet_ids):
        """kaggle_environments <= 1.28 collision model (native-C / numba).

        Step 5: fleet-movement-segment vs STATIONARY (pre-rotation) planet.
        Step 6: planet-rotation-sweep vs STATIONARY (already-moved) fleet.
        Two independent one-sided checks. Kept for COLLISION_MODEL=='1.28.0'
        and for parity against the old engine. Mutates s and fills arrivals /
        expired_comet_ids.
        """
        # --- 5. Fleet movement + planet/OOB/sun checks ---
        surviving_fleets: List[Fleet] = []
        if USE_NJIT_FLEET_MOVEMENT and s.fleets:
            # v15 Tier D: njit'd inner loop
            fx_arr = np.array([f.x for f in s.fleets], dtype=np.float64)
            fy_arr = np.array([f.y for f in s.fleets], dtype=np.float64)
            fspeed_arr = np.array([f.speed for f in s.fleets], dtype=np.float64)
            fangle_arr = np.array([f.angle for f in s.fleets], dtype=np.float64)
            px_arr = np.array([p.x for p in s.planets], dtype=np.float64)
            py_arr = np.array([p.y for p in s.planets], dtype=np.float64)
            pr_arr = np.array([p.radius for p in s.planets], dtype=np.float64)
            new_fx, new_fy, hit_idx = _fleet_movement_njit(
                fx_arr, fy_arr, fspeed_arr, fangle_arr,
                px_arr, py_arr, pr_arr,
            )
            for i, f in enumerate(s.fleets):
                h = int(hit_idx[i])
                if h == -2:
                    continue
                if h >= 0:
                    pid = s.planets[h].id
                    arrivals.setdefault(pid, []).append(f)
                    continue
                f.x = float(new_fx[i]); f.y = float(new_fy[i])
                surviving_fleets.append(f)
        else:
            # v11/v12/v13/v14 LEGACY: pure Python loop
            for f in s.fleets:
                fx = f.x; fy = f.y
                new_x = fx + f.speed * math.cos(f.angle)
                new_y = fy + f.speed * math.sin(f.angle)
                hit_planet_id = None
                for p in s.planets:
                    if _seg_circle_scalar(fx, fy, new_x, new_y, p.x, p.y, p.radius):
                        hit_planet_id = p.id
                        break
                if hit_planet_id is not None:
                    arrivals.setdefault(hit_planet_id, []).append(f)
                    continue
                if not (0 <= new_x <= BOARD and 0 <= new_y <= BOARD):
                    continue
                if _seg_circle_scalar(fx, fy, new_x, new_y, SUN_X, SUN_Y, SUN_R):
                    continue
                f.x, f.y = new_x, new_y
                surviving_fleets.append(f)
        s.fleets = surviving_fleets

        # --- 6. Planet rotation + comet movement, sweep fleets ---
        if USE_NJIT_SWEEP and len(s.planets) > 0:
            # v17 Tier F: njit'd sweep
            n_planets = len(s.planets)
            p_old_x = np.empty(n_planets, dtype=np.float64)
            p_old_y = np.empty(n_planets, dtype=np.float64)
            p_new_x = np.empty(n_planets, dtype=np.float64)
            p_new_y = np.empty(n_planets, dtype=np.float64)
            p_radius_arr = np.empty(n_planets, dtype=np.float64)
            p_active = np.ones(n_planets, dtype=np.int64)
            for j, p in enumerate(s.planets):
                p_old_x[j] = p.x; p_old_y[j] = p.y
                if p.is_orbiting:
                    p.orbital_angle += s.angular_velocity
                    p.x = SUN_X + p.orbital_radius * math.cos(p.orbital_angle)
                    p.y = SUN_Y + p.orbital_radius * math.sin(p.orbital_angle)
                elif p.is_comet and p.comet_path is not None:
                    p.comet_path_index += 1
                    if p.comet_path_index < len(p.comet_path):
                        p.x, p.y = p.comet_path[p.comet_path_index]
                    else:
                        expired_comet_ids.add(p.id)
                        p_active[j] = 0
                p_new_x[j] = p.x; p_new_y[j] = p.y
                p_radius_arr[j] = p.radius

            if s.fleets:
                fx_arr = np.array([f.x for f in s.fleets], dtype=np.float64)
                fy_arr = np.array([f.y for f in s.fleets], dtype=np.float64)
                hit_planet = _sweep_collisions_njit(
                    fx_arr, fy_arr,
                    p_old_x, p_old_y, p_new_x, p_new_y, p_radius_arr, p_active,
                )
                remaining_fleets = []
                for i, f in enumerate(s.fleets):
                    h = int(hit_planet[i])
                    if h >= 0:
                        arrivals.setdefault(s.planets[h].id, []).append(f)
                    else:
                        remaining_fleets.append(f)
                s.fleets = remaining_fleets
        else:
            # v11..v15 LEGACY: pure Python sweep loop
            for p in s.planets:
                old_pos = (p.x, p.y)
                if p.is_orbiting:
                    p.orbital_angle += s.angular_velocity
                    p.x = SUN_X + p.orbital_radius * math.cos(p.orbital_angle)
                    p.y = SUN_Y + p.orbital_radius * math.sin(p.orbital_angle)
                elif p.is_comet and p.comet_path is not None:
                    p.comet_path_index += 1
                    if p.comet_path_index < len(p.comet_path):
                        p.x, p.y = p.comet_path[p.comet_path_index]
                    else:
                        expired_comet_ids.add(p.id)
                        continue
                new_x = p.x; new_y = p.y
                old_x, old_y = old_pos
                if old_pos != (new_x, new_y):
                    remaining = []
                    p_radius = p.radius
                    for f in s.fleets:
                        fx = f.x; fy = f.y
                        if _seg_circle_scalar(fx, fy, fx, fy, new_x, new_y, p_radius) or \
                           _seg_circle_scalar(old_x, old_y, new_x, new_y, fx, fy, p_radius):
                            arrivals.setdefault(p.id, []).append(f)
                        else:
                            remaining.append(f)
                    s.fleets = remaining

    # --- Combat resolution ---
    @staticmethod
    def _resolve_combat(planet: Planet, incoming: List[Fleet]) -> None:
        """Mutates planet in-place per README combat rules."""
        if not incoming:
            return
        # Group by owner, sum ships
        by_owner: Dict[int, int] = {}
        for f in incoming:
            by_owner[f.owner] = by_owner.get(f.owner, 0) + f.ships
        sorted_groups = sorted(by_owner.items(), key=lambda x: -x[1])

        if len(sorted_groups) == 1:
            surv_owner, surv_ships = sorted_groups[0]
        else:
            top_owner, top_ships = sorted_groups[0]
            sec_owner, sec_ships = sorted_groups[1]
            if top_ships == sec_ships:
                # Two-way tie: all attackers destroyed (README rule 4)
                return
            surv_owner = top_owner
            surv_ships = top_ships - sec_ships
            # Third+ ranked attackers are also destroyed (README: "a surviving attacker", singular)

        # Survivor meets garrison
        if surv_owner == planet.owner:
            planet.ships += surv_ships
        else:
            if surv_ships > planet.ships:
                planet.owner = surv_owner
                planet.ships = surv_ships - planet.ships
            else:
                planet.ships -= surv_ships

    # --- Comet expiration check ---
    # Used at section 1 (start of step) to clean up. With the in-step removal
    # in section 6, this is mostly redundant — but kept as a safety net for
    # comets that might get into the state from external injection (e.g.
    # parity_test's inject_new_comets) with already-expired path_index.
    # Matches kaggle's logic: only path_index >= len. NO off-board check
    # (kaggle's path generator can include slightly off-board positions
    # during entry/exit; we don't want to expire those).
    @staticmethod
    def _comet_expired(p: Planet) -> bool:
        if not p.is_comet:
            return False
        if p.comet_path is None:
            return False
        if p.comet_path_index >= len(p.comet_path):
            return True
        return False


# ----------------------------- Conversion from kaggle_environments obs -----------------------------
def from_kaggle_obs(obs, n_players: int = 2) -> GameState:
    """Convert a kaggle_environments observation dict into a GameState."""
    # obs.planets: list of [id, owner, x, y, radius, ships, production]
    # obs.fleets:  list of [id, owner, x, y, angle, from_planet_id, ships]
    # obs.angular_velocity, obs.initial_planets, obs.comets, obs.comet_planet_ids
    # Obs may be a dict or a named attribute object; handle both.
    def get(o, key, default=None):
        if isinstance(o, dict):
            return o.get(key, default)
        return getattr(o, key, default)

    raw_planets = get(obs, "planets", [])
    raw_fleets = get(obs, "fleets", [])
    ang_vel = get(obs, "angular_velocity", 0.0)
    comet_planet_ids = set(get(obs, "comet_planet_ids", []) or [])
    initial_planets = get(obs, "initial_planets", raw_planets)
    comets_info = get(obs, "comets", []) or []
    # Build id -> initial planet for orbital info
    init_by_id = {ip[0]: ip for ip in initial_planets}

    # Comet path lookup
    comet_path_by_id: Dict[int, Tuple[List[Tuple[float, float]], int]] = {}
    for group in comets_info:
        ids = group.get("planet_ids", [])
        paths = group.get("paths", [])
        idx = group.get("path_index", 0)
        for pid, path in zip(ids, paths):
            comet_path_by_id[pid] = (path, idx)

    planets: List[Planet] = []
    for pt in raw_planets:
        pid, owner, x, y, radius, ships, production = pt[:7]
        is_comet = pid in comet_planet_ids
        # Derive orbital info from initial position if rotating
        init = init_by_id.get(pid)
        is_orbiting = False
        orbital_radius = 0.0
        orbital_angle = 0.0
        if init is not None and not is_comet:
            _, _, ix, iy, iradius, _, _ = init[:7]
            dx, dy = ix - SUN_X, iy - SUN_Y
            orbital_radius = math.hypot(dx, dy)
            if orbital_radius + iradius < ROTATION_LIMIT:
                is_orbiting = True
                # current angle derived from current x,y
                orbital_angle = math.atan2(y - SUN_Y, x - SUN_X)

        comet_path: Optional[List[Tuple[float, float]]] = None
        comet_path_index = 0
        if is_comet and pid in comet_path_by_id:
            path, idx = comet_path_by_id[pid]
            comet_path = [tuple(pt) for pt in path]
            comet_path_index = idx

        planets.append(Planet(
            id=pid,
            owner=int(owner),
            x=float(x),
            y=float(y),
            radius=float(radius),
            ships=int(ships),
            production=int(production),
            orbital_radius=orbital_radius,
            orbital_angle=orbital_angle,
            is_orbiting=is_orbiting,
            is_comet=is_comet,
            comet_path=comet_path,
            comet_path_index=comet_path_index,
        ))

    fleets: List[Fleet] = []
    for ft in raw_fleets:
        fid, owner, x, y, angle, from_id, ships = ft[:7]
        fleets.append(Fleet(
            id=int(fid),
            owner=int(owner),
            x=float(x),
            y=float(y),
            angle=float(angle),
            ships=int(ships),
            from_planet_id=int(from_id),
            speed=fleet_speed(int(ships)),
        ))

    # next_fleet_id is kaggle env's GLOBAL counter (incremented per launch,
    # never decremented when fleets die). The kaggle env always populates
    # obs.next_fleet_id (orbit_wars.json schema, default 0; orbit_wars.py
    # propagates it to every agent obs at end-of-step). Read it directly.
    nfid_obs = get(obs, "next_fleet_id", None)
    if nfid_obs is None:
        raise ValueError(
            "from_kaggle_obs: obs is missing `next_fleet_id`. This field is "
            "always set by the kaggle orbit_wars env — passing a non-kaggle "
            "obs is unsupported."
        )

    return GameState(
        planets=planets,
        fleets=fleets,
        step=int(get(obs, "step", 0) or 0),
        angular_velocity=float(ang_vel or 0.0),
        n_players=n_players,
        next_fleet_id=int(nfid_obs),
    )


def score_for_player(state: GameState, player_id: int) -> int:
    """Sum of ships on owned planets + ships in flight."""
    s = 0
    for p in state.planets:
        if p.owner == player_id:
            s += p.ships
    for f in state.fleets:
        if f.owner == player_id:
            s += f.ships
    return s
