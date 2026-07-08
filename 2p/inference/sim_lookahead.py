"""Self-contained 1-ply forward-sim for the SEARCH 2p agent (deploy-safe).

VENDORED, byte-faithful subset of shared/sim/forward_sim.py (OrbitSimulator 1.30.1
swept-pair model, Planet/Fleet/GameState, from_kaggle_obs) + shared/sim/runner.py
(state_to_obs). Pure math+numpy — NO shared.sim, NO native/numba (the legacy 1.28.0
two-phase path that needed those is dropped; the installed/scoring env is 1.30.1 and
the swept path is the ONLY one forward_sim reaches).

Parity: sim_lookahead.OrbitSimulator.step / from_kaggle_obs / state_to_obs are a
literal copy of the corresponding shared.sim functions (swept branch only). A parity
harness (parity_check.py) verifies byte/f32-equality vs shared.sim on real states.

Used by rl_agent.py's 1-ply value lookahead: from_kaggle_obs(obs) -> sim.step({me,opp})
-> state_to_obs -> rl_infer.value_of. NOT used by the greedy path.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Tuple

# ----------------------------- Constants (from README; match forward_sim.py) -----------------------------
BOARD = 100.0
SUN_X, SUN_Y = 50.0, 50.0
SUN_R = 10.0
MAX_SPEED = 6.0
SPEED_LOG_REF = math.log(1000.0)
COMET_RADIUS = 1.0
COMET_PRODUCTION = 1
COMET_SPAWN_STEPS = {50, 150, 250, 350, 450}
ROTATION_LIMIT = 50.0
EPISODE_STEPS = 500


# ----------------------------- Dataclasses -----------------------------
@dataclass
class Planet:
    id: int
    owner: int
    x: float
    y: float
    radius: float
    ships: int
    production: int
    orbital_radius: float = 0.0
    orbital_angle: float = 0.0
    is_orbiting: bool = False
    is_comet: bool = False
    comet_path: Optional[List[Tuple[float, float]]] = None
    comet_path_index: int = 0


@dataclass
class Fleet:
    id: int
    owner: int
    x: float
    y: float
    angle: float
    ships: int
    from_planet_id: int
    speed: float = 0.0


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
    planets: List[Planet]
    fleets: List[Fleet]
    step: int
    angular_velocity: float
    n_players: int
    next_fleet_id: int = 0
    terminated: bool = False
    reason: str = ""

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
    if ships <= 1:
        return 1.0
    ratio = math.log(min(ships, 1000)) / SPEED_LOG_REF
    return 1.0 + (MAX_SPEED - 1.0) * (ratio ** 1.5)


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


def _swept_pair_hit(ax, ay, bx, by, p0x, p0y, p1x, p1y, r):
    """Continuous swept-pair collision (kaggle_environments 1.30.1 model).
    Byte-identical to orbit_wars.py::swept_pair_hit / forward_sim._swept_pair_hit."""
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


# ----------------------------- Core Simulator (1.30.1 swept-only) -----------------------------
class OrbitSimulator:
    """Pure-Python forward simulator (1.30.1 swept-pair model). Each step returns a NEW GameState.

    Comet SPAWNING is unsupported here (precomputed_comets always None) — a 1-ply lookahead
    starting from the CURRENT obs never crosses a spawn boundary within its single step, and
    existing comets are advanced along their cached paths exactly as in forward_sim."""

    def __init__(self, precomputed_comets: dict = None):
        self.precomputed_comets = precomputed_comets

    def step(self, state: GameState, actions_by_player: Dict[int, List[List]]) -> GameState:
        s = GameState(
            planets=[_clone_planet(p) for p in state.planets],
            fleets=[_clone_fleet(f) for f in state.fleets],
            step=state.step,
            angular_velocity=state.angular_velocity,
            n_players=state.n_players,
            next_fleet_id=state.next_fleet_id,
        )

        next_step = s.step + 1

        # --- 1. Comet expiration ---
        s.planets = [p for p in s.planets if not self._comet_expired(p)]

        # --- 2. Comet spawning (only if precomputed_comets supplied; never for 1-ply lookahead) ---
        if self.precomputed_comets is not None and next_step in COMET_SPAWN_STEPS:
            steps_sorted = [50, 150, 250, 350, 450]
            if next_step in steps_sorted:
                spawn_idx = steps_sorted.index(next_step)
                paths = self.precomputed_comets["paths_per_spawn"][spawn_idx]
                ships = self.precomputed_comets["ships_per_spawn"][spawn_idx]
                if paths is not None:
                    next_id = max((p.id for p in s.planets), default=-1) + 1
                    for i, path in enumerate(paths):
                        s.planets.append(Planet(
                            id=next_id + i, owner=-1, x=-99.0, y=-99.0,
                            radius=COMET_RADIUS, ships=int(ships), production=COMET_PRODUCTION,
                            orbital_radius=0.0, orbital_angle=0.0, is_orbiting=False,
                            is_comet=True,
                            comet_path=[(float(x), float(y)) for x, y in path],
                            comet_path_index=-1,
                        ))

        # --- 3. Fleet launch ---
        planet_by_id = {p.id: p for p in s.planets}
        for player_id, action_list in actions_by_player.items():
            if action_list is None:
                continue
            for move in action_list:
                if len(move) < 3:
                    continue
                from_id, angle, ships = move[0], float(move[1]), int(move[2])
                if from_id not in planet_by_id:
                    continue
                p = planet_by_id[from_id]
                if p.owner != player_id:
                    continue
                if ships <= 0 or ships > p.ships:
                    continue
                p.ships -= ships
                spawn_dist = p.radius + 0.1
                fx = p.x + spawn_dist * math.cos(angle)
                fy = p.y + spawn_dist * math.sin(angle)
                s.fleets.append(Fleet(
                    id=s.next_fleet_id, owner=player_id, x=fx, y=fy, angle=angle,
                    ships=ships, from_planet_id=from_id, speed=fleet_speed(ships),
                ))
                s.next_fleet_id += 1

        # --- 4. Production ---
        for p in s.planets:
            if p.owner >= 0:
                p.ships += p.production

        # --- 5+6. Fleet movement + planet collision (1.30.1 swept-pair) ---
        arrivals: Dict[int, List[Fleet]] = {}
        expired_comet_ids = set()
        self._move_collide_swept(s, arrivals, expired_comet_ids)

        if expired_comet_ids:
            s.planets = [p for p in s.planets if p.id not in expired_comet_ids]

        # --- 7. Combat resolution ---
        for planet_id, fleets_hitting in arrivals.items():
            p = planet_by_id.get(planet_id)
            if p is None:
                continue
            self._resolve_combat(p, fleets_hitting)

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

    def _move_collide_swept(self, s, arrivals, expired_comet_ids):
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
                    expired_comet_ids.add(p.id)
                else:
                    nx, ny = p.comet_path[nidx]
                    check = (ox >= 0.0)
            planet_paths.append((ox, oy, nx, ny, p.radius, check))

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

        for j, p in enumerate(s.planets):
            _, _, pnx, pny, _, _ = planet_paths[j]
            if p.is_orbiting:
                p.orbital_angle += s.angular_velocity
            elif p.is_comet and p.comet_path is not None:
                p.comet_path_index += 1
            p.x, p.y = pnx, pny

    @staticmethod
    def _resolve_combat(planet: Planet, incoming: List[Fleet]) -> None:
        if not incoming:
            return
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
                return
            surv_owner = top_owner
            surv_ships = top_ships - sec_ships

        if surv_owner == planet.owner:
            planet.ships += surv_ships
        else:
            if surv_ships > planet.ships:
                planet.owner = surv_owner
                planet.ships = surv_ships - planet.ships
            else:
                planet.ships -= surv_ships

    @staticmethod
    def _comet_expired(p: Planet) -> bool:
        if not p.is_comet:
            return False
        if p.comet_path is None:
            return False
        if p.comet_path_index >= len(p.comet_path):
            return True
        return False


# ----------------------------- Conversion from kaggle obs -----------------------------
def from_kaggle_obs(obs, n_players: int = 2) -> GameState:
    """kaggle obs dict -> GameState (literal copy of forward_sim.from_kaggle_obs)."""
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
    init_by_id = {ip[0]: ip for ip in initial_planets}

    comet_path_by_id: Dict[int, Tuple[List[Tuple[float, float]], int]] = {}
    for group in comets_info:
        ids = group.get("planet_ids", []) if isinstance(group, dict) else get(group, "planet_ids", [])
        paths = group.get("paths", []) if isinstance(group, dict) else get(group, "paths", [])
        idx = group.get("path_index", 0) if isinstance(group, dict) else get(group, "path_index", 0)
        for pid, path in zip(ids, paths):
            comet_path_by_id[pid] = (path, idx)

    planets: List[Planet] = []
    for pt in raw_planets:
        pid, owner, x, y, radius, ships, production = pt[:7]
        is_comet = pid in comet_planet_ids
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
                orbital_angle = math.atan2(y - SUN_Y, x - SUN_X)

        comet_path: Optional[List[Tuple[float, float]]] = None
        comet_path_index = 0
        if is_comet and pid in comet_path_by_id:
            path, idx = comet_path_by_id[pid]
            comet_path = [tuple(pt2) for pt2 in path]
            comet_path_index = idx

        planets.append(Planet(
            id=pid, owner=int(owner), x=float(x), y=float(y), radius=float(radius),
            ships=int(ships), production=int(production),
            orbital_radius=orbital_radius, orbital_angle=orbital_angle,
            is_orbiting=is_orbiting, is_comet=is_comet,
            comet_path=comet_path, comet_path_index=comet_path_index,
        ))

    fleets: List[Fleet] = []
    for ft in raw_fleets:
        fid, owner, x, y, angle, from_id, ships = ft[:7]
        fleets.append(Fleet(
            id=int(fid), owner=int(owner), x=float(x), y=float(y), angle=float(angle),
            ships=int(ships), from_planet_id=int(from_id), speed=fleet_speed(int(ships)),
        ))

    nfid_obs = get(obs, "next_fleet_id", None)
    if nfid_obs is None:
        # The live kaggle obs always carries next_fleet_id. If absent (defensive), derive
        # a safe upper bound: max existing fleet id + 1 (never reuses a live id this tick).
        nfid_obs = (max((f.id for f in fleets), default=-1) + 1)

    return GameState(
        planets=planets, fleets=fleets,
        step=int(get(obs, "step", 0) or 0),
        angular_velocity=float(ang_vel or 0.0),
        n_players=n_players,
        next_fleet_id=int(nfid_obs),
    )


# ----------------------------- GameState -> kaggle obs -----------------------------
def state_to_obs(state, player: int, initial_planets: list) -> dict:
    """GameState -> kaggle-style per-player obs (literal copy of runner.state_to_obs)."""
    planets = [[int(p.id), int(p.owner), float(p.x), float(p.y), float(p.radius),
                int(p.ships), int(p.production)] for p in state.planets]
    fleets = [[int(f.id), int(f.owner), float(f.x), float(f.y), float(f.angle),
               int(f.from_planet_id), int(f.ships)] for f in state.fleets]
    comet_ids, comet_paths, comet_pidx = [], [], 0
    for p in state.planets:
        if getattr(p, "is_comet", False):
            comet_ids.append(int(p.id))
            comet_paths.append(getattr(p, "comet_path", []) or [])
            comet_pidx = int(getattr(p, "comet_path_index", 0) or 0)
    comets = [{"path_index": comet_pidx, "paths": comet_paths, "planet_ids": comet_ids}] if comet_ids else []
    return {"step": int(state.step), "planets": planets, "fleets": fleets,
            "comets": comets, "comet_planet_ids": comet_ids,
            "initial_planets": initial_planets, "angular_velocity": float(state.angular_velocity),
            "next_fleet_id": int(state.next_fleet_id), "player": int(player),
            "remainingOverageTime": 60}
