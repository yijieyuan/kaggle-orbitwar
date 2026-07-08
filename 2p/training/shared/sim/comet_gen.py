"""Port of kaggle_environments orbit_wars `generate_comet_paths`.

This is a faithful re-implementation: same math (eccentric ellipses, dense
sampling, arc-length resampling, on-board segment extraction, sun+planet
collision validation), same physics constants. The only difference is that
this version accepts an explicit `random.Random` instance for reproducibility
under our own seed (kaggle uses module-level `random`).

Returns 4 symmetric extra-solar comet paths per call, or None on failure
(after 300 rejection-sampling attempts).
"""
from __future__ import annotations

import math
import random
from typing import List, Optional, Set, Tuple

# Constants — must match kaggle_environments orbit_wars
BOARD_SIZE = 100.0
CENTER = BOARD_SIZE / 2.0
SUN_RADIUS = 10.0
ROTATION_RADIUS_LIMIT = 50.0
COMET_RADIUS = 1.0
COMET_PRODUCTION = 1
COMET_SPAWN_STEPS = [50, 150, 250, 350, 450]


def _distance(p1, p2) -> float:
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def generate_comet_paths(
    initial_planets: List[List],
    angular_velocity: float,
    spawn_step: int,
    comet_planet_ids: Optional[Set[int]] = None,
    comet_speed: float = 4.0,
    rng: Optional[random.Random] = None,
) -> Optional[List[List[List[float]]]]:
    """Generate 4 symmetric elliptical extra-solar comet paths.

    Args:
        initial_planets: list of [id, owner, x, y, radius, ships, production].
            Used for collision validation. Comets already in the game (via
            comet_planet_ids) are excluded from validation.
        angular_velocity: rad/turn for orbiting planets (used to predict
            their position at each game_step during validation).
        spawn_step: the game step at which these comets enter (1-indexed).
        comet_planet_ids: set of planet ids that are existing comets (skip).
        comet_speed: fleet/comet movement units per turn.
        rng: a `random.Random` instance for reproducibility. If None, uses
            the global `random` module (matching kaggle behaviour).

    Returns:
        List of 4 paths (each a list of [x, y]), or None on failure.
    """
    r = rng if rng is not None else random
    excluded = set(comet_planet_ids) if comet_planet_ids else set()

    for _attempt in range(300):
        # Sample ellipse params
        e = r.uniform(0.75, 0.93)
        a = r.uniform(60.0, 150.0)
        perihelion = a * (1 - e)
        if perihelion < SUN_RADIUS + COMET_RADIUS:
            continue

        b = a * math.sqrt(1 - e ** 2)
        c_val = a * e
        phi = r.uniform(math.pi / 6, math.pi / 3)

        # Dense sample around perihelion half of the orbit
        dense: List[Tuple[float, float]] = []
        num = 5000
        for i in range(num):
            t = 0.3 * math.pi + 1.4 * math.pi * i / (num - 1)
            ex = c_val + a * math.cos(t)
            ey = b * math.sin(t)
            x = CENTER + ex * math.cos(phi) - ey * math.sin(phi)
            y = CENTER + ex * math.sin(phi) + ey * math.cos(phi)
            dense.append((x, y))

        # Re-sample at constant `comet_speed` arc-length intervals
        path: List[Tuple[float, float]] = [dense[0]]
        cum = 0.0
        target = comet_speed
        for i in range(1, len(dense)):
            cum += _distance(dense[i], dense[i - 1])
            if cum >= target:
                path.append(dense[i])
                target += comet_speed

        # Extract contiguous on-board segment
        board_start: Optional[int] = None
        board_end: Optional[int] = None
        for i, (x, y) in enumerate(path):
            if 0 <= x <= BOARD_SIZE and 0 <= y <= BOARD_SIZE:
                if board_start is None:
                    board_start = i
                board_end = i
        if board_start is None:
            continue
        visible = path[board_start: board_end + 1]
        if not (5 <= len(visible) <= 40):
            continue

        # Build 4 rotationally symmetric variants (4-fold rotation about center).
        # Q1 and opposite copies are reflected across the y=x diagonal so all 4
        # copies are 90° rotations of each other (PR #1016, kaggle env master).
        paths = [
            [[y, x] for x, y in visible],
            [[BOARD_SIZE - x, y] for x, y in visible],
            [[x, BOARD_SIZE - y] for x, y in visible],
            [[BOARD_SIZE - y, BOARD_SIZE - x] for x, y in visible],
        ]

        # Separate planets into static vs orbiting (exclude existing comets)
        static_planets = []
        orbiting_planets = []
        for planet in initial_planets:
            if planet[0] in excluded:
                continue
            pr = _distance((planet[2], planet[3]), (CENTER, CENTER))
            if pr + planet[4] < ROTATION_RADIUS_LIMIT:
                orbiting_planets.append(planet)
            else:
                static_planets.append(planet)

        # Validate paths against sun + static + orbiting planets
        valid = True
        buf = COMET_RADIUS + 0.5
        for k, (cx, cy) in enumerate(visible):
            # Sun
            if _distance((cx, cy), (CENTER, CENTER)) < SUN_RADIUS + COMET_RADIUS:
                valid = False
                break
            sym_pts = [
                (cy, cx),
                (BOARD_SIZE - cx, cy),
                (cx, BOARD_SIZE - cy),
                (BOARD_SIZE - cy, BOARD_SIZE - cx),
            ]
            # Static planets
            for planet in static_planets:
                for sp in sym_pts:
                    if _distance(sp, (planet[2], planet[3])) < planet[4] + buf:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
            # Orbiting planets at projected positions
            game_step = spawn_step - 1 + k
            for planet in orbiting_planets:
                dx = planet[2] - CENTER
                dy = planet[3] - CENTER
                orb_r = math.sqrt(dx ** 2 + dy ** 2)
                init_angle = math.atan2(dy, dx)
                cur_angle = init_angle + angular_velocity * game_step
                px = CENTER + orb_r * math.cos(cur_angle)
                py = CENTER + orb_r * math.sin(cur_angle)
                for sp in sym_pts:
                    if _distance(sp, (px, py)) < planet[4] + COMET_RADIUS:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break

        if valid:
            return paths

    return None


def precompute_all_comets(
    initial_planets: List[List],
    angular_velocity: float,
    seed: int,
    comet_speed: float = 4.0,
) -> dict:
    """Pre-compute all 5 comet spawn groups (turns 50/150/250/350/450).

    Sequentially: each spawn's validation uses initial_planets PLUS comets
    spawned at earlier turns (matching kaggle's behaviour where
    obs0.initial_planets is mutated to include new comets).

    Returns dict with:
      'paths_per_spawn':  list of 5 entries; each is None or list of 4 paths
      'ships_per_spawn':  list of 5 ints (or 0 if no spawn)
      'spawn_steps':      [50, 150, 250, 350, 450]
    """
    rng = random.Random(seed)
    expanded_planets = [list(p) for p in initial_planets]   # copy
    comet_planet_ids: Set[int] = set()
    next_id = max(p[0] for p in expanded_planets) + 1

    paths_per_spawn = []
    ships_per_spawn = []

    for spawn_step in COMET_SPAWN_STEPS:
        paths = generate_comet_paths(
            expanded_planets,
            angular_velocity,
            spawn_step,
            comet_planet_ids,
            comet_speed,
            rng=rng,
        )
        if paths is None:
            paths_per_spawn.append(None)
            ships_per_spawn.append(0)
            continue
        # 4 random.randint draws for ships, then min — matches kaggle
        comet_ships = min(
            rng.randint(1, 99),
            rng.randint(1, 99),
            rng.randint(1, 99),
            rng.randint(1, 99),
        )
        paths_per_spawn.append(paths)
        ships_per_spawn.append(comet_ships)
        # Append placeholder comet planets to expanded_planets for next iteration
        for i, p_path in enumerate(paths):
            pid = next_id + i
            expanded_planets.append([
                pid, -1, -99.0, -99.0, COMET_RADIUS,
                comet_ships, COMET_PRODUCTION,
            ])
            comet_planet_ids.add(pid)
        next_id += 4

    return {
        "paths_per_spawn": paths_per_spawn,
        "ships_per_spawn": ships_per_spawn,
        "spawn_steps": list(COMET_SPAWN_STEPS),
    }
