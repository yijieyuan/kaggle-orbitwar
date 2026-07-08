"""Canonical observation parsing into a named :class:`ParsedObs` dataclass.

Converts the raw 7-field observation tensors (produced by
:func:`adapter.single_obs_to_tensor`) into named per-planet/per-fleet fields.

Field index definitions
-----------------------
``planets`` / ``initial_planets``  ``[P, 7]`` float32::

    0 – planet_id   (alive sentinel: id >= 0; padding value: -1)
    1 – owner       (absolute player index; -1 = neutral)
    2 – x           (board coordinates, 0–100)
    3 – y
    4 – radius
    5 – ships       (current count)
    6 – production  (ships added per turn when owned)

``fleets``  ``[F, 7]`` float32::

    0 – fleet_id    (alive sentinel: id >= 0)
    1 – owner
    2 – x
    3 – y
    4 – angle       (radians)
    5 – from_planet_id
    6 – ships

No field indices appear outside this module; all downstream modules consume
:class:`ParsedObs` named fields instead.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .constants import CENTER, ROT_RADIUS_LIMIT


# ---------------------------------------------------------------------------
# ParsedObs
# ---------------------------------------------------------------------------

@dataclass
class ParsedObs:
    """Named per-planet fields decoded from a raw batch observation dict.

    All tensor fields have shape ``[P]`` unless stated otherwise.
    """

    # --- raw planet fields --------------------------------------------------
    alive: Tensor       # bool  – planet_id >= 0 (not a padding slot)
    x: Tensor           # float – current x position (0–100)
    y: Tensor           # float – current y position (0–100)
    r: Tensor           # float – radius
    ships: Tensor       # float – current ship count
    prod: Tensor        # float – production per turn
    owner_abs: Tensor   # float – absolute owner id (-1 = neutral)

    # --- relative ownership masks (computed from owner_abs + player_id) -----
    owned: Tensor       # bool – alive & owner_abs == player_id
    is_enemy: Tensor    # bool – alive & owner_abs >= 0 & owner_abs != player_id
    is_neutral: Tensor  # bool – alive & owner_abs < 0

    # --- orbital parameters (reconstructed from initial_planets) ------------
    orb_r: Tensor       # float – orbital radius; 0.0 for static planets
    orb_a0: Tensor      # float – initial angle from CENTER (radians)
    is_orbiting: Tensor # bool  – True for rotating planets

    # --- game scalars -------------------------------------------------------
    angvel: Tensor      # [B] float – board angular velocity (rad/turn)
    step: Tensor        # [B] float – current game step

    # --- fleet fields -------------------------------------------------------
    #  Available when parse_obs is called with include_fleets=True;
    #  shapes are [F, *] and accessed as attributes rather than being
    #  indexed per-column.
    f_alive: Tensor     # [F] bool
    f_owner: Tensor     # [F] float – absolute owner
    f_x: Tensor         # [F] float
    f_y: Tensor         # [F] float
    f_angle: Tensor     # [F] float – radians
    f_ships: Tensor     # [F] float

    # --- metadata -----------------------------------------------------------
    player_id: int
    P: int
    F: int
    device: torch.device


# ---------------------------------------------------------------------------
# parse_obs
# ---------------------------------------------------------------------------

def parse_obs(obs_tensors: dict, player_id: int | None = None) -> ParsedObs:
    """Decode a raw batch observation dict into a :class:`ParsedObs`.

    Args:
        obs_tensors: dict as produced by ``adapter.single_obs_to_tensor`` or
                     Required keys:
                     ``"planets"`` ``[P, 7]``,
                     ``"initial_planets"`` ``[P, 7]``,
                     ``"fleets"`` ``[F, 7]``,
                     ``"angular_velocity"`` scalar,
                     ``"step"`` scalar,
                     ``"player"`` scalar.
        player_id:   Which player to compute ownership masks for.  Defaults
                     to ``int(obs_tensors["player"][0])``.

    Returns:
        :class:`ParsedObs` with all tensors on the same device as ``planets``.
    """
    planets = obs_tensors["planets"]          # [P, 7]
    initial = obs_tensors["initial_planets"]  # [P, 7]
    fleets = obs_tensors["fleets"]            # [F, 7]
    angvel = obs_tensors["angular_velocity"].float()  # scalar
    step = obs_tensors["step"].float()        # scalar

    if player_id is None:
        player_id = int(obs_tensors["player"].flatten()[0].item())

    P, _ = planets.shape
    F, _ = fleets.shape
    device = planets.device

    # -- planet columns -------------------------------------------------------
    pid = planets[..., 0]        # [P]
    owner_abs = planets[..., 1]
    x = planets[..., 2]
    y = planets[..., 3]
    r = planets[..., 4]
    ships = planets[..., 5]
    prod = planets[..., 6]

    alive = pid >= 0.0

    owned = alive & (owner_abs == float(player_id))
    is_enemy = alive & (owner_abs >= 0.0) & (owner_abs != float(player_id))
    is_neutral = alive & (owner_abs < 0.0)

    # -- orbital parameters from initial_planets ------------------------------
    # A planet is "orbiting" when its distance from the board centre plus its
    # radius is below ROT_RADIUS_LIMIT (mirroring the engine's initialisation
    # logic).  We reconstruct the orbital radius and initial angle from the
    # initial position stored in the observation.
    ix = initial[..., 2]  # [P]
    iy = initial[..., 3]
    i_r = initial[..., 4]  # initial radius (same as current for orbiting)

    dx0 = ix - CENTER
    dy0 = iy - CENTER
    orb_r_raw = torch.sqrt(dx0 * dx0 + dy0 * dy0)
    orb_a0 = torch.atan2(dy0, dx0)

    # Orbiting: alive, initial orbital radius + planet radius < limit, and
    # non-trivially away from the centre (avoids treating dead/padding slots
    # with ix=iy=0 as orbiting).
    is_orbiting = alive & ((orb_r_raw + i_r) < ROT_RADIUS_LIMIT) & (orb_r_raw > 0.5)

    # Static planets carry orb_r = 0 so downstream maths stay correct.
    orb_r = torch.where(is_orbiting, orb_r_raw, torch.zeros_like(orb_r_raw))

    # -- fleet columns --------------------------------------------------------
    f_pid = fleets[..., 0]      # [F]
    f_alive = f_pid >= 0.0
    f_owner = fleets[..., 1]
    f_x = fleets[..., 2]
    f_y = fleets[..., 3]
    f_angle = fleets[..., 4]
    f_ships = fleets[..., 6]

    return ParsedObs(
        alive=alive,
        x=x, y=y, r=r,
        ships=ships, prod=prod,
        owner_abs=owner_abs,
        owned=owned,
        is_enemy=is_enemy,
        is_neutral=is_neutral,
        orb_r=orb_r,
        orb_a0=orb_a0,
        is_orbiting=is_orbiting,
        angvel=angvel,
        step=step,
        f_alive=f_alive,
        f_owner=f_owner,
        f_x=f_x, f_y=f_y,
        f_angle=f_angle,
        f_ships=f_ships,
        player_id=player_id,
        P=P, F=F,
        device=device,
    )
