"""Collision physics for the JAX 2p forward sim — kaggle 1.30.1 swept-pair.

All functions are pure jnp and broadcast over leading axes (so they vmap over
fleets×planets). Ported byte-for-byte from shared/sim/forward_sim @1.30.1
(_swept_pair_hit, _seg_circle_scalar) and the planet_paths logic of
_move_collide_swept.
"""
import jax.numpy as jnp

from constants import SUN_X, SUN_Y, SUN_R, BOARD


def swept_pair_hit(ax, ay, bx, by, p0x, p0y, p1x, p1y, r):
    """Continuous swept-pair collision: fleet A->B vs planet P0->P1, within r,
    over t in [0,1]. Broadcasts. Returns bool array. Matches forward_sim._swept_pair_hit."""
    d0x = ax - p0x; d0y = ay - p0y
    dvx = (bx - ax) - (p1x - p0x); dvy = (by - ay) - (p1y - p0y)
    a = dvx * dvx + dvy * dvy
    b = 2.0 * (d0x * dvx + d0y * dvy)
    c = d0x * d0x + d0y * d0y - r * r
    disc = b * b - 4.0 * a * c
    sq = jnp.sqrt(jnp.maximum(disc, 0.0))
    two_a = 2.0 * a
    # guard a==0 (no relative motion): hit iff already within r (c<=0)
    safe = a > 1e-12
    t1 = jnp.where(safe, (-b - sq) / jnp.where(safe, two_a, 1.0), 0.0)
    t2 = jnp.where(safe, (-b + sq) / jnp.where(safe, two_a, 1.0), 0.0)
    moving_hit = (disc >= 0.0) & (t2 >= 0.0) & (t1 <= 1.0)
    static_hit = c <= 0.0
    return jnp.where(safe, moving_hit, static_hit)


def seg_hits_sun(ax, ay, bx, by):
    """True if segment A->B comes within SUN_R of the sun (raw radius, env-fatal).
    Mirrors forward_sim._seg_circle_scalar with center=sun, radius=SUN_R."""
    dx = bx - ax; dy = by - ay
    L2 = dx * dx + dy * dy
    t = jnp.where(L2 > 1e-12, ((SUN_X - ax) * dx + (SUN_Y - ay) * dy) / jnp.where(L2 > 1e-12, L2, 1.0), 0.0)
    t = jnp.clip(t, 0.0, 1.0)
    fx = ax + t * dx; fy = ay + t * dy
    d2 = (SUN_X - fx) ** 2 + (SUN_Y - fy) ** 2
    return d2 < SUN_R * SUN_R


def in_board(x, y):
    return (x >= 0.0) & (x <= BOARD) & (y >= 0.0) & (y <= BOARD)


def planet_next_positions(state):
    """End-of-tick planet positions (NOT yet applied) + per-planet collision-check
    flag, mirroring _move_collide_swept step (a).

    Returns (old_x, old_y, new_x, new_y, check):
      - orbiting:  new = rotate by +av;            check = True
      - comet:     new = path[idx+1] (or stay if expiring);
                   check = (old_x >= 0)  (kaggle disables the check on a comet's
                            first on-board placement, where old is the -99 placeholder)
      - static:    new = old;                       check = True
    All (P,) arrays.
    """
    ox, oy = state.p_x, state.p_y
    # orbiting
    na = state.p_orbital_a + state.av
    orb_x = SUN_X + state.p_orbital_r * jnp.cos(na)
    orb_y = SUN_Y + state.p_orbital_r * jnp.sin(na)
    # comet: advance index; clamp for gather; "expiring" if next idx >= len
    nidx = state.p_comet_idx + 1
    safe_idx = jnp.clip(nidx, 0, state.p_comet_path_x.shape[1] - 1)
    com_x = jnp.take_along_axis(state.p_comet_path_x, safe_idx[:, None], axis=1)[:, 0]
    com_y = jnp.take_along_axis(state.p_comet_path_y, safe_idx[:, None], axis=1)[:, 0]
    expiring = nidx >= state.p_comet_len
    com_x = jnp.where(expiring, ox, com_x)          # expiring comet stays put this tick
    com_y = jnp.where(expiring, oy, com_y)

    new_x = jnp.where(state.p_is_comet, com_x, jnp.where(state.p_is_orbiting, orb_x, ox))
    new_y = jnp.where(state.p_is_comet, com_y, jnp.where(state.p_is_orbiting, orb_y, oy))
    # check flag: comets disable on first on-board placement (old==-99 placeholder)
    check = jnp.where(state.p_is_comet, ox >= 0.0, True)
    return ox, oy, new_x, new_y, check
