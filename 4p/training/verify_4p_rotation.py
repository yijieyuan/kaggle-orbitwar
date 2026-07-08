"""exp022-4p SEAT-EQUIVALENCE VERIFICATION GATE (A7 machine proof).

Ports 17v12-4p's verify_17v12_4p_rotation.py to exp022-4p's feature LAYOUT
(static (P,29) = 5-hot owner + 3-hot type + 9 geom + 12 projection; ts (P,50,10) =
[proj_ships, 5 role channels, exists, x, y, ramp]; glob (61,)).

The board's TRUE 4p symmetry is C4 ROTATION (90 deg about (50,50)). With the seat canonicalization
on that SAME C4 rotation (RINGPOS=[0,1,3,2], _rot_xy/_rot_vec/_ring_role), the features must be
EXACTLY seat-invariant under the board's C4 symmetry. We build st_rot = R+90(st) about (50,50)
((x,y)->(100-y,x); velocity vectors likewise; orbital_a += pi/2; comet path rotated; f_angle += pi/2)
with owners ring-shifted by sigma=[1,3,0,2] (next seat on the av>0 ring 0->1->3->2->0), then assert:

    basic_features(st, me=o) == basic_features(st_rot, me=sigma(o))   for ALL 4 seats o,

with static/ts/glob within atol 1e-5 and mask EXACTLY equal. R+90 maps each seat o's home quadrant
exactly onto sigma(o)'s home (BR->BL->TL->TR->BR), so st_rot is the SAME game seen through the board's
true C4 symmetry; if canonicalization is exact the features coincide.

Also re-checks (per seat) the canonical FIRST-future-position (A2-style, on the ts path slot 0) and the
ring-role ownership channels (A3-style) against the numpy reference, for all 4 seats.

Run ONLY on a GPU host (imports jax + the env; Rule 7):
    python verify_4p_rotation.py [board_pool_4p.npz]
(py_compile is run locally; execution is GPU-only.)
"""
import os, sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "jax_env"))

import numpy as np
import jax
import jax.numpy as jnp

from state import JaxState
from step import step as env_step
from env import basic_features, EPISODE_STEPS, N_PLAYERS, FORECAST_H, _forecast, N_STATIC, N_GLOBAL, TS_C, N_ECON

POOL = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "boards_4p.npz")  # provide your own board pool, or omit (defaults to generated boards)
FAILS = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)


def idle_roll(st, n):
    P = st.p_owner.shape[0]
    z = jnp.zeros((P,), jnp.float32); zi = jnp.zeros((P,), jnp.int32); zb = jnp.zeros((P,), bool)
    for _ in range(n):
        st = env_step(st, zb, z, zi)
    return st


RINGPOS = np.array([0, 1, 3, 2])     # ring position along av>0: 0 -> 1 -> 3 -> 2 -> 0


def rot_xy_np(x, y, me):
    """obs -> canonical positions (seat me's home -> TOP-LEFT): 3:id, 0:R180, 1:R+90, 2:R-90."""
    if me == 3:
        return x, y
    if me == 0:
        return 100.0 - x, 100.0 - y
    if me == 1:
        return 100.0 - y, x
    return y, 100.0 - x                  # me == 2


def ring_role_np(o, me):
    """Role = ring distance (RINGPOS[o] - RINGPOS[me]) % 4; valid for o >= 0 (arrays OK)."""
    return (RINGPOS[o] - RINGPOS[me]) % 4


if os.path.exists(POOL):
    z = np.load(POOL)
    pool = JaxState(**{f: jnp.asarray(z[f]) for f in JaxState._fields})
    st0 = jax.tree_util.tree_map(lambda x: x[0], pool)
else:
    print(f"[gate] pool {POOL} missing -> gen_init_states(2) fallback (4-owner boards via reset(4))")
    from env import gen_init_states
    st0 = jax.tree_util.tree_map(lambda x: x[0], gen_init_states(2, seed=0))
st = idle_roll(st0, 55)          # past the first comet spawn -> comets on board too
P = st.p_owner.shape[0]
m = np.asarray(st.p_mask)
px = np.asarray(st.p_x); py = np.asarray(st.p_y)
po = np.asarray(st.p_owner); ps = np.asarray(st.p_ships)
iscom = np.asarray(st.p_is_comet)

# Sanity: feature DIMENSIONS.
static0, ts0, glob0, mask0, econ0 = [np.asarray(x) for x in basic_features(st, 0)]   # exp25: 5-tuple
check(f"DIM static last-dim == N_STATIC ({N_STATIC})", static0.shape[-1] == N_STATIC, f"static {static0.shape}")
check(f"DIM ts == (P,{FORECAST_H},{TS_C})", ts0.shape[1:] == (FORECAST_H, TS_C), f"ts {ts0.shape}")
check(f"DIM glob == ({N_GLOBAL},)", glob0.shape == (N_GLOBAL,), f"glob {glob0.shape}")
check(f"DIM econ_curves == ({FORECAST_H},{N_ECON})", econ0.shape == (FORECAST_H, N_ECON), f"econ {econ0.shape}")

print("=== A. PER-SEAT FEATURE CHECKS (ALL FOUR SEATS) ===")
# A2-style: ts path slot 0 (ts[:, 0, 7:9] = x,y for t+1) must equal the numpy C4 rotation of each
# planet's t+1 position /100 (zeroed where the slot doesn't exist). Rebuild t+1 positions like the env.
ships_raw_chk, owner_h_chk, exists_chk = [np.asarray(x) for x in _forecast(st, FORECAST_H)]
okA2 = True
# ts channel layout: [0]=proj_ships, [1..5]=role channels, [6]=exists, [7]=x, [8]=y, [9]=ramp
X_CH, Y_CH = 7, 8
for me in range(N_PLAYERS):
    ts_me = np.asarray(basic_features(st, me)[1])              # (P,50,10)
    ex0 = exists_chk[:, 0] > 0
    a1 = np.asarray(st.p_orbital_a) + float(st.av) * 1.0
    orb_x = 50.0 + np.asarray(st.p_orbital_r) * np.cos(a1)
    orb_y = 50.0 + np.asarray(st.p_orbital_r) * np.sin(a1)
    cpx = np.asarray(st.p_comet_path_x); cpy = np.asarray(st.p_comet_path_y); cidx = np.asarray(st.p_comet_idx)
    L = cpx.shape[1]
    ci = np.clip(cidx + 1, 0, L - 1)
    com_x = cpx[np.arange(P), ci]; com_y = cpy[np.arange(P), ci]
    isorb = np.asarray(st.p_is_orbiting)
    fx = np.where(iscom, com_x, np.where(isorb, orb_x, px))
    fy = np.where(iscom, com_y, np.where(isorb, orb_y, py))
    rcx, rcy = rot_xy_np(fx, fy, me)
    sel = m & ex0
    okA2 &= bool(np.allclose(ts_me[sel, 0, X_CH], rcx[sel] / 100.0, atol=1e-5))
    okA2 &= bool(np.allclose(ts_me[sel, 0, Y_CH], rcy[sel] / 100.0, atol=1e-5))
check("A2 ts path slot0 == C4 rotation of t+1 pos /100 (seats 0-3)", okA2)

# A3-style: static ownership 5-hot channels (static[:, :5]) = [mine,q1,q2,q3,neu] = ring-distance role
# one-hot (col 4 = neutral non-comet), all 4 seats.
okA3 = True
for me in range(N_PLAYERS):
    s_me = np.asarray(basic_features(st, me)[0])
    exp_ch = np.zeros((P, 5), np.float32)
    for i in range(P):
        if not m[i]:
            continue
        if po[i] >= 0:
            r = int(ring_role_np(int(po[i]), me))        # 0=mine,1=q1,2=q2,3=q3
            exp_ch[i, r] = 1.0
        elif not iscom[i]:
            exp_ch[i, 4] = 1.0                           # neutral (non-comet)
    okA3 &= bool(np.allclose(s_me[m, :5], exp_ch[m], atol=1e-6))
check("A3 static ownership 5-hot = ring distance, RINGPOS [0,1,3,2] (seats 0-3)", okA3)

print("=== A7. SEAT-EQUIVALENCE GOLD TEST (official R+90) ===")
# st_rot = R+90(st) about (50,50): (x,y)->(100-y,x); velocity vectors rotate same linear part;
# orbital_a += pi/2; comet path rotated; f_angle += pi/2; owners ring-shifted by sigma=[1,3,0,2].
SIGMA = np.array([1, 3, 0, 2])


def _relabel_np(o_arr):
    o_arr = np.asarray(o_arr)
    out = o_arr.copy()
    valid = o_arr >= 0
    out[valid] = SIGMA[o_arr[valid]]
    return out.astype(np.int32)


_f32 = np.float32
st_rot = st._replace(
    p_owner=jnp.asarray(_relabel_np(po)),
    p_x=jnp.asarray((100.0 - py).astype(_f32)),
    p_y=jnp.asarray(px.astype(_f32)),
    p_orbital_a=jnp.asarray((np.asarray(st.p_orbital_a) + np.pi / 2).astype(_f32)),
    p_comet_path_x=jnp.asarray((100.0 - np.asarray(st.p_comet_path_y)).astype(_f32)),
    p_comet_path_y=jnp.asarray(np.asarray(st.p_comet_path_x).astype(_f32)),
    f_owner=jnp.asarray(_relabel_np(np.asarray(st.f_owner))),
    f_x=jnp.asarray((100.0 - np.asarray(st.f_y)).astype(_f32)),
    f_y=jnp.asarray(np.asarray(st.f_x).astype(_f32)),
    f_angle=jnp.asarray((np.asarray(st.f_angle) + np.pi / 2).astype(_f32)),
)
okA7 = True
badA7 = []
for o in range(N_PLAYERS):
    f_orig = basic_features(st, o)
    f_rot = basic_features(st_rot, int(SIGMA[o]))
    # exp25: compare static(0), ts(1), glob(2), econ_curves(4) [mask(3) checked exactly below]
    for nm, a, b in zip(("static", "ts", "glob", "econ"),
                        (f_orig[0], f_orig[1], f_orig[2], f_orig[4]),
                        (f_rot[0], f_rot[1], f_rot[2], f_rot[4])):
        a = np.asarray(a); b = np.asarray(b)
        if not np.allclose(a, b, atol=1e-5):
            okA7 = False
            badA7.append(f"seat{o}->sigma{int(SIGMA[o])} {nm} maxdiff={np.max(np.abs(a - b)):.2e}")
    if not np.array_equal(np.asarray(f_orig[3]), np.asarray(f_rot[3])):
        okA7 = False
        badA7.append(f"seat{o}->sigma{int(SIGMA[o])} mask mismatch")
check("A7 seat-equivalence under official R+90 (static/ts/glob identical across relabeled seats)",
      okA7, ";".join(badA7))

print()
if FAILS:
    print(f"VERIFY_4P_ROTATION FAIL ({len(FAILS)}): {FAILS}")
    sys.exit(1)
print("VERIFY_4P_ROTATION PASS")
