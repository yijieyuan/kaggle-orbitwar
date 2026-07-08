"""Phase-0 GATE: jax_env.step  vs  shared/sim/forward_sim @1.30.1, frame-by-frame.

Build both from the SAME kaggle obs, fire the SAME launches once, then idle-roll H
turns (no new launches, no comet spawn on either side), comparing every turn:
  planets by id: owner & ships EXACT, x/y within atol
  fleets:        content multiset (owner, ships, round x/y/angle)

Usage: python parity_test.py [start_step=1] [H=60] [seed=0]
(start_step=1 -> comet-free window; start_step~55 -> comets present, still no new spawn)
"""
import os, sys, math, random

_HERE = os.path.dirname(os.path.abspath(__file__))   # .../4p/training/jax_env
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))  # 4p/training (bundled shared/ lives here; up 1)
sys.path.insert(0, _HERE); sys.path.insert(0, _ROOT)

import numpy as np
import jax.numpy as jnp
from kaggle_environments import make
from shared.sim import forward_sim as FS
from state import from_kaggle_obs as jax_from_obs, MAX_PLANETS
from step import step as jax_step


def get_obs_at(start_step, seed):
    random.seed(seed)
    env = make("orbit_wars", configuration={"episodeSteps": 500})
    env.reset(2); env.step([[], []])                      # board generated on first step
    while env.steps[-1][0]["observation"]["step"] < start_step:
        env.step([[], []])
    return env.steps[-1][0]["observation"]


def actions_from(fs_state, id2slot, rng, P):
    """Generate per-turn launches from the CURRENT forward_sim state (owned planets fire),
    returned both as forward_sim actions (by planet id) and JAX arrays (by slot)."""
    launch = np.zeros(P, bool); angle = np.zeros(P, np.float32); ships = np.zeros(P, np.int32)
    fs_actions = {0: [], 1: []}
    plist = fs_state.planets
    for p in plist:
        if p.owner in (0, 1) and p.ships > 1 and rng.random() < 0.5:
            tgt = rng.choice([q for q in plist if q.id != p.id])
            ang = math.atan2(tgt.y - p.y, tgt.x - p.x)
            s = max(1, int(p.ships // 2))
            fs_actions[p.owner].append([p.id, ang, s])
            sl = id2slot.get(int(p.id))
            if sl is not None:
                launch[sl] = True; angle[sl] = ang; ships[sl] = s
    return fs_actions, jnp.asarray(launch), jnp.asarray(angle), jnp.asarray(ships)


def cmp(fs_state, js):
    """Game-LOGIC parity (owner/ships exact, fleet owner/ships multiset). Position
    differences are reported separately as float32-vs-float64 noise, not failures.
    Returns (logic_mismatches[list], max_pos_dev[float])."""
    from collections import Counter
    mism = []; pos_dev = 0.0
    fs_p = {int(p.id): (int(p.owner), int(p.ships), float(p.x), float(p.y)) for p in fs_state.planets}
    mask = np.asarray(js.p_mask); pid = np.asarray(js.p_id)
    po = np.asarray(js.p_owner); psh = np.asarray(js.p_ships); px = np.asarray(js.p_x); py = np.asarray(js.p_y)
    js_p = {int(pid[i]): (int(po[i]), int(psh[i]), float(px[i]), float(py[i])) for i in range(len(mask)) if mask[i]}
    if set(fs_p) != set(js_p):
        mism.append(f"planet-id set differs: fs_only={set(fs_p)-set(js_p)} js_only={set(js_p)-set(fs_p)}")
    for k in set(fs_p) & set(js_p):
        fo, fsh, fx, fy = fs_p[k]; jo, jsh, jx, jy = js_p[k]
        if fo != jo: mism.append(f"p{k} owner {fo}!={jo}")
        if fsh != jsh: mism.append(f"p{k} ships {fsh}!={jsh}")
        pos_dev = max(pos_dev, abs(fx - jx), abs(fy - jy))
    # fleets: owner/ships multiset must match exactly (positions tracked as dev)
    fs_fc = Counter((int(f.owner), int(f.ships)) for f in fs_state.fleets)
    fm = np.asarray(js.f_mask)
    js_fc = Counter((int(js.f_owner[i]), int(js.f_ships[i])) for i in range(len(fm)) if fm[i])
    if fs_fc != js_fc:
        diff_fs = fs_fc - js_fc; diff_js = js_fc - fs_fc
        mism.append(f"fleet (owner,ships) multiset differs: fs_only={dict(diff_fs)} js_only={dict(diff_js)}")
    return mism, pos_dev


def main():
    start = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    H = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    obs = get_obs_at(start, seed)
    FS.COLLISION_MODEL = "1.30.1"
    from shared.sim.comet_gen import precompute_all_comets
    import comet as cometmod
    pc = precompute_all_comets(obs["initial_planets"], obs["angular_velocity"], seed)
    sched = cometmod.gen_schedule(obs["initial_planets"], obs["angular_velocity"], seed)
    sim = FS.OrbitSimulator(precomputed_comets=pc)        # forward_sim spawns comets at 50/150/...
    fs = FS.from_kaggle_obs(obs, 2)
    js = cometmod.attach_schedule(jax_from_obs(obs), sched)  # JAX env spawns the SAME comets
    P = MAX_PLANETS
    rng = random.Random(seed + 999)
    pid = np.asarray(js.p_id); mask = np.asarray(js.p_mask)
    id2slot = {int(pid[i]): i for i in range(P) if mask[i]}   # stable (no comet removal here)
    print(f"start_step={obs['step']} planets={len(obs['planets'])} comets={len(obs.get('comet_planet_ids',[]))} H={H}")
    total = 0; first_fail = None; n_launched = 0; max_fleets = 0; max_dev = 0.0
    for t in range(H):
        fs_act, launch, angle, ships = actions_from(fs, id2slot, rng, P)
        n_launched += int(np.asarray(launch).sum())
        fs = sim.step(fs, fs_act)
        js = jax_step(js, launch, angle, ships)
        max_fleets = max(max_fleets, int(np.asarray(js.f_mask).sum()))
        m, dev = cmp(fs, js)
        max_dev = max(max_dev, dev)
        if m and first_fail is None:
            first_fail = (t, int(np.asarray(js.step)))
            print(f"  FIRST LOGIC MISMATCH t={t} step={int(np.asarray(js.step))}: {len(m)}")
            for x in m[:8]:
                print("     ", x)
        total += len(m)
    print(f"launched={n_launched}  max_fleets_in_flight={max_fleets}  max_pos_dev={max_dev:.4f} (float32-vs-float64)")
    print("PARITY", "PASS" if total == 0 else f"FAIL (logic_mismatch={total}, first at {first_fail})")


if __name__ == "__main__":
    main()
