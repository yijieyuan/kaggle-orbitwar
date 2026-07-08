"""Generate a reusable POOL of random orbit_wars 4p opening boards ONCE and cache it to
<training>/shared/board_pool_4p/<ver>/boards.npz (+ config.json) — the 4p analog of
2p/training/gen_board_pool.py. The 4p decode_parity.py / parity checks look for the pool at
shared/board_pool_4p/v1/boards.npz, so `--version v1` makes them runnable folder-only. Training
generates boards procedurally by default (gen_init_states); this saved pool is optional.

Each board needs a FRESH make()+reset(4)+step (~2-3s; reset alone does NOT re-randomize), so we
PARALLELISE the one-time gen across workers. Boards are a batched JaxState (version-independent).

  python -u 4p/training/gen_board_pool_4p.py --num 2048 --version v1 --workers 16
"""
import argparse, hashlib, json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))   # 4p/training (self-contained)
VJAX = os.path.join(HERE, "jax_env")                 # local 4p jax engine (state.py, comet.py)
sys.path.insert(0, VJAX)


def _gen_one(arg):
    """Worker: one fresh 4p board -> numpy dict (picklable). FRESH make() per board (reset does not
    re-randomize). Reseed BEFORE make so the board is deterministic per index."""
    import random
    from kaggle_environments import make
    from state import from_kaggle_obs
    import comet as cometmod
    i, seed = arg
    random.seed(seed + i)
    env = make("orbit_wars", configuration={"episodeSteps": 500})
    env.reset(4)
    env.step([[], [], [], []])
    obs = env.steps[1][0]["observation"]
    st = from_kaggle_obs(obs)
    sched = cometmod.gen_schedule(obs["initial_planets"], obs["angular_velocity"], seed + i)
    board = cometmod.attach_schedule(st, sched)
    return {f: np.asarray(getattr(board, f)) for f in board._fields}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--version", default=None, help="save to shared/board_pool_4p/<version>/ if set")
    args = ap.parse_args()

    from concurrent.futures import ProcessPoolExecutor
    t0 = time.time()
    out = [None] * args.num
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, d in enumerate(ex.map(_gen_one, [(j, args.seed) for j in range(args.num)])):
            out[i] = d; done += 1
            if done % 256 == 0:
                print(f"  {done}/{args.num} ({(time.time()-t0):.0f}s)", flush=True)
    fields = list(out[0].keys())
    stacked = {f: np.stack([r[f] for r in out], 0) for f in fields}
    dt = time.time() - t0
    px = stacked["p_x"]
    uniq = len({hashlib.md5(px[i].tobytes()).hexdigest() for i in range(args.num)})
    print(f"gen {args.num} 4p boards in {dt:.0f}s ({1000*dt/args.num:.0f} ms/board, {args.workers}w) | distinct {uniq}/{args.num}", flush=True)

    if args.version:
        outdir = os.path.join(HERE, "shared", "board_pool_4p", args.version)
        os.makedirs(outdir, exist_ok=True)
        npz = os.path.join(outdir, "boards.npz")
        np.savez_compressed(npz, **stacked)   # comet-path arrays are mostly zeros -> compresses ~10x
        json.dump({"n_boards": int(args.num), "seed": int(args.seed), "distinct": int(uniq),
                   "fields": fields, "P": int(px.shape[1]), "gen_sec": round(dt, 1)},
                  open(os.path.join(outdir, "config.json"), "w"), indent=2)
        print(f"saved -> {npz}  ({os.path.getsize(npz)/1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
