"""exp025 v1 (simplefrac) decode-parity gate.

Proves the numpy `rl_infer.decode` byte-matches the jax greedy decode (= exp25 training decode = the
cross-daemon make_greedy simplefrac branch) so run_battery / kaggle-deploy play IDENTICALLY to training.

exp25 arch (= the v41 OrbitNet19 + SimpleFracMLP lineage, all VERBATIM): OrbitNet19 E=128 n_layers=6
n_heads=4 (econ-CNN folded into gtok; board ATTN-ONLY [gemb || attn] = 2E; 3E ctx REMOVED; pointer reads
emb, value reads board) + SimpleFracMLP(emb, emb_tid, gemb) -> (mu, sigma) (no edge, no coordination).
The frac head, the trunk, and ALL physics/lead/edge/forecast helpers are byte-identical to the
parity-verified v41 port; ONLY the SELF-CAP differs (exp25 trains under a per-player in-flight cap of
FLEET_CAP_PER_PLAYER=128; the official deploy engine is uncapped so rl_infer re-applies it). Both the jax
reference greedy AND the numpy decode here apply that owner-relative cap, so the decode-to-decode compare
is exact.

Features are IDENTICAL between exp25 jax env.basic_features and the v41 numpy basic_features:
static (P,23), ts (P,50,7), glob (28,), econ_curves (50,2). (The "(P,21)/(P,50,6)/glob 32" comments in
env.py / agent_meta.json are STALE -- trust the code, which this gate verifies to f32.)

train.py.greedy_action is stale -> we BUILD the jax greedy decode INLINE here, using env.basic_features
(5-tuple, +econ), the model/frac apply with econ, and train.py's econ-INDEPENDENT edge_features /
first_hit_gate (reused as-is). This is the SAME path as cross_daemon_2p.make_greedy(simplefrac).

Run (GPU host, Rule 7 -- do NOT run jax locally):
  GARRISON_REMAT=0 python decode_parity.py <ckpt.msgpack> <weights.npz> [board_pool.npz] [n_boards=16] [n_steps=10]
Acceptance: feature static/ts/glob/econ max|d|<=~1e-5, pointer tid mismatch=0, frac f=clip(mu,0,1)
max|d|<=~3e-3, decode launch/tid/ships exact, angle<=1e-3. The ~1e-2 raw-logit/ctx drift is the known
jax-0.10 numerical artifact (acceptable when tid_mm=0 and frac f matches).
"""
import os, sys
from pathlib import Path

os.environ.setdefault("GARRISON_REMAT", "0")

_HERE = Path(__file__).resolve().parent
PROJ_ROOT = str(_HERE)   # bundled shared/ lives under this dir
sys.path.insert(0, str(_HERE / "jax_env"))
sys.path.insert(0, str(_HERE))
sys.path.insert(0, PROJ_ROOT)

import numpy as np
import jax
import jax.numpy as jnp
import flax.serialization as fser

from state import JaxState                              # noqa: E402
from step import step as env_step                       # noqa: E402
from constants import FLEET_CAP_PER_PLAYER              # noqa: E402  (exp25 train-only per-player cap)
import train as TR                                       # noqa: E402  (edge_features, first_hit_gate)
from env import basic_features as jax_basic_features     # noqa: E402  (5-tuple, +econ)
from env import _forecast as jax_forecast                # noqa: E402
from targeting import reach_solve_static as jax_reach_solve_static  # noqa: E402
from targeting import lead_for_ships as jax_lead_for_ships          # noqa: E402
from model import OrbitNet19, SimpleFracMLP              # noqa: E402
import rl_infer as R                                     # noqa: E402

if len(sys.argv) < 3:
    sys.exit("usage: python decode_parity.py <ckpt.msgpack> <weights.npz> [board_pool.npz] [n_boards=16] [n_steps=10]")
CKPT = sys.argv[1]
NPZ = sys.argv[2]
BOARD_POOL = sys.argv[3] if len(sys.argv) > 3 else f"{PROJ_ROOT}/shared/board_pool/f256/boards.npz"
N_BOARDS = int(sys.argv[4]) if len(sys.argv) > 4 else 16
N_STEPS = int(sys.argv[5]) if len(sys.argv) > 5 else 10

assert R.FLEET_CAP_PER_PLAYER == FLEET_CAP_PER_PLAYER, \
    f"rl_infer cap {R.FLEET_CAP_PER_PLAYER} != env cap {FLEET_CAP_PER_PLAYER}"

W = R.load_weights(NPZ)
E_ = 128
NL = 6
print(f"[exp25-parity] ckpt={CKPT}\n               npz={NPZ}  E={E_} n_layers={NL} n_heads=4 "
      f"(SimpleFracMLP, FLEET_CAP={FLEET_CAP_PER_PLAYER})\n               "
      f"pool={BOARD_POOL}  n_boards={N_BOARDS}  n_steps={N_STEPS}", flush=True)

# ---------------- net + frac + params ----------------
net = OrbitNet19(E=E_, n_layers=NL, n_heads=4)
frac = SimpleFracMLP(E=E_, n_heads=4)
tree = fser.msgpack_restore(open(CKPT, "rb").read())
assert all(k in tree for k in ("net", "frac")), f"not a 2-tree {{net,frac}} ckpt: {list(tree)}"
params = jax.tree_util.tree_map(jnp.asarray, tree)


def jax_greedy(st, me):
    """Inline jax greedy decode == cross_daemon_2p.make_greedy(simplefrac) == exp25 train decode, PLUS
    the owner-relative SELF-CAP (mirror of step._cap_ok for the single seat `me`, exactly as rl_infer.decode
    does). Returns (launch, angle, ships, tid, arrival) (legacy greedy_action tuple order)."""
    P = st.p_owner.shape[0]; ar = jnp.arange(P); f32 = jnp.float32
    fc = jax_forecast(st)
    Rr, ANG, TURNS, Rg = jax_reach_solve_static(st)
    static, ts, glob, m, econ = jax_basic_features(st, me, fc=fc)
    is_mine = (st.p_owner == me) & st.p_mask
    reach = Rg & is_mine[:, None]
    _, _, _, edge = TR.edge_features(st, fc=fc, lead=(Rr, ANG, TURNS))
    tgt, emb, gemb, _b, _v = net.apply(params["net"], static, ts, glob, reach, m, edge, econ)
    tid = jnp.argmax(tgt, -1)
    is_real = is_mine & (tid != ar) & Rr[ar, tid]
    emb_tid = emb[tid]
    mu, sigma = frac.apply(params["frac"], emb, emb_tid, gemb)          # SimpleFracMLP([emb[s] || emb_tid || gemb])
    f = jnp.clip(mu, 0.0, 1.0)
    garrison = st.p_ships
    ships = jnp.clip(jnp.round(f * garrison.astype(f32)).astype(jnp.int32), 0, garrison)
    Rx, ANGx, TURNSx = jax_lead_for_ships(st, ships)
    angle = ANGx[ar, tid]; turns = TURNSx[ar, tid]
    cand = is_real & (ships > 0) & Rx[ar, tid]
    launch = cand & TR.first_hit_gate(st, tid, angle, ships)
    # --- SELF-CAP (owner-relative mirror of step._cap_ok for seat `me`) ---
    _ex_me = jnp.sum(st.f_mask & (st.f_owner == me))
    _rank_me = jnp.cumsum((launch & (st.p_owner == me)).astype(jnp.int32)) - 1
    launch = launch & (_ex_me + _rank_me < FLEET_CAP_PER_PLAYER)
    ships = jnp.where(launch, ships, 0)
    return launch, angle, ships, tid, st.step + turns


# ---------------- board pool ----------------
if not os.path.exists(BOARD_POOL):
    sys.exit(
        f"board pool not found: {BOARD_POOL}\n"
        "The board pool is a large regeneratable artifact and is NOT bundled in this repo.\n"
        "Pass an explicit pool as the 3rd arg, e.g.:\n"
        "  python decode_parity.py <ckpt.msgpack> <weights.npz> /path/to/boards.npz [n_boards] [n_steps]\n"
        "Generate one with: python gen_board_pool.py --num 256 --version f256 --workers 16\n"
        "  (writes 2p/training/shared/board_pool/f256/boards.npz)"
    )
z = np.load(BOARD_POOL)
missing = [f for f in JaxState._fields if f not in z.files]
if missing:
    sys.exit(f"board pool {BOARD_POOL} missing JaxState fields: {missing}")
pool = JaxState(**{f: jnp.asarray(z[f]) for f in JaxState._fields})
n_pool = int(pool.p_owner.shape[0])
N_BOARDS = min(N_BOARDS, n_pool)
print(f"[exp25-parity] board pool: {n_pool} boards (using first {N_BOARDS})", flush=True)


def state_to_arr(st):
    a = lambda x: np.asarray(x)
    return dict(
        p_owner=a(st.p_owner), p_x=a(st.p_x), p_y=a(st.p_y), p_radius=a(st.p_radius),
        p_ships=a(st.p_ships), p_prod=a(st.p_prod), p_mask=a(st.p_mask),
        p_is_comet=a(st.p_is_comet), p_is_orbiting=a(st.p_is_orbiting),
        p_orbital_r=a(st.p_orbital_r), p_orbital_a=a(st.p_orbital_a),
        p_comet_path_x=a(st.p_comet_path_x), p_comet_path_y=a(st.p_comet_path_y),
        p_comet_idx=a(st.p_comet_idx), p_comet_len=a(st.p_comet_len),
        f_owner=a(st.f_owner), f_ships=a(st.f_ships),
        f_target=a(st.f_target), f_arrival=a(st.f_arrival), f_mask=a(st.f_mask),
        step=int(st.step), av=float(st.av),
    )


# ============================================================
# (a) FEATURE + FORWARD parity on board0 / seat0 (incl FRAC head mu/sigma)
# ============================================================
def feature_parity():
    st0 = jax.tree_util.tree_map(lambda x: x[0], pool)
    arr0 = state_to_arr(st0)
    me = 0
    P = int(st0.p_owner.shape[0]); ar = np.arange(P)
    print("\n[exp25-parity] --- FEATURE parity (board 0, seat 0) ---", flush=True)

    j_sr, j_oh, j_ex = jax_forecast(st0)
    n_sr, n_oh, n_ex = R._forecast(arr0)
    print(f"  forecast ships_raw  max|d| = {np.abs(np.asarray(j_sr) - n_sr).max():.3e}", flush=True)
    print(f"  forecast owner_h    mismatch = {int((np.asarray(j_oh) != n_oh).sum())}", flush=True)

    jR, jANG, jTURNS, jRg = jax_reach_solve_static(st0)
    nR, nANG, nTURNS, nRg = R.reach_solve_static(arr0)
    print(f"  reach R   mismatch = {int((np.asarray(jR) != nR).sum())}", flush=True)
    print(f"  reach Rg  mismatch = {int((np.asarray(jRg) != nRg).sum())}", flush=True)

    fc_j = jax_forecast(st0); fc_n = R._forecast(arr0)
    _, _, _, j_edge = TR.edge_features(st0, fc=fc_j, lead=(jR, jANG, jTURNS))
    _, _, _, n_edge = R.edge_features(arr0, fc=fc_n, lead=(nR, nANG, nTURNS))
    print(f"  edge (P,P,6)        max|d| = {np.abs(np.asarray(j_edge) - n_edge).max():.3e}", flush=True)

    j_static, j_ts, j_glob, j_m, j_econ = jax_basic_features(st0, me, fc=fc_j)
    n_static, n_ts, n_glob, n_m, n_econ = R.basic_features(arr0, me, fc=fc_n)
    print(f"  static (P,23)       max|d| = {np.abs(np.asarray(j_static) - n_static).max():.3e}", flush=True)
    print(f"  ts (P,50,7)         max|d| = {np.abs(np.asarray(j_ts) - n_ts).max():.3e}", flush=True)
    print(f"  glob (28,)          max|d| = {np.abs(np.asarray(j_glob) - n_glob).max():.3e}", flush=True)
    print(f"  econ_curves (50,2)  max|d| = {np.abs(np.asarray(j_econ) - n_econ).max():.3e}", flush=True)
    print(f"  static shape jax={tuple(np.asarray(j_static).shape)} np={n_static.shape} | "
          f"ts jax={tuple(np.asarray(j_ts).shape)} np={n_ts.shape} | "
          f"glob jax={tuple(np.asarray(j_glob).shape)} np={n_glob.shape}", flush=True)

    # full forward (5-tuple, frac reads gemb)
    is_mine = (arr0["p_owner"] == me) & arr0["p_mask"]
    reach = nRg & is_mine[:, None]
    j_reach = jRg & ((st0.p_owner == me) & st0.p_mask)[:, None]
    j_out, j_state = net.apply(params["net"], j_static, j_ts, j_glob, j_reach, j_m, j_edge, j_econ,
                               capture_intermediates=True, mutable=["intermediates"])
    j_tgt, j_emb, j_gemb, j_board, j_v = j_out
    n_tgt, n_emb, n_gemb, n_board, n_v = R.orbitnet19_forward(n_static, n_ts, n_glob, reach, n_m, n_edge, n_econ, W)
    ji = j_state["intermediates"]
    for nm, key in [("ts_emb", "LayerNorm_0"), ("static_emb", "LayerNorm_1"), ("econ_emb", "LayerNorm_2")]:
        try:
            jv = np.asarray(ji[key]["__call__"][0]); nv = np.asarray(R._DBG[nm])
            print(f"  [dbg] {nm:11s} max|d| = {np.abs(jv - nv).max():.3e}", flush=True)
        except Exception as e:
            print(f"  [dbg] {nm}: {e}", flush=True)
    print(f"  gemb (E,)           max|d| = {np.abs(np.asarray(j_gemb) - n_gemb).max():.3e}", flush=True)
    jt = np.asarray(j_tgt); nt = np.asarray(n_tgt)
    finite = np.isfinite(jt) & (jt > -1e8) & np.isfinite(nt) & (nt > -1e8)
    print(f"  tgt logits (legal)  max|d| = {(np.abs(jt[finite] - nt[finite]).max() if finite.any() else 0.0):.3e}", flush=True)
    print(f"  emb (P,E)           max|d| = {np.abs(np.asarray(j_emb) - n_emb).max():.3e}", flush=True)
    print(f"  board (2E)          max|d| = {np.abs(np.asarray(j_board) - n_board).max():.3e}", flush=True)
    print(f"  value v             |d|    = {abs(float(j_v) - float(n_v)):.3e}", flush=True)

    # --- FRAC head (mu, sigma) ---
    j_tid = np.asarray(jnp.argmax(j_tgt, -1)); n_tid = np.argmax(n_tgt, axis=1)
    print(f"  pointer tid         mismatch = {int((j_tid != n_tid).sum())}", flush=True)
    jtid = jnp.argmax(j_tgt, -1)
    j_mu, j_sig = frac.apply(params["frac"], j_emb, j_emb[jtid], j_gemb)       # SimpleFracMLP([emb[s] || emb_tid || gemb])
    n_mu, n_sig = R.cfracgauss_forward(n_emb, n_emb[n_tid], n_gemb, W)
    print(f"  frac mu             max|d| = {np.abs(np.asarray(j_mu) - n_mu).max():.3e}", flush=True)
    print(f"  frac sigma          max|d| = {np.abs(np.asarray(j_sig) - n_sig).max():.3e}", flush=True)
    print(f"  frac f=clip(mu,0,1) max|d| = {np.abs(np.clip(np.asarray(j_mu),0,1) - np.clip(n_mu,0,1)).max():.3e}", flush=True)


feature_parity()


# ============================================================
# (b) DECODE parity over boards x stepped-states x seats (launch/tid/ships/angle, post SELF-CAP)
# ============================================================
def compare_decode(st, board, step_idx):
    arr = state_to_arr(st)
    res = {"slots": 0, "launch_mm": 0, "tid_mm": 0, "ships_mm": 0, "ang_max": 0.0}
    bad = None
    for me in (0, 1):
        jl, ja, js, jt, jar = jax_greedy(st, me)          # (launch, angle, ships, tid, arrival)
        nl, nt, na, ns = R.decode(arr, W, me)             # (launch, tid, angle, ships)
        jl_ = np.asarray(jl); ja_ = np.asarray(ja); js_ = np.asarray(js); jt_ = np.asarray(jt)
        nl_ = np.asarray(nl); na_ = np.asarray(na); ns_ = np.asarray(ns); nt_ = np.asarray(nt)
        lm = int((jl_ != nl_).sum())
        both = jl_ & nl_
        bi = np.where(both)[0]
        tm = int((jt_[bi] != nt_[bi]).sum())
        sm = int((js_[bi] != ns_[bi]).sum())
        ad = (float(np.abs(((ja_[bi] - na_[bi] + np.pi) % (2 * np.pi)) - np.pi).max()) if bi.size else 0.0)
        res["slots"] += int(jl_.shape[0])
        res["launch_mm"] += lm; res["tid_mm"] += tm; res["ships_mm"] += sm
        res["ang_max"] = max(res["ang_max"], ad)
        if bad is None and (lm or tm or sm or ad > 1e-3):
            if lm:
                p = int(np.where(jl_ != nl_)[0][0]); bad = (board, step_idx, me, p, "launch", int(jl_[p]), int(nl_[p]))
            elif tm:
                p = int(bi[np.where(jt_[bi] != nt_[bi])[0][0]]); bad = (board, step_idx, me, p, "tid", int(jt_[p]), int(nt_[p]))
            elif sm:
                p = int(bi[np.where(js_[bi] != ns_[bi])[0][0]]); bad = (board, step_idx, me, p, "ships", int(js_[p]), int(ns_[p]))
            else:
                p = int(bi[np.argmax(np.abs(((ja_[bi] - na_[bi] + np.pi) % (2 * np.pi)) - np.pi))])
                bad = (board, step_idx, me, p, "angle", float(ja_[p]), float(na_[p]))
    return res, bad


tot = {"slots": 0, "launch_mm": 0, "tid_mm": 0, "ships_mm": 0, "ang_max": 0.0}
first_bad = None
print(f"\n[exp25-parity] --- DECODE parity ({N_BOARDS} boards x (1 + {N_STEPS} stepped) x 2 seats) ---", flush=True)

for b in range(N_BOARDS):
    st = jax.tree_util.tree_map(lambda x: x[b], pool)
    for s in range(N_STEPS + 1):
        res, bad = compare_decode(st, b, s)
        for k in ("slots", "launch_mm", "tid_mm", "ships_mm"):
            tot[k] += res[k]
        tot["ang_max"] = max(tot["ang_max"], res["ang_max"])
        if first_bad is None and bad is not None:
            first_bad = bad
        if s == N_STEPS:
            break
        acts = {me: jax_greedy(st, me) for me in (0, 1)}
        o = st.p_owner
        l0, a0, s0, t0, ar0 = acts[0]; l1, a1, s1, t1, ar1 = acts[1]
        launch = jnp.where(o == 0, l0, jnp.where(o == 1, l1, False))
        angle = jnp.where(o == 0, a0, a1); ships = jnp.where(o == 0, s0, s1)
        target = jnp.where(o == 0, t0, t1); arrival = jnp.where(o == 0, ar0, ar1)
        st = env_step(st, launch, angle, ships, target, arrival)

ok = (tot["launch_mm"] == 0 and tot["tid_mm"] == 0 and tot["ships_mm"] == 0 and tot["ang_max"] <= 1e-3)
print(f"\n[exp25-parity] decode slots={tot['slots']} | launch_mm={tot['launch_mm']} "
      f"tid_mm={tot['tid_mm']} ships_mm={tot['ships_mm']} ang_max={tot['ang_max']:.3e}", flush=True)
if first_bad:
    b, s, me, p, field, jv, nv = first_bad
    print(f"[exp25-parity] FIRST divergence: board={b} step={s} seat={me} planet={p} field={field} jax={jv} np={nv}", flush=True)
print(f"[exp25-parity] {'PASS' if ok else 'FAIL'}", flush=True)
sys.exit(0 if ok else 1)
