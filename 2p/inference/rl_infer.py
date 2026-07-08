"""exp025 v1 (simplefrac) numpy inference port (deploy path; mirrors the daemon make_greedy simplefrac
branch == jax greedy == exp25 train decode).

exp025 v1 = exp24 self-play train_league (clipped-Gaussian dist + winrate-gated league + capped F=256
TRAIN-ONLY env, FLEET_CAP_PER_PLAYER=128) + v41 model (OrbitNet19 E=128 n_layers=6 n_heads=4 + SimpleFracMLP
[emb || emb_tid || gemb]). RANDOM init (no IL warm-start). The frac head, the OrbitNet19 trunk, ALL physics /
lead / first-hit / edge / forecast helpers, and n_heads=4 are BORROWED VERBATIM from the parity-verified v41
lineage (2p/experiments/experiment-020-imitate/v41/rl_infer.py) -- they are byte-identical between v41 and
exp25 (same frac, trunk, heads, weight-key convention). cfracgauss_forward(emb,emb_tid,gemb) IS exp25's
SimpleFracMLP: x=concat([emb, emb_tid, broadcast(gemb)],-1) (P,3E=384) -> gelu(x@frac/mlp_in)+ -> @frac/mlp_out
-> (mu,sigma); greedy f=clip(mu,0,1).

ONLY TWO things differ from v41 (verified by diffing exp25 jax_env/env.basic_features vs the v41 numpy
basic_features -- they are functionally IDENTICAL: same static (P,23), ts (P,50,7), glob (28,), econ_curves
(50,2); the stale "(P,21)/(P,50,6)/glob 32" comments in env.py / agent_meta.json are WRONG, trust the code):
  (1) SELF-CAP: exp25's train env caps in-flight fleets at FLEET_CAP_PER_PLAYER=128 per player
      (jax_env/step.py _cap_ok). The official deploy engine is UNCAPPED, so this numpy port re-applies the
      cap in decode() (owner-relative mirror of step._cap_ok for the single seat `me` that launches here)
      so deploy plays IDENTICALLY to training.
  (2) this docstring.
Everything else (features + trunk + frac + physics) is the v41 code unchanged.

ACTION: argmax pointer (self=hold) -> fraction f=clip(mu,0,1) -> ships=round(f*garrison) -> EXECUTED-count
lead solve (a partial fleet has a different speed/arrival than all-in) -> first-hit gate -> SELF-CAP.
2-tree ckpt -> flat npz (net + 'frac/'-prefixed) via export_weights.py.

v32 deltas vs v30 (FEATURE layer UNCHANGED: static 23 / ts 50,7 / glob 28 / edge 6 / edge_tid 11; ONLY the
NEW econ_curves (50,2) 5th return + the econ-CNN in the model):
  - basic_features returns a 5-tuple (static, ts, glob, mask, econ_curves); econ_curves=(50,2) =
    stack([ship_lead/2000, prod_lead/40], -1).
  - ts-CNN first conv channel 32->16 (now 16-32-64) -- handled automatically by W shapes (no code change).
  - NEW econ-CNN: _conv1d_mm(econ,5,16,'econ_c0')->...c1(32)->...c2(64) (im2col 1D conv); 3-pool = mean+max+
    attn-pool (attn query from glob, pool_heads=2); econ_pooled=concat([e_mean,e_max,e_attn]);
    econ_emb=LayerNorm(gelu(Dense(E//2)(econ_pooled))).
  - gtok = Dense(E)(concat([glob, econ_emb])) (was Dense(E)(glob)).

Mirrors (byte-for-byte the action decisions of):
  model.py   OrbitNet19.__call__(static, ts, glob, reach, mask, edge, econ) -> (tgt, emb, gemb, board, v)  # v42: 3rd=gemb
  env.py     basic_features(state, me) -> (static (P,23), ts (P,50,7), glob (28,), mask (P,), econ_curves (50,2))
  targeting  reach_solve_static(state) -> (R, ANG, TURNS, Rg)  [ONE full-garrison _validated solve]
  train.py   edge_features(state, lead=(R,ANG,TURNS)) -> (R, ANG, TURNS, edge (P,P,6))
             edge_partial(state, 0.5) -> (P,P,5)
             greedy_action(net, frac, params, state, me) -> (launch, angle, ships, tid, arrival)
             first_hit_gate(state, tid, angle, ships)

NET param map (E=256, 6-layer trunk, 8 heads d=32; flax AUTO-NAME = leftmost-constructor-first per expression,
NOT execution order; derived from model.py construction order + VERIFIED against the dumped ckpt key shapes
2026-06-17). econ-CNN Denses (Dense_7..10) INSERTED after FiLM (Dense_6), shifting token/gtok/edge-bias/trunk/
post-trunk by +6 (4 extra Denses + the econ convs are NAMED, no Dense/Conv shift) AND LN by +1 (LayerNorm_2):
  ts-CNN:  Conv_0(7->16) Conv_1(16->32) Conv_2(32->64)
  ts attn-pool (pool_heads=2, pd=C//2=32): pq Dense_0(static 23->64) pk Dense_1(64->64) pv Dense_2(64->64)
  ts_emb:  Dense_3(3C=192->128); LayerNorm_0
  static-enc: Dense_5(23->128 INNER) Dense_4(128->128 OUTER); LayerNorm_1
  FiLM:    Dense_6(glob 28 -> 2*256=512, zero-init)
  econ-CNN: econ_c0(5*2->16) econ_c1(5*16->32) econ_c2(5*32->64) [NAMED -> no Conv_N consumed]
  econ attn-pool (pool_heads=2, epd=eC//2=32): eq Dense_7(glob 28->64) ek Dense_8(64->64) ev Dense_9(64->64)
  econ_emb: Dense_10(3eC=192->128); LayerNorm_2
  token MLP: Dense_12(256->256 INNER) Dense_11(256->256 OUTER)
  gtok:    Dense_13([glob 28 || econ_emb 128]=156 -> 256)
  edge-bias EdgeMLP: Dense_15(6->32 INNER) Dense_14(32->8=H OUTER)
  trunk L0: LN_3(pre-attn) Dense_16(qkv 256->768) Dense_17(out 256->256) LN_4(pre-ffn)
            Dense_19(ffn-in 256->512 INNER) Dense_18(ffn-out 512->256 OUTER)
  trunk L1..L5: +4 Dense / +2 LN each (Dense_20..39, LN_5..14)
  final LN: LayerNorm_15
  board pool (ATTN-ONLY, pool_heads=2, bd=E//2=64): bq Dense_40(gemb 128->128) bk Dense_41(emb) bv Dense_42(emb)
  pointer (v42): q2 Dense_43(emb E=128->128) k2 Dense_44(emb 128->128); ptr-EdgeMLP Dense_46(6->16 INNER) Dense_45(16->1 OUTER)
  value:    Dense_48(board 2E=256->128 INNER) Dense_47(128->1 OUTER)
  frac (v42 SimpleFracMLP, EXPLICIT names): mlp_in([gemb 128 || emb_tid 128]=256 -> 256) mlp_out(256->2)

Physics/lead/gate primitives VERBATIM from the parity-verified exp17/exp19 lineage.
flax nn.gelu default approximate=True (tanh) -> gelu() here is the tanh approximation (confirmed).
"""
import math
import numpy as np

SUN_X = SUN_Y = 50.0
SUN_R = 10.0
MAX_SPEED = 6.0
LAUNCH_CLEARANCE = 0.1
EPISODE_STEPS = 500
ROTATION_LIMIT = 50.0
COMET_SPAWN_STEPS = (50, 150, 250, 350, 450)
FORECAST_H = 50
_LOG1000 = math.log(1000.0)
_SQRT2PI = math.sqrt(2.0 / math.pi)
BOARD = 100.0
H_MAX_FIRSTHIT = 100      # comet period; walk horizon (matches jax_env/step.py)
GATE_H = 50
LEAD_POS_TOL = 0.5
_DIAG = 100.0 * 1.4142135623730951
E_EDGE = 6
FLEET_CAP_PER_PLAYER = 128   # exp25 train-only per-player in-flight cap (jax_env/step.py _cap_ok); re-applied
                             # here because the official deploy engine is UNCAPPED.


def load_weights(path):
    z = np.load(path)
    return {k: z[k].astype(np.float32) for k in z.files}


def gelu(x):
    # flax nn.gelu default approximate=True -> tanh approximation
    return 0.5 * x * (1.0 + np.tanh(_SQRT2PI * (x + 0.044715 * x ** 3)))


def fleet_speed(ships):
    n = np.clip(np.asarray(ships, np.float32), 1.0, 1000.0)
    sp = 1.0 + (MAX_SPEED - 1.0) * np.power(np.log(n) / _LOG1000, 1.5)
    return np.where(np.asarray(ships) <= 1, np.float32(1.0), sp).astype(np.float32)


def seg_hits_sun(ax, ay, bx, by):
    dx = bx - ax; dy = by - ay
    L2 = dx * dx + dy * dy
    safe = L2 > 1e-12
    t = np.where(safe, ((SUN_X - ax) * dx + (SUN_Y - ay) * dy) / np.where(safe, L2, 1.0), 0.0)
    t = np.clip(t, 0.0, 1.0)
    fx = ax + t * dx; fy = ay + t * dy
    return (SUN_X - fx) ** 2 + (SUN_Y - fy) ** 2 < SUN_R * SUN_R


def _ln_named(x, W, name, eps=1e-6):
    mu = x.mean(-1, keepdims=True); var = x.var(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * W[name + "/scale"] + W[name + "/bias"]


def _conv1d_mm(x, K, kernel, bias):
    """SAME-padded stride-1 1D correlation via im2col + matmul (econ-CNN; mirrors model._conv1d_mm).
    x (L,C_in); kernel (K*C_in, C_out) C-order; bias (C_out,)."""
    L, C_in = x.shape
    lo = K // 2; hi = (K - 1) - lo
    xp = np.pad(x, ((lo, hi), (0, 0)))
    idx = np.arange(L)[:, None] + np.arange(K)[None, :]
    cols = xp[idx].reshape(L, K * C_in)
    return cols @ kernel + bias


def _conv1d_same(x, kernel, bias, dilation=1):
    P, L, _ = x.shape
    k = kernel.shape[0]
    eff = (k - 1) * dilation + 1
    pad = eff - 1
    pl = pad // 2; pr = pad - pl
    xp = np.pad(x, ((0, 0), (pl, pr), (0, 0)))
    out = np.zeros((P, L, kernel.shape[2]), np.float32)
    for i in range(k):
        out += xp[:, i * dilation:i * dilation + L, :] @ kernel[i]
    return out + bias


def _softmax_last(x):
    x = x - x.max(-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(-1, keepdims=True)


# ===================== physics / first-hit / lead solve (verbatim lineage) =====================


def _future_positions(arr, me, exists, H):
    """env._future_positions: per-planet future (x,y)/100 over next H turns, canonical reflect me==1."""
    p_x = arr["p_x"]; p_y = arr["p_y"]; av = float(arr["av"])
    orb_r = arr["p_orbital_r"]; orb_a = arr["p_orbital_a"]
    cpx = arr["p_comet_path_x"]; cpy = arr["p_comet_path_y"]; cidx0 = arr["p_comet_idx"]
    is_comet = arr["p_is_comet"]; is_orb = arr["p_is_orbiting"]
    P = p_x.shape[0]
    k_f = np.arange(1, H + 1, dtype=np.float32)
    k_i = np.arange(1, H + 1, dtype=np.int32)
    a_k = orb_a[:, None] + av * k_f[None, :]
    orb_x = SUN_X + orb_r[:, None] * np.cos(a_k)
    orb_y = SUN_Y + orb_r[:, None] * np.sin(a_k)
    L = cpx.shape[1]
    cidx = np.clip(cidx0[:, None] + k_i[None, :], 0, L - 1)
    com_x = np.take_along_axis(cpx, cidx, axis=1)
    com_y = np.take_along_axis(cpy, cidx, axis=1)
    stat_x = np.broadcast_to(p_x[:, None], (P, H))
    stat_y = np.broadcast_to(p_y[:, None], (P, H))
    fx = np.where(is_comet[:, None], com_x, np.where(is_orb[:, None], orb_x, stat_x))
    fy = np.where(is_comet[:, None], com_y, np.where(is_orb[:, None], orb_y, stat_y))
    if me == 1:
        fx = 100.0 - fx; fy = 100.0 - fy
    pos = np.stack([fx / 100.0, fy / 100.0], axis=-1).astype(np.float32)
    return pos * exists[:, :, None]


def _swept_pair_hit(ax, ay, bx, by, p0x, p0y, p1x, p1y, r):
    """Continuous swept-pair collision fleet A->B vs planet P0->P1 within r. Mirrors physics.swept_pair_hit."""
    d0x = ax - p0x; d0y = ay - p0y
    dvx = (bx - ax) - (p1x - p0x); dvy = (by - ay) - (p1y - p0y)
    a = dvx * dvx + dvy * dvy
    b = 2.0 * (d0x * dvx + d0y * dvy)
    c = d0x * d0x + d0y * d0y - r * r
    disc = b * b - 4.0 * a * c
    sq = np.sqrt(np.maximum(disc, 0.0))
    safe = a > 1e-12
    two_a = np.where(safe, 2.0 * a, 1.0)
    t1 = np.where(safe, (-b - sq) / two_a, 0.0)
    t2 = np.where(safe, (-b + sq) / two_a, 0.0)
    moving_hit = (disc >= 0.0) & (t2 >= 0.0) & (t1 <= 1.0)
    static_hit = c <= 0.0
    return np.where(safe, moving_hit, static_hit)


def _in_board(x, y):
    return (x >= 0.0) & (x <= BOARD) & (y >= 0.0) & (y <= BOARD)


def _planet_traj(arr, H):
    """Planet (x,y) at relative turns 0..H -> (H+1,P) each. Mirrors jax_env/step._planet_traj."""
    px = arr["p_x"]; py = arr["p_y"]; P = px.shape[0]
    orbr = arr["p_orbital_r"]; orba = arr["p_orbital_a"]
    isorb = arr["p_is_orbiting"]; iscom = arr["p_is_comet"]
    cpx = arr["p_comet_path_x"]; cpy = arr["p_comet_path_y"]; cidx = arr["p_comet_idx"]
    av = arr["av"]; L = cpx.shape[1]
    ks = np.arange(H + 1).astype(np.float32)
    ang = orba[None, :] + av * ks[:, None]
    orb_x = SUN_X + orbr[None, :] * np.cos(ang); orb_y = SUN_Y + orbr[None, :] * np.sin(ang)
    ci = np.clip(cidx[None, :] + np.arange(H + 1)[:, None], 0, L - 1)
    pcol = np.arange(P)[None, :]
    com_x = cpx[pcol, ci]; com_y = cpy[pcol, ci]
    bx = np.broadcast_to(px[None, :], (H + 1, P)); by = np.broadcast_to(py[None, :], (H + 1, P))
    tx = np.where(iscom[None, :], com_x, np.where(isorb[None, :], orb_x, bx))
    ty = np.where(iscom[None, :], com_y, np.where(isorb[None, :], orb_y, by))
    return tx, ty


def predict_first_hits(arr, sx, sy, angle, ships, H=H_MAX_FIRSTHIT):
    """REAL first-hit per source slot. Returns (target_slot (P,) or -1, arrival (P,)).
    Mirrors jax_env/step.predict_first_hits."""
    px = arr["p_x"]; P = px.shape[0]
    p_mask = np.asarray(arr["p_mask"], bool); p_radius = arr["p_radius"]
    tx, ty = _planet_traj(arr, H)
    sp = fleet_speed(ships)
    vx = np.cos(angle) * sp; vy = np.sin(angle) * sp
    done = np.zeros(P, bool); slot = np.full(P, -1, np.int32)
    kind = np.zeros(P, np.int32); arrv = np.full(P, -1, np.int32)
    for k in range(1, H + 1):
        ax = sx + (k - 1.0) * vx; ay = sy + (k - 1.0) * vy
        bxx = sx + k * vx; byy = sy + k * vy
        hit = _swept_pair_hit(ax[:, None], ay[:, None], bxx[:, None], byy[:, None],
                              tx[k - 1][None, :], ty[k - 1][None, :], tx[k][None, :], ty[k][None, :],
                              p_radius[None, :]) & p_mask[None, :]
        any_hit = hit.any(1)
        ev = np.where(any_hit, 1, np.where(~_in_board(bxx, byy), 2,
                      np.where(seg_hits_sun(ax, ay, bxx, byy), 3, 0))).astype(np.int32)
        new = (~done) & (ev > 0)
        slot = np.where(new, np.where(any_hit, np.argmax(hit, axis=1).astype(np.int32), -1), slot)
        kind = np.where(new, ev, kind); arrv = np.where(new, k, arrv)
        done = done | (ev > 0)
        if done.all():
            break
    return np.where(kind == 1, slot, -1).astype(np.int32), arrv.astype(np.int32)


def _planet_pos_grid(arr, kgrid):
    """(P,P) position of planet[col] at relative turn kgrid[i,j], using the ENV _planet_traj formula."""
    px = arr["p_x"]; py = arr["p_y"]
    orbr = arr["p_orbital_r"]; orba = arr["p_orbital_a"]
    isorb = arr["p_is_orbiting"]; iscom = arr["p_is_comet"]
    cpx = arr["p_comet_path_x"]; cpy = arr["p_comet_path_y"]; cidx = arr["p_comet_idx"]
    av = arr["av"]; P = px.shape[0]; L = cpx.shape[1]
    kf = kgrid.astype(np.float32)
    a = orba[None, :] + av * kf
    ox = SUN_X + orbr[None, :] * np.cos(a); oy = SUN_Y + orbr[None, :] * np.sin(a)
    ci = np.clip(cidx[None, :] + kgrid.astype(np.int32), 0, L - 1)
    cols = np.broadcast_to(np.arange(P)[None, :], (P, P))
    comx = cpx[cols, ci]; comy = cpy[cols, ci]
    bx = np.broadcast_to(px[None, :], (P, P)); by = np.broadcast_to(py[None, :], (P, P))
    tx = np.where(iscom[None, :], comx, np.where(isorb[None, :], ox, bx))
    ty = np.where(iscom[None, :], comy, np.where(isorb[None, :], oy, by))
    return tx, ty


def _lead_core(arr, ships_src):
    """(P,P) lead solve at ships (P,) per-source OR (P,P) per-pair -> (ang, turns int32, sun_blk, conv).
    Mirrors jax_env/targeting.solve_lead's 5-iter fixed point."""
    px = arr["p_x"]; py = arr["p_y"]; pr = arr["p_radius"]
    orbr = arr["p_orbital_r"]; orba = arr["p_orbital_a"]
    isorb = arr["p_is_orbiting"]; iscom = arr["p_is_comet"]
    cpx = arr["p_comet_path_x"]; cpy = arr["p_comet_path_y"]; cidx = arr["p_comet_idx"]
    av = arr["av"]; P = px.shape[0]; L = cpx.shape[1]
    sx = px[:, None]; sy = py[:, None]; sr = pr[:, None]; trad = pr[None, :]
    sp = fleet_speed(ships_src)
    sp = sp[:, None] if sp.ndim == 1 else sp
    lc = sr + LAUNCH_CLEARANCE

    def estimate(tx, ty):
        ang = np.arctan2(ty - sy, tx - sx); ca = np.cos(ang); sa = np.sin(ang)
        lx = sx + ca * lc; ly = sy + sa * lc
        center_d = np.sqrt((tx - sx) ** 2 + (ty - sy) ** 2)
        hit_d = np.maximum(0.0, center_d - lc - trad)
        ex = lx + ca * hit_d; ey = ly + sa * hit_d
        sun_blk = seg_hits_sun(lx, ly, ex, ey)
        turns = np.maximum(1, np.ceil(hit_d / sp).astype(np.int32))
        return ang, turns, sun_blk

    def predict(steps):
        eff = np.maximum(steps - 1, 0).astype(np.float32)
        a = orba[None, :] + av * eff
        ox = SUN_X + orbr[None, :] * np.cos(a); oy = SUN_Y + orbr[None, :] * np.sin(a)
        nidx = np.clip(cidx[None, :] + eff.astype(np.int32), 0, L - 1)
        comx = np.take_along_axis(cpx[None, :, :], nidx[:, :, None], axis=2)[:, :, 0]
        comy = np.take_along_axis(cpy[None, :, :], nidx[:, :, None], axis=2)[:, :, 0]
        tx = np.where(iscom[None, :], comx, np.where(isorb[None, :], ox, px[None, :]))
        ty = np.where(iscom[None, :], comy, np.where(isorb[None, :], oy, py[None, :]))
        return tx, ty

    tx = np.broadcast_to(px[None, :], (P, P)); ty = np.broadcast_to(py[None, :], (P, P))
    ang, turns, sun_blk = estimate(tx, ty)
    prev_turns = turns; prev_tx = tx; prev_ty = ty
    for _ in range(5):
        prev_turns = turns; prev_tx = tx; prev_ty = ty
        tx, ty = predict(turns)
        ang, turns, sun_blk = estimate(tx, ty)
    pos_d = np.sqrt((tx - prev_tx) ** 2 + (ty - prev_ty) ** 2)
    conv = (np.abs(turns - prev_turns) <= 1) & (pos_d < LEAD_POS_TOL)
    return ang, turns, sun_blk, conv


def _swept_verify(arr, ang, turns, sp_row):
    """(P,P) bool: env swept_pair_hit fires near the lead arrival. sp_row=(P,1) per-source or (P,P)."""
    px = arr["p_x"]; py = arr["p_y"]; pr = arr["p_radius"]; P = px.shape[0]
    ca = np.cos(ang); sa = np.sin(ang)
    lc = pr[:, None] + LAUNCH_CLEARANCE
    spx = px[:, None] + lc * ca; spy = py[:, None] + lc * sa
    vx = sp_row * ca; vy = sp_row * sa
    trad = pr[None, :]
    verified = np.zeros((P, P), bool)
    for dk in (-1, 0, 1):
        k = np.maximum(turns + dk, 1)
        fax = spx + (k - 1) * vx; fay = spy + (k - 1) * vy
        fbx = spx + k * vx; fby = spy + k * vy
        tax, tay = _planet_pos_grid(arr, k - 1)
        tbx, tby = _planet_pos_grid(arr, k)
        verified |= _swept_pair_hit(fax, fay, fbx, fby, tax, tay, tbx, tby, trad)
    return verified


def _validated(arr, ships_pp, extra_ok=None):
    """Full validation for a ships grid ((P,) per-source or (P,P) per-pair) -> (R, ang, turns).
    Mirrors jax_env/targeting._validated."""
    step = arr["step"]
    remaining = EPISODE_STEPS - 1 - step
    iscom = arr["p_is_comet"]; cidx = arr["p_comet_idx"]; clen = arr["p_comet_len"]
    ang, turns, sun_blk, conv = _lead_core(arr, ships_pp)
    comet_alive = (~iscom[None, :]) | ((cidx[None, :] + turns) < clen[None, :])
    sp = fleet_speed(ships_pp)
    sp = sp[:, None] if sp.ndim == 1 else sp
    sv = _swept_verify(arr, ang, turns, sp)
    spp = ships_pp[:, None] if ships_pp.ndim == 1 else ships_pp
    R = (~sun_blk) & conv & (turns <= remaining) & arr["p_mask"][None, :] & comet_alive & sv & (spp > 0)
    if extra_ok is not None:
        R = R & extra_ok
    return R, ang, turns.astype(np.int32)


def _gather_garrison(gar_raw, turns):
    """gar_raw (P,H) RAW target garrison; turns (P,P) -> (P,P) garrison at arrival (idx turns-1)."""
    Hh = gar_raw.shape[1]
    idx = np.clip(turns - 1, 0, Hh - 1)
    return gar_raw[np.arange(gar_raw.shape[0])[None, :], idx]


# ===================== seat-independent forecast =====================


def _forecast(arr, H=FORECAST_H):
    """SEAT-INDEPENDENT passive projection. Returns (ships_raw (P,H), owner_h (P,H) int, exists (P,H)).
    Mirrors jax_env/env._forecast."""
    P = arr["p_x"].shape[0]
    step = arr["step"]
    turns_remain = arr["f_arrival"] - step
    has_t = arr["f_mask"] & (arr["f_target"] >= 0) & (turns_remain >= 1) & (turns_remain <= H)
    inc0 = np.zeros((H, P), np.float32); inc1 = np.zeros((H, P), np.float32)
    a_idx = np.clip(turns_remain - 1, 0, H - 1)
    t_idx = np.clip(arr["f_target"], 0, P - 1)
    fs = arr["f_ships"].astype(np.float32)
    fo = arr["f_owner"]
    sel0 = has_t & (fo == 0); sel1 = has_t & (fo == 1)
    np.add.at(inc0, (a_idx[sel0], t_idx[sel0]), fs[sel0])
    np.add.at(inc1, (a_idx[sel1], t_idx[sel1]), fs[sel1])
    p_prod = arr["p_prod"].astype(np.float32)
    ships = arr["p_ships"].astype(np.float32).copy()
    owner = arr["p_owner"].astype(np.int32).copy()
    ships_h = np.zeros((P, H), np.float32); owner_h = np.zeros((P, H), np.int32)
    for h in range(H):
        s0 = inc0[h]; s1 = inc1[h]
        ships = ships + np.where((owner >= 0) & arr["p_mask"], p_prod, 0.0)
        both = (s0 > 0) & (s1 > 0)
        tie = both & (s0 == s1)
        no_atk = (s0 == 0) & (s1 == 0)
        surv_owner = np.where(s0 > 0, np.where(s1 > 0, np.where(s0 >= s1, 0, 1), 0),
                              np.where(s1 > 0, 1, -1)).astype(np.int32)
        surv = np.where(both, np.abs(s0 - s1), np.where(s0 > 0, s0, s1))
        surv = np.where(tie, 0.0, surv)
        has_s = (~no_atk) & (~tie)
        same = has_s & (surv_owner == owner)
        capture = has_s & (surv_owner != owner) & (surv > ships)
        repel = has_s & (surv_owner != owner) & (surv <= ships)
        ships = np.where(same, ships + surv,
                np.where(capture, surv - ships,
                np.where(repel, ships - surv, ships)))
        owner = np.where(capture, surv_owner, owner)
        ships_h[:, h] = ships; owner_h[:, h] = owner
    hrange = np.arange(H, dtype=np.int32)
    alive = (~arr["p_is_comet"][:, None]) | ((arr["p_comet_idx"][:, None] + hrange[None, :] + 1) < arr["p_comet_len"][:, None])
    exists = (arr["p_mask"][:, None] & alive).astype(np.float32)
    return ships_h * exists, owner_h, exists


def garrison_forecast_raw(arr, H=FORECAST_H):
    return _forecast(arr, H)[0]


def _planet_velocity(arr, me):
    """(vx, vy) per-turn velocity /MAX_SPEED, reflect sign flip for me==1, expiring-comet clamp."""
    na = arr["p_orbital_a"] + arr["av"]
    orb_nx = SUN_X + arr["p_orbital_r"] * np.cos(na)
    orb_ny = SUN_Y + arr["p_orbital_r"] * np.sin(na)
    L = arr["p_comet_path_x"].shape[1]
    nidx = np.clip(arr["p_comet_idx"] + 1, 0, L - 1)
    com_nx = np.take_along_axis(arr["p_comet_path_x"], nidx[:, None], axis=1)[:, 0]
    com_ny = np.take_along_axis(arr["p_comet_path_y"], nidx[:, None], axis=1)[:, 0]
    nx = np.where(arr["p_is_comet"], com_nx, np.where(arr["p_is_orbiting"], orb_nx, arr["p_x"]))
    ny = np.where(arr["p_is_comet"], com_ny, np.where(arr["p_is_orbiting"], orb_ny, arr["p_y"]))
    expiring = arr["p_is_comet"] & ((arr["p_comet_idx"] + 1) >= arr["p_comet_len"])
    nx = np.where(expiring, arr["p_x"], nx)
    ny = np.where(expiring, arr["p_y"], ny)
    vx = (nx - arr["p_x"]) / MAX_SPEED
    vy = (ny - arr["p_y"]) / MAX_SPEED
    sgn = -1.0 if me == 1 else 1.0
    return vx * sgn, vy * sgn


# ===================== v32 features (static 23 / ts 50,7 / glob 28 / econ_curves 50,2) =====================


def basic_features(arr, me, fc=None, H=FORECAST_H):
    """v32 (= v30 features + NEW econ_curves): (static (P,23), ts (P,50,7), glob (28,), mask (P,),
    econ_curves (50,2)). Mirrors jax_env/env.basic_features. `fc` = optional precomputed _forecast(arr) =
    (ships_raw, owner_h, exists), seat-independent."""
    f32 = np.float32
    m = arr["p_mask"]
    P = arr["p_x"].shape[0]
    owner = arr["p_owner"]
    me_i = int(me); opp_i = 1 - me_i
    is_mine = (owner == me_i) & m
    is_opp = (owner != me_i) & (owner >= 0) & m
    is_neu = (owner == -1) & m & ~arr["p_is_comet"]
    is_comet = arr["p_is_comet"] & m
    is_orb = arr["p_is_orbiting"] & m
    is_static = m & ~arr["p_is_comet"] & ~arr["p_is_orbiting"]
    ships_f = arr["p_ships"].astype(f32)
    pr_f = arr["p_prod"].astype(f32)
    cx = (100.0 - arr["p_x"]) if me_i == 1 else arr["p_x"]
    cy = (100.0 - arr["p_y"]) if me_i == 1 else arr["p_y"]
    remaining = f32(EPISODE_STEPS - 1) - f32(arr["step"])
    comet_rem = np.maximum(arr["p_comet_len"] - arr["p_comet_idx"], 0).astype(f32)
    life = np.where(arr["p_is_comet"], comet_rem, remaining)
    life = np.clip(life, 0.0, 100.0) / 100.0
    inv_log_ships = 1.0 / (np.log(ships_f + 1.0) + 1.0)
    vx, vy = _planet_velocity(arr, me_i)
    static_base = np.stack([
        is_mine.astype(f32), is_opp.astype(f32), is_neu.astype(f32),
        is_static.astype(f32), is_orb.astype(f32), is_comet.astype(f32),
        cx / 100.0, cy / 100.0, arr["p_radius"] / 5.0,
        ships_f / 500.0, pr_f / 5.0,
        life, inv_log_ships, vx.astype(f32), vy.astype(f32),
    ], axis=-1)                                                                  # (P,15) base

    if fc is None:
        ships_raw, owner_h, exists = _forecast(arr, H)
    else:
        ships_raw, owner_h, exists = fc
    proj_ships = ships_raw / 500.0
    own_me_ts = (owner_h == me_i).astype(f32) * exists                           # v26: owner 2-ch (was signed own_sign)
    own_op_ts = ((owner_h != me_i) & (owner_h >= 0)).astype(f32) * exists
    pos = _future_positions(arr, me_i, exists, H)                                # (P,H,2)
    ramp = np.broadcast_to((np.arange(1, H + 1, dtype=f32) / f32(H))[None, :, None], (P, H, 1))
    ts = np.concatenate([proj_ships[:, :, None], own_me_ts[:, :, None], own_op_ts[:, :, None], exists[:, :, None],
                         pos, ramp], axis=-1) * m[:, None, None].astype(f32)     # (P,50,7) v26: owner 2-ch

    # --- exp21 horizon mask: forecast slot t is turn step+t+1; cut at final game turn 499. ---
    slot = np.arange(H, dtype=np.int32)
    abs_turn = arr["step"] + slot + 1                                            # (50,)
    valid = abs_turn <= (EPISODE_STEPS - 1)                                      # (50,)
    n_valid = max(float(np.sum(valid.astype(f32))), 1.0)
    last_v = (H - 1) - int(np.argmax(valid[::-1]))                               # last valid slot

    def _sgn(owner_arr):
        return np.where(owner_arr == me_i, f32(1.0), np.where(owner_arr == opp_i, f32(-1.0), f32(0.0)))

    # --- exp21 PER-PLANET forecast dynamics (-> static, 8 channels) ---
    cur_owner = arr["p_owner"]                                                   # (P,)
    act = valid[None, :] & (exists > 0)                                          # (P,50)
    flip = act & (owner_h != cur_owner[:, None])
    any_flip = np.any(flip, axis=1)                                              # (P,)
    ft = np.argmax(flip, axis=1)                                                 # (P,) first flip slot
    flip_turn = np.where(any_flip, (ft.astype(f32) + 1.0) / f32(H), f32(1.0))
    ft_owner = np.take_along_axis(owner_h, ft[:, None], axis=1)[:, 0]
    flip_player_me = np.where(any_flip & (ft_owner == me_i), f32(1.0), f32(0.0))  # v26: owner 2-ch
    flip_player_op = np.where(any_flip & (ft_owner == opp_i), f32(1.0), f32(0.0))
    secured = (~any_flip).astype(f32)
    has_act = np.any(act, axis=1)
    last_act = (H - 1) - np.argmax(act.astype(f32)[:, ::-1], axis=1)             # last active slot
    fo_owner = np.take_along_axis(owner_h, last_act[:, None], axis=1)[:, 0]
    fin_owner = np.where(has_act, fo_owner, cur_owner)                           # owner id at horizon end
    final_owner_me = (fin_owner == me_i).astype(f32)                            # v26: owner 2-ch
    final_owner_op = (fin_owner == opp_i).astype(f32)
    pct_me_hold = np.sum((act & (owner_h == me_i)).astype(f32), axis=1) / n_valid
    pct_cur_hold = np.sum((act & (owner_h == cur_owner[:, None])).astype(f32), axis=1) / n_valid
    planet_dyn = np.stack([flip_turn, flip_player_me, flip_player_op, secured, final_owner_me, final_owner_op,
                           pct_me_hold, pct_cur_hold], axis=-1)                  # (P,8) v26: owner 2-ch
    static = np.concatenate([static_base, planet_dyn], axis=-1) * m[:, None].astype(f32)  # (P,23) v26

    # --- exp21 GLOBAL econ dynamics (-> glob, 14 scalars KEPT) + v32 econ_curves (-> econ-CNN) ---
    fmine = (owner_h == me_i) & (exists > 0)                                     # (P,50)
    fopp = (owner_h == opp_i) & (exists > 0)
    ship_lead = np.sum(ships_raw * (fmine.astype(f32) - fopp.astype(f32)), axis=0)        # (50,)
    prod_lead = np.sum(pr_f[:, None] * (fmine.astype(f32) - fopp.astype(f32)), axis=0)    # (50,)

    def _econ_feats(L):
        s = np.sign(L).astype(f32)
        s0 = s[0]
        lead_pct = np.sum(np.where(valid, (L >= 0).astype(f32), 0.0)) / n_valid
        flp = valid & (s != s0)
        anyf = np.any(flp)
        ftn = int(np.argmax(flp))
        next_flip = np.where(anyf, (f32(ftn) + 1.0) / f32(H), f32(1.0))
        sec = (~anyf).astype(f32)
        sl = s[last_v]
        return np.stack([lead_pct.astype(f32), next_flip.astype(f32), sec.astype(f32),  # v26: sign 2-ch
                         (s0 > 0).astype(f32), (s0 < 0).astype(f32),
                         (sl > 0).astype(f32), (sl < 0).astype(f32)])            # (7,)

    econ = np.concatenate([_econ_feats(ship_lead), _econ_feats(prod_lead)]).astype(f32)   # (14,) v26: sign 2-ch
    # v32: econ-CNN curve input (ADDITIONAL to the 14 scalars): full lead curves
    econ_curves = np.stack([ship_lead / 2000.0, prod_lead / 40.0], axis=-1).astype(f32)   # (50,2)

    step_f = f32(arr["step"])
    g_turn = step_f / 500.0
    g_left = (f32(EPISODE_STEPS - 1) - step_f) / 500.0
    g_rot = f32(arr["av"]) * 10.0
    spawns = np.asarray(COMET_SPAWN_STEPS, dtype=f32)
    g_cd = float(np.clip(np.min(np.where(spawns > step_f, spawns - step_f, 1000.0)) / 100.0, 0.0, 1.0))
    fl_mine = (arr["f_owner"] == me_i) & arr["f_mask"]
    fl_opp = (arr["f_owner"] != me_i) & (arr["f_owner"] >= 0) & arr["f_mask"]
    fsh = arr["f_ships"].astype(f32)
    ts_me = float(np.sum(np.where(is_mine, ships_f, 0.0)) + np.sum(np.where(fl_mine, fsh, 0.0)))
    ts_op = float(np.sum(np.where(is_opp, ships_f, 0.0)) + np.sum(np.where(fl_opp, fsh, 0.0)))
    tp_me = float(np.sum(np.where(is_mine, pr_f, 0.0))); tp_op = float(np.sum(np.where(is_opp, pr_f, 0.0)))
    total_pl = float(np.sum(m)); comets_on = float(np.sum(is_comet))
    rem = max(float(EPISODE_STEPS - 1) - float(arr["step"]), 0.0)
    pf_me = tp_me * rem; pf_op = tp_op * rem
    afl_me = float(np.sum(fl_mine.astype(f32))); afl_op = float(np.sum(fl_opp.astype(f32)))
    glob = np.concatenate([np.asarray([
        g_turn, g_left, g_rot, g_cd,                                            # phase (4)
        ts_me / 2000.0, ts_op / 2000.0,                                         # v30: ship totals (2) [diff REMOVED]
        tp_me / 40.0, tp_op / 40.0,                                             # v30: prod totals (2) [diff REMOVED]
        pf_me / 10000.0, pf_op / 10000.0,                                       # v30: pf (2) [diff REMOVED]
        total_pl / 20.0, comets_on / 5.0,                                       # v30: territory (2) [neu_rem REMOVED]
    ], dtype=f32), econ,                                                        # (12 + 14 = 26)
        np.asarray([afl_me / 256.0, afl_op / 256.0], dtype=f32)])               # v25 ABLATION: afl /256 -> 28
    return static, ts, glob, m, econ_curves


# ===================== exp21 reach + edge =====================


def reach_solve_static(arr):
    """exp21 hot-path: ONE full-garrison _validated solve -> (R, ANG, TURNS) AND pointer-legality Rg
    (moving-target override). Mirrors jax_env/targeting.reach_solve_static."""
    P = arr["p_x"].shape[0]
    ships_pp = np.broadcast_to(arr["p_ships"][:, None], (P, P))
    R, ANG, TURNS = _validated(arr, ships_pp)
    moving = arr["p_is_orbiting"] | arr["p_is_comet"]
    Rg = np.where(moving[None, :], arr["p_mask"][None, :].astype(bool), R)
    return R, ANG, TURNS, Rg


def edge_features(arr, fc=None, lead=None):
    """Per ordered pair (s->j) features for the ALL-IN fleet (ships = source garrison). SEAT-INDEPENDENT.
    Returns (R, ANG, TURNS, edge (P,P,6)). edge = [dist/diag, reach, arrival/50, eff/500, margin/500,
    margin>0] (the geo channels gated by reach). Mirrors train.edge_features."""
    f32 = np.float32
    P = arr["p_x"].shape[0]
    ships = arr["p_ships"]                                       # all-in = full garrison
    if lead is None:
        ships_pp = np.broadcast_to(ships[:, None], (P, P))
        R, ANG, TURNS = _validated(arr, ships_pp)
    else:
        R, ANG, TURNS = lead
    gar_raw = fc[0] if fc is not None else garrison_forecast_raw(arr)            # (P,H)
    eff = _gather_garrison(gar_raw, TURNS)                                       # (P,P)
    shipsf = ships.astype(f32)
    margin = shipsf[:, None] - eff - 1.0                                         # (P,P)
    dx = arr["p_x"][:, None] - arr["p_x"][None, :]
    dy = arr["p_y"][:, None] - arr["p_y"][None, :]
    dist = np.sqrt(dx * dx + dy * dy) / _DIAG
    Rf = R.astype(f32)
    geo = np.stack([np.clip(TURNS.astype(f32), 0.0, 50.0) / 50.0,
                    eff / 500.0, margin / 500.0, (margin > 0).astype(f32)], axis=-1) * Rf[:, :, None]
    edge = np.concatenate([dist[:, :, None], Rf[:, :, None], geo], axis=-1)      # (P,P,6)
    return R, ANG, TURNS, edge


HALF_FRAC = 0.5   # exp022 v2 second operating point (besides all-in=1.0)


def edge_partial(arr, frac, fc=None):
    """exp022 v2: the 5 f-DEPENDENT edge dims at ships=max(1,round(frac*garrison)): [reach, arrival/50,
    eff/500, margin/500, can_capture], (P,P,5), geo gated by reach (matches edge_features' [Rf ‖ geo]
    MINUS the f-independent dist). Mirror of train.edge_partial."""
    f32 = np.float32
    P = arr["p_x"].shape[0]
    ships = np.maximum(1, np.round(frac * arr["p_ships"].astype(f32)).astype(np.int32))   # >=1
    ships_pp = np.broadcast_to(ships[:, None], (P, P))
    R, ANG, TURNS = _validated(arr, ships_pp)
    gar_raw = fc[0] if fc is not None else garrison_forecast_raw(arr)
    eff = _gather_garrison(gar_raw, TURNS)
    margin = ships.astype(f32)[:, None] - eff - 1.0
    Rf = R.astype(f32)
    geo = np.stack([np.clip(TURNS.astype(f32), 0.0, 50.0) / 50.0,
                    eff / 500.0, margin / 500.0, (margin > 0).astype(f32)], axis=-1) * Rf[:, :, None]
    return np.concatenate([Rf[:, :, None], geo], axis=-1)                       # (P,P,5)


# ===================== v32 OrbitNet19 forward =====================


_DBG = {}  # debug intermediates (parity isolation)


def orbitnet19_forward(static, ts, glob, reach, mask, edge, econ, W):
    """v42 OrbitNet19 -> (tgt (P,P), emb (P,128), gemb (128,), board (256=2E), v ()). Mirror of
    model.OrbitNet19.__call__ (v42: E=128, 6-layer trunk, 8 heads d=16, econ-CNN folded into gtok, board
    ATTN-ONLY = 2E; 3E ctx REMOVED -> 3rd return is gemb; pointer q2 from emb). Param indices unchanged
    from v36 (n_layers=6 -> same Dense index map)."""
    E_ = 128; Hh = 4; d = E_ // Hh; NL = 6                                      # v41: E=128, 4 heads (d=32), 6 trunk layers
    EH = E_ // 2                                                                 # 64 (ts_emb/static_emb/econ_emb)
    P = static.shape[0]; Pp = P + 1
    mbool = mask.astype(bool)

    # ---- ts encoder: conv 16/32/64 (k5 SAME) -> mean+max+attn(q from static) pool -> Dense(192->128) -> LN ----
    h = gelu(_conv1d_same(ts, W["Conv_0/kernel"], W["Conv_0/bias"]))             # (P,T,16)
    h = gelu(_conv1d_same(h, W["Conv_1/kernel"], W["Conv_1/bias"]))             # (P,T,32)
    h = gelu(_conv1d_same(h, W["Conv_2/kernel"], W["Conv_2/bias"]))             # (P,T,C=64)
    T = h.shape[1]; C = h.shape[2]
    p_mean = h.mean(axis=1); p_max = h.max(axis=1)                               # (P,C) each
    ph = 2; pd = C // ph                                                         # pool heads, head dim (=32)
    pq = (static @ W["Dense_0/kernel"] + W["Dense_0/bias"]).reshape(P, ph, pd)   # query from static
    pk = (h @ W["Dense_1/kernel"] + W["Dense_1/bias"]).reshape(P, T, ph, pd)
    pv = (h @ W["Dense_2/kernel"] + W["Dense_2/bias"]).reshape(P, T, ph, pd)
    psc = np.einsum('phd,pthd->pht', pq, pk) / math.sqrt(pd)                     # (P,ph,T)
    p_attn = np.einsum('pht,pthd->phd', _softmax_last(psc), pv).reshape(P, ph * pd)  # (P,C)
    pooled = np.concatenate([p_mean, p_max, p_attn], axis=-1)                    # (P,3C=192)
    ts_emb = _ln_named(gelu(pooled @ W["Dense_3/kernel"] + W["Dense_3/bias"]), W, "LayerNorm_0")  # (P,128)
    _DBG["ts_emb"] = ts_emb

    # ---- static encoder: Dense(E//2) gelu -> Dense(E//2) -> LN (inner=Dense_5, outer=Dense_4) ----
    se = gelu(static @ W["Dense_5/kernel"] + W["Dense_5/bias"])                  # inner (23->128)
    static_emb = _ln_named(se @ W["Dense_4/kernel"] + W["Dense_4/bias"], W, "LayerNorm_1")  # (P,128)
    _DBG["static_emb"] = static_emb

    # ---- concat [static_emb ‖ ts_emb] = (P,256) ----
    concat = np.concatenate([static_emb, ts_emb], axis=-1)                       # (P,256)
    Cc = concat.shape[-1]

    # ---- FiLM (global enters via FiLM + gtok) : Dense_6 (zero-init, 28->2*256) ----
    film = glob @ W["Dense_6/kernel"] + W["Dense_6/bias"]                        # (2C,)
    g_delta, beta = film[:Cc], film[Cc:]
    feat = (1.0 + g_delta)[None, :] * concat + beta[None, :]                     # (P,256)

    # ---- v32 econ-CNN (im2col econ_c0/c1/c2, k5 SAME) over econ(50,2) -> mean+max+ATTN(q from glob) 3-pool
    #      -> Dense_10(192->128) -> LN_2. attn-pool: eq=Dense_7, ek=Dense_8, ev=Dense_9 ----
    eh = gelu(_conv1d_mm(econ, 5, W["econ_c0/kernel"], W["econ_c0/bias"]))       # (T,16)
    eh = gelu(_conv1d_mm(eh, 5, W["econ_c1/kernel"], W["econ_c1/bias"]))        # (T,32)
    eh = gelu(_conv1d_mm(eh, 5, W["econ_c2/kernel"], W["econ_c2/bias"]))        # (T,eC=64)
    eT = eh.shape[0]; eC = eh.shape[1]
    e_mean = eh.mean(axis=0); e_max = eh.max(axis=0)                             # (eC,) each
    eph = 2; epd = eC // eph                                                     # econ pool heads, head dim (=32)
    eq = (glob @ W["Dense_7/kernel"] + W["Dense_7/bias"]).reshape(eph, epd)      # query from glob
    ek = (eh @ W["Dense_8/kernel"] + W["Dense_8/bias"]).reshape(eT, eph, epd)
    ev = (eh @ W["Dense_9/kernel"] + W["Dense_9/bias"]).reshape(eT, eph, epd)
    esc = np.einsum('hd,thd->ht', eq, ek) / math.sqrt(epd)                       # (eph,T)
    e_attn = np.einsum('ht,thd->hd', _softmax_last(esc), ev).reshape(eph * epd)  # (eC,)
    econ_pooled = np.concatenate([e_mean, e_max, e_attn], axis=-1)               # (3eC=192,)
    econ_emb = _ln_named(gelu(econ_pooled @ W["Dense_10/kernel"] + W["Dense_10/bias"]), W, "LayerNorm_2")  # (128,)
    _DBG["econ_emb"] = econ_emb

    # ---- token MLP: Dense(E)(gelu(Dense(E)(feat))) (outer=Dense_11, inner=Dense_12) ----
    tok = gelu(feat @ W["Dense_12/kernel"] + W["Dense_12/bias"])                 # inner (256->256)
    tok = (tok @ W["Dense_11/kernel"] + W["Dense_11/bias"]) * mask[:, None]      # outer (256->256)
    # ---- gtok: Dense_13([glob 28 || econ_emb 128] = 156 -> 256) ----
    gtok = (np.concatenate([glob, econ_emb]) @ W["Dense_13/kernel"] + W["Dense_13/bias"])[None, :]  # (1,256)
    x = np.concatenate([tok, gtok], axis=0)                                      # (P+1,256)
    amask = np.concatenate([mbool, np.ones((1,), bool)])

    # ---- edge bias: Dense(H)(gelu(Dense(32)(edge))) (outer=Dense_14, inner=Dense_15) -> (H,P,P) ----
    eb = gelu(edge @ W["Dense_15/kernel"] + W["Dense_15/bias"])                  # inner (6->32)
    eb = eb @ W["Dense_14/kernel"] + W["Dense_14/bias"]                          # outer (32->H)
    eb = np.transpose(eb, (2, 0, 1))                                            # (H,P,P)
    edge_bias = np.zeros((Hh, Pp, Pp), np.float32)
    edge_bias[:, :P, :P] = eb

    # ---- trunk: NL-layer pre-LN MHSA (+ edge bias). per layer: qkv=16+4L out=17+4L ffn_out=18+4L ffn_in=19+4L ----
    for L in range(NL):
        qkv_i = 16 + 4 * L; out_i = 17 + 4 * L; ffo_i = 18 + 4 * L; ffi_i = 19 + 4 * L
        xn = _ln_named(x, W, "LayerNorm_%d" % (3 + 2 * L))
        qkv = xn @ W["Dense_%d/kernel" % qkv_i] + W["Dense_%d/bias" % qkv_i]
        qh = qkv[:, :E_].reshape(Pp, Hh, d)
        kh = qkv[:, E_:2 * E_].reshape(Pp, Hh, d)
        vh = qkv[:, 2 * E_:].reshape(Pp, Hh, d)
        sc2 = np.einsum('phd,qhd->hpq', qh, kh) / math.sqrt(d) + edge_bias
        sc2 = np.where(amask[None, None, :], sc2, -1e9)
        out = np.einsum('hpq,qhd->phd', _softmax_last(sc2), vh).reshape(Pp, E_)
        x = x + (out @ W["Dense_%d/kernel" % out_i] + W["Dense_%d/bias" % out_i])
        xn = _ln_named(x, W, "LayerNorm_%d" % (4 + 2 * L))
        ff = gelu(xn @ W["Dense_%d/kernel" % ffi_i] + W["Dense_%d/bias" % ffi_i])  # inner (256->512)
        x = x + (ff @ W["Dense_%d/kernel" % ffo_i] + W["Dense_%d/bias" % ffo_i])   # outer (512->256)
    x = _ln_named(x, W, "LayerNorm_15")                                          # final trunk LN
    emb = x[:P]; gemb = x[P]

    # ---- board: ATTENTION-ONLY pooling [gtok ‖ attn-pool] = 2E (bq=Dense_40 bk=Dense_41 bv=Dense_42) ----
    bh, bd = 2, E_ // 2
    bq = (gemb @ W["Dense_40/kernel"] + W["Dense_40/bias"]).reshape(bh, bd)
    bk = (emb @ W["Dense_41/kernel"] + W["Dense_41/bias"]).reshape(P, bh, bd)
    bv = (emb @ W["Dense_42/kernel"] + W["Dense_42/bias"]).reshape(P, bh, bd)
    bsc = np.einsum('hd,phd->hp', bq, bk) / math.sqrt(bd)                        # (bh,P)
    bsc = np.where(mbool[None, :], bsc, -1e9)
    batt = _softmax_last(bsc)
    board_att = np.einsum('hp,phd->hd', batt, bv).reshape(E_)                    # (E,)
    board = np.concatenate([gemb, board_att], axis=-1)                          # (2E,)

    # ---- pointer (v42): q2=Dense_43(emb E->E), k2=Dense_44(emb); ptr-edge inner=Dense_46(6->16) outer=Dense_45(16->1) ----
    q2 = emb @ W["Dense_43/kernel"] + W["Dense_43/bias"]
    k2 = emb @ W["Dense_44/kernel"] + W["Dense_44/bias"]
    tgt = (q2 @ k2.T) / math.sqrt(E_)                                           # (P,P)
    pe = gelu(edge @ W["Dense_46/kernel"] + W["Dense_46/bias"])                  # inner (6->16)
    ptr_edge = (pe @ W["Dense_45/kernel"] + W["Dense_45/bias"])[:, :, 0]         # outer (16->1) -> (P,P)
    tgt = tgt + ptr_edge
    eye = np.eye(P, dtype=bool)
    legal = (reach & mask[None, :].astype(bool) & ~eye) | eye                    # SELF=HOLD legal
    tgt = np.where(legal, tgt, -1e9)

    # ---- value : inner=Dense_48 (board 2E->E), outer=Dense_47 (E->1) ----
    vv = (gelu(board @ W["Dense_48/kernel"] + W["Dense_48/bias"])
          @ W["Dense_47/kernel"] + W["Dense_47/bias"])[0]
    return tgt, emb, gemb, board, vv


def first_hit_gate(arr, tid, angle, ships, H=GATE_H):
    lc = arr["p_radius"] + LAUNCH_CLEARANCE
    sx = arr["p_x"] + lc * np.cos(angle)
    sy = arr["p_y"] + lc * np.sin(angle)
    fh, _ = predict_first_hits(arr, sx, sy, angle, ships, H=H)
    return fh == tid


# ===================== exp21 decode =====================


def lead_for_ships(arr, ships_src):
    """Full validated lead solve for a per-source ship count (the executed fleet sizes).
    Returns (R, ANG, TURNS) each (P,P). Mirrors jax_env/targeting.lead_for_ships."""
    P = arr["p_x"].shape[0]
    ships_pp = np.broadcast_to(ships_src[:, None], (P, P))
    return _validated(arr, ships_pp)


def cfracgauss_forward(emb, emb_tid, gemb, W):
    """v41 SimpleFracMLP: (mu,sigma) from 2-layer MLP on [emb[s] || emb_tid || gemb(broadcast)] = 3E."""
    P = emb_tid.shape[0]
    g = np.broadcast_to(gemb[None, :], (P, gemb.shape[-1]))
    x = np.concatenate([emb, emb_tid, g], axis=-1)
    inner = gelu(x @ W["frac/mlp_in/kernel"] + W["frac/mlp_in/bias"])
    o = inner @ W["frac/mlp_out/kernel"] + W["frac/mlp_out/bias"]
    mu = o[:, 0]; sigma = np.exp(np.clip(o[:, 1], -2.0, 0.0))   # exp25 SimpleFracMLP upper-clip 0.5->0.0 (sigma unused by greedy; fidelity only)
    return mu, sigma


def decode(arr, W, me):
    """exp25 simplefrac decode: argmax pointer (self=hold) -> CONTINUOUS clipped-Gaussian fraction
    (greedy = clipped mean mu) -> ships=round(f*garrison) -> EXECUTED-count lead solve (partial != all-in
    speed) -> first_hit_gate -> SELF-CAP (per-player in-flight cap, owner-relative mirror of step._cap_ok).
    Mirror of the cross-daemon make_greedy simplefrac branch == exp25 train decode. Returns
    (launch, tid, angle, ships). frac = SimpleFracMLP on [emb || emb_tid || gemb] (3E)."""
    P = arr["p_x"].shape[0]; ar = np.arange(P)
    me_i = int(me)
    # step-0 phase-feature clamp (2026-06-23): the jax training env inits from the kaggle STEP-1 obs, so
    # train state.step is always >=1; deploy's step-0 phase features (g_turn/g_left/horizon/life) are then
    # out-of-domain. Clamp the FEATURE step to >=1 to match training's domain. Orbital geometry is ALREADY
    # baked into arr by the engine using the REAL obs.step (the rotation_count quirk), so it is untouched.
    # Behaviorally a no-op (0/48 openings changed in test) but aligns deploy with training exactly.
    arr = {**arr, "step": max(1, int(arr["step"]))}
    fc = _forecast(arr)
    R, ANG, TURNS, Rg = reach_solve_static(arr)
    static, ts, glob, m, econ = basic_features(arr, me_i, fc=fc)                   # v32: + econ_curves (50,2)
    is_mine = (arr["p_owner"] == me_i) & arr["p_mask"]
    reach = Rg & is_mine[:, None]
    _, _, _, edge = edge_features(arr, fc=fc, lead=(R, ANG, TURNS))
    tgt, emb, gemb, _b, _v = orbitnet19_forward(static, ts, glob, reach, m, edge, econ, W)
    tid = np.argmax(tgt, axis=1)
    is_real = is_mine & (tid != ar) & R[ar, tid]                                  # pointer launch (all-in reach)
    emb_tid = emb[tid]
    edge_tid = edge[ar, tid]                                                       # v41: (P,6) all-in only (unused by frac)
    mu, sigma = cfracgauss_forward(emb, emb_tid, gemb, W)
    f = np.clip(mu, 0.0, 1.0)                                                     # greedy fraction = clipped mean
    garrison = arr["p_ships"]
    ships = np.clip(np.round(f * garrison.astype(np.float32)).astype(np.int32), 0, garrison)
    Rx, ANGx, TURNSx = lead_for_ships(arr, ships)                                 # EXECUTED-count solve
    angle = ANGx[ar, tid].astype(np.float32)
    cand = is_real & (ships > 0) & Rx[ar, tid]
    launch = cand & first_hit_gate(arr, tid, angle, ships)
    # --- exp25 SELF-CAP (owner-relative mirror of jax_env/step.py _cap_ok for the single seat `me`): a
    # player already holding FLEET_CAP_PER_PLAYER in-flight has further launches REJECTED this tick; a
    # multi-launch tick crossing the cap drops the excess in planet-slot order (per-owner cumsum rank).
    # The deploy engine is uncapped, so we re-apply the train-only cap here. ---
    _ex_me = int((arr["f_mask"] & (arr["f_owner"] == me_i)).sum())
    _rank_me = np.cumsum((launch & (arr["p_owner"] == me_i)).astype(np.int32)) - 1
    launch = launch & (_ex_me + _rank_me < FLEET_CAP_PER_PLAYER)
    ships = np.where(launch, ships, 0).astype(np.int32)
    return launch, tid, angle, ships


def value_of(arr, W, me):
    """V(s) for winprob recording. Mirrors train._value (reach = Rg & is_mine)."""
    me_i = int(me)
    arr = {**arr, "step": max(1, int(arr["step"]))}   # step-0 phase-feature clamp (see decode())
    fc = _forecast(arr)
    R, ANG, TURNS, Rg = reach_solve_static(arr)
    static, ts, glob, m, econ = basic_features(arr, me_i, fc=fc)                   # v32: + econ_curves (50,2)
    is_mine = (arr["p_owner"] == me_i) & arr["p_mask"]
    reach = Rg & is_mine[:, None]
    _, _, _, edge = edge_features(arr, fc=fc, lead=(R, ANG, TURNS))
    _t, _e, _g, _b, v = orbitnet19_forward(static, ts, glob, reach, m, edge, econ, W)
    return v
