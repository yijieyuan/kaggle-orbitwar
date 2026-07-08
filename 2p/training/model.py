"""exp19 v2 OrbitNet — v1 body UNCHANGED; fraction head = ONE-INFLATED Beta (Bernoulli all-in x Beta f).

ENCODERS (design locked 2026-06-11):
  ts (P,50,6)=[traj3 ⊕ pos2 ⊕ ramp] -> PLAIN Conv k5 d=1 x3, channels 8/16/32 (RF=13; v4 ablation: dilation removed)
      -> THREE pools: mean(32) + max(32) + 2-head attention pool (q from static15) (32)
      -> concat 96 -> Dense(96->128)+GELU+LN = ts_emb (P,128)
  static (P,15) -> Dense(15->96)+GELU -> Dense(96->64)+LN = static_emb (P,64)
  glob (16)     -> THREE routes simultaneously:
      (a) MLP Dense(16->64)+GELU+Dense(64->64) = glob_emb, BROADCAST-concat into every planet
      (b) FiLM Dense(16->2*256, ZERO-INIT) modulating the fused (P,256) token features
      (c) global token Dense(16->E) appended as token P+1
  (gecon20 REMOVED 2026-06-11 — was a 15v7-only pattern; nothing bypasses the trunk.)

TRUNK: token = Dense(256->128)+GELU+Dense(128->128) ×mask; (P+1,128) -> 3-layer pre-LN MHSA (4 heads)
  -> final LN; board = [gtok ‖ mean ‖ max ‖ attn-pool] = 512 (triple pooling 2026-06-11); ctx = [emb ‖ board] = 640.

HEADS (exp19): NO launch head. Pointer q=Dense(ctx),k=Dense(emb), legal = (reach&mask&~eye) | eye —
  SELF-TARGET IS THE HOLD ACTION (15v7 convention restored). CoordFracBeta (separate module, applied after
  tid is sampled): coordination attn -> (g_allin, alpha, beta); see class docstring.
  Value from board. Params = 2 trees {net, cfrac} (cfrac needs emb[tid] post-sampling).
"""
import jax
import jax.numpy as jnp
import flax.linen as nn

_ZERO = nn.initializers.zeros
K_BINS = 6


def _conv1d_mm(x, K, C_out, name):
    """SAME-padded stride-1 1D correlation via im2col + Dense (the GEMM path). Used for the v32
    econ-CNN ONLY (ported from exp22 v3). WHY: the econ curve is a small UN-BATCHED (T,C) sequence;
    XLA's conv autotuner picks a numerically-broken cuDNN *backward* algorithm for that shape -> NaN
    grads. im2col+matmul avoids cuDNN conv entirely (correct + fast) and is MATHEMATICALLY IDENTICAL
    to nn.Conv(C_out,(K,),padding='SAME'). The ts-CNN stays nn.Conv (batched over P -> a good algo)."""
    L, C_in = x.shape[-2], x.shape[-1]
    lo = K // 2; hi = (K - 1) - lo                              # lax 'SAME' stride-1 pad split
    xp = jnp.pad(x, ((lo, hi), (0, 0)))                         # (L+K-1, C_in)
    idx = jnp.arange(L)[:, None] + jnp.arange(K)[None, :]       # (L, K) window indices
    cols = xp[idx].reshape(L, K * C_in)                         # im2col (L, K*C_in)
    return nn.Dense(C_out, name=name)(cols)                     # (L, C_out)


class CoordFracBeta(nn.Module):
    """exp19 v2 (2026-06-11 redesign): ONE-INFLATED Beta head — Bernoulli(ALL-IN) × Beta(f) — with
    the SAME cross-planet COORDINATION attention as v1's CoordFrac6 (assignment token z over
    intending planets; co-attackers of one target share emb_tid and deconflict the joint
    allocation). The precise-kill Bernoulli branch is REMOVED (user decision: pure fraction
    semantics; the kill solver stays in targeting.py only as a reach probe).

    Per-planet outputs (g_allin, alpha, beta):
      g_allin       ALL-IN logit (Bernoulli; NO mask — the full garrison is always a legal count)
      alpha, beta = 1 + softplus(raw) >= 1  ->  UNIMODAL Beta over f∈(0,1) (no U-shaped mass at 0/1)
    allin -> ships = ENTIRE garrison; else f ~ Beta(alpha,beta), ships = round(f·garrison) (ENV-side
    deterministic map — PPO logp is on the SAMPLED continuous f, never the rounded ships). Greedy:
    all-in if g_allin>0 else Beta MEAN alpha/(alpha+beta) (MODE degenerate at the fp32 softplus
    floor alpha=beta=1). WHY the atom: P(f=1)=0 for any density; round(f·g)=g needs f>1-1/(2g) and
    holding that tail mass needs alpha~4.6·g (geometric in garrison) against a vanishing
    dlogp/dalpha and the ent_frac pull — exact full-send is near-unlearnable from the Beta tail
    alone. 0/hold already has its own atom (pointer self-target); this restores symmetry and keeps
    the ablation fair vs v1's K4 grid where full-send g is a first-class bin."""
    E: int = 64
    n_heads: int = 4

    @nn.compact
    def __call__(self, ctx, emb_tid, intend):         # ctx (P,640), emb_tid (P,128), intend (P,) bool
        P = ctx.shape[0]; H = self.n_heads; d = self.E // H
        ib = intend[:, None].astype(ctx.dtype)
        z = nn.Dense(self.E)(jnp.concatenate([ctx, emb_tid], axis=-1))     # (P,E) assignment token
        zn = nn.LayerNorm()(z)
        qkv = nn.Dense(3 * self.E)(zn)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        q = q.reshape(P, H, d); k = k.reshape(P, H, d); v = v.reshape(P, H, d)
        scores = jnp.einsum('phd,qhd->hpq', q, k) / jnp.sqrt(float(d))     # (H,P,P)
        kmask = jnp.broadcast_to(intend[None, None, :], (H, P, P))         # attend only INTENDING planets
        scores = jnp.where(kmask, scores, -1e9)                            # all-hold -> uniform (no NaN)
        attn = jax.nn.softmax(scores, axis=-1)
        out = jnp.einsum('hpq,qhd->phd', attn, v).reshape(P, self.E)
        h_coord = z + nn.Dense(self.E)(out)                                # (P,E)
        x = jnp.concatenate([ctx, emb_tid, ib, h_coord], axis=-1)
        o = nn.Dense(3)(nn.gelu(nn.Dense(2 * self.E)(x)))                  # (P,3): all-in logit + raw a,b
        g_allin = o[:, 0]
        alpha = 1.0 + jax.nn.softplus(o[:, 1])
        beta = 1.0 + jax.nn.softplus(o[:, 2])
        return g_allin, alpha, beta


class CoordFracGauss(nn.Module):
    """exp022 v2 (2026-06-14): edge-aware CLIPPED-GAUSSIAN fraction head, REDESIGNED.
    Changes vs v1:
      (a) edge_tid is the MULTI-OPERATING-POINT edge [all-in 6 ‖ half-garrison 5 = 11] so the head sees
          the speed-dependent demand SLOPE, not just the all-in point (a partial fleet is slower ->
          arrives later -> faces more defense; one all-in snapshot cannot express that).
      (b) the 11-d edge is ENCODED by a small shared MLP (edge_enc, 11->32) BEFORE use, not raw-
          concatenated (a raw relational vector drowns in the ~770d processed block; encode-first is the
          ECC/MPNN/Graphormer-readout standard).
      (c) the coordination self-attention is masked to SAME-TARGET launching sources
          (intend[q] & tid[q]==tid[s]); co-attackers of ONE target deconflict their joint load. v1's mask
          attended ALL launching sources (diluted).
      (d) LEAN readout [edge_enc ‖ h_coord]: ctx/emb_tid already enter the token z (-> h_coord via the
          residual), so re-feeding the 640-d ctx (incl the per-source-identical board) at the readout was
          redundant heft; dropped.
    f = clip(N(mu,sigma),0,1): f=0 HOLD, f=1 ALL-IN (clip CDF atoms). Explicit layer names -> deterministic
    msgpack keys for the numpy port (no flax lexical-autoname guessing)."""
    E: int = 64
    n_heads: int = 4

    @nn.compact
    def __call__(self, ctx, emb_tid, edge_tid, tid, intend):   # ctx(P,5E) emb_tid(P,E) edge_tid(P,11) tid(P,)int intend(P,)bool
        P = ctx.shape[0]; H = self.n_heads; d = self.E // H
        edge_enc = nn.gelu(nn.Dense(32, name='edge_enc')(edge_tid))                              # (P,32) shared edge encoder
        z = nn.Dense(self.E, name='ztok')(jnp.concatenate([ctx, emb_tid, edge_enc], axis=-1))    # assignment token
        zn = nn.LayerNorm(name='ln')(z)
        qkv = nn.Dense(3 * self.E, name='qkv')(zn)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        q = q.reshape(P, H, d); k = k.reshape(P, H, d); v = v.reshape(P, H, d)
        scores = jnp.einsum('phd,qhd->hpq', q, k) / jnp.sqrt(float(d))     # (H,P,P): scores[h, s(query), q(key)]
        same_t = (tid[:, None] == tid[None, :])                            # (P,P) [s,q] q chose the SAME target as s
        keep = intend[None, :] & same_t                                    # attend launching SAME-target sources
        kmask = jnp.broadcast_to(keep[None, :, :], (H, P, P))
        scores = jnp.where(kmask, scores, -1e9)
        attn = jax.nn.softmax(scores, axis=-1)                            # no co-attacker row -> uniform (no NaN); launching s always keeps self
        out = jnp.einsum('hpq,qhd->phd', attn, v).reshape(P, self.E)
        h_coord = z + nn.Dense(self.E, name='hproj')(out)                 # (P,E)
        x = jnp.concatenate([edge_enc, h_coord], axis=-1)                 # (P, 32+E) LEAN readout
        o = nn.Dense(2, name='mlp_out')(nn.gelu(nn.Dense(2 * self.E, name='mlp_in')(x)))   # (P,2): mu, raw log_sigma
        mu = o[:, 0]
        sigma = jnp.exp(jnp.clip(o[:, 1], -2.0, 0.5))                     # sigma in [~0.135, ~1.65]
        return mu, sigma


class SimpleFracMLP(nn.Module):
    """v41: frac head = 2-layer MLP on [emb[s] || emb_tid || gemb] = 3E (SOURCE + TARGET + GLOBAL).
    emb[s] = the launching source's own trunk embedding (so two sources hitting the same target can size
    differently); emb_tid = chosen target's embedding; gemb = post-trunk global token. NO edge, NO
    coordination. f = clip(N(mu,sigma),0,1). Explicit names mlp_in/mlp_out for the numpy port."""
    E: int = 64
    n_heads: int = 4          # unused (ctor-signature parity)
    @nn.compact
    def __call__(self, emb, emb_tid, gemb):            # emb (P,E) SOURCE, emb_tid (P,E) TARGET, gemb (E,) GLOBAL
        P = emb_tid.shape[0]
        g = jnp.broadcast_to(gemb[None, :], (P, gemb.shape[-1]))     # (P,E)
        x = jnp.concatenate([emb, emb_tid, g], axis=-1)             # (P,3E=384)
        o = nn.Dense(2, name='mlp_out')(nn.gelu(nn.Dense(2 * self.E, name='mlp_in')(x)))   # 384 -> 256 -> 2
        mu = o[:, 0]
        sigma = jnp.exp(jnp.clip(o[:, 1], -2.0, 0.0))   # exp25: upper-clip 0.5->0.0 -> sigma in [~0.135, 1.0] (was 1.65)
        return mu, sigma


class OrbitNet19(nn.Module):
    E: int = 64
    n_heads: int = 4
    n_layers: int = 4
    pool_heads: int = 2          # attention-pooling heads (time axis)

    @nn.compact
    def __call__(self, static, ts, glob, reach, mask, edge, econ):   # exp21: + edge (P,P,E_edge); v32: + econ (T,2) curves
        P = static.shape[0]
        H, d = self.n_heads, self.E // self.n_heads
        Pp = P + 1

        # ---- time-series encoder (v10): conv 32/64/128 (k5) -> ATTENTION POOL over time
        # (query from the planet's static state, 17v12-style: "what the planet IS decides which
        # timesteps matter") -> Dense+gelu+LN. Replaces exp21's no-pool flatten, which forced a
        # 4-channel bottleneck (flatten = ch*50); attn-pool affords 128 channels cheaply. ----
        h = nn.gelu(nn.Conv(16, kernel_size=(5,), padding='SAME')(ts))               # v32: 32->16 (16-32-64)
        h = nn.gelu(nn.Conv(32, kernel_size=(5,), padding='SAME')(h))
        h = nn.gelu(nn.Conv(64, kernel_size=(5,), padding='SAME')(h))               # (P,T,C=64)
        T, C = h.shape[1], h.shape[2]
        p_mean = h.mean(axis=1)                                                     # (P,C) level
        p_max = jnp.max(h, axis=1)                                                  # (P,C) event
        ph, pd = self.pool_heads, C // self.pool_heads                              # attn pool out = C
        pq = nn.Dense(ph * pd)(static).reshape(P, ph, pd)                           # query from static
        pk = nn.Dense(ph * pd)(h).reshape(P, T, ph, pd)
        pv = nn.Dense(ph * pd)(h).reshape(P, T, ph, pd)
        psc = jnp.einsum('phd,pthd->pht', pq, pk) / jnp.sqrt(float(pd))             # (P,ph,T)
        patt = jax.nn.softmax(psc, axis=-1)
        p_attn = jnp.einsum('pht,pthd->phd', patt, pv).reshape(P, ph * pd)          # (P,C) attended
        pooled = jnp.concatenate([p_mean, p_max, p_attn], axis=-1)                  # (P,3C) mean+max+attn
        ts_emb = nn.LayerNorm()(nn.gelu(nn.Dense(self.E // 2)(pooled)))             # (P,E//2)

        # ---- static encoder (v10: -> E//2 so static & ts each get half the fused token) ----
        static_emb = nn.LayerNorm()(nn.Dense(self.E // 2)(nn.gelu(nn.Dense(self.E // 2)(static))))  # (P,E//2)

        # ---- concat (exp21: broadcast-MLP global route DROPPED; global enters via FiLM + gtok only) ----
        concat = jnp.concatenate([static_emb, ts_emb], axis=-1)                     # (P,192)
        C = concat.shape[-1]

        # ---- global (b): FiLM (zero-init -> identity at start) ----
        film = nn.Dense(2 * C, kernel_init=_ZERO, bias_init=_ZERO)(glob)
        g_delta, beta = jnp.split(film, 2)
        feat = (1.0 + g_delta)[None, :] * concat + beta[None, :]

        # ---- v32 ECON-CNN: full me-vs-opp lead curves econ (T,2) -> im2col conv 16/32/64 (k5)
        # -> mean+max+attn 3-pool (attn query from glob, the global analog of the ts static-query)
        # -> econ_emb (E//2), fed into gtok. REPLACES v26-v30's 14 econ summary scalars. ----
        eh = nn.gelu(_conv1d_mm(econ, 5, 16, name='econ_c0'))                       # (T,16)
        eh = nn.gelu(_conv1d_mm(eh, 5, 32, name='econ_c1'))                         # (T,32)
        eh = nn.gelu(_conv1d_mm(eh, 5, 64, name='econ_c2'))                         # (T,64)
        eT, eC = eh.shape[0], eh.shape[1]
        e_mean = eh.mean(axis=0); e_max = jnp.max(eh, axis=0)                        # (eC,) each
        eph, epd = self.pool_heads, eC // self.pool_heads                            # attn pool out = eC
        eq = nn.Dense(eph * epd)(glob).reshape(eph, epd)                             # query from glob
        ek = nn.Dense(eph * epd)(eh).reshape(eT, eph, epd)
        ev = nn.Dense(eph * epd)(eh).reshape(eT, eph, epd)
        esc = jnp.einsum('hd,thd->ht', eq, ek) / jnp.sqrt(float(epd))                # (eph,T)
        e_attn = jnp.einsum('ht,thd->hd', jax.nn.softmax(esc, axis=-1), ev).reshape(eph * epd)  # (eC,)
        econ_pooled = jnp.concatenate([e_mean, e_max, e_attn], axis=-1)              # (3eC,) mean+max+attn
        econ_emb = nn.LayerNorm()(nn.gelu(nn.Dense(self.E // 2)(econ_pooled)))       # (E//2,)

        # ---- token MLP + (c) global token (v32: gtok reads [glob || econ_emb]) ----
        tok = nn.Dense(self.E)(nn.gelu(nn.Dense(2 * self.E)(feat))) * mask[:, None]  # v41: token MLP inner 2E (128->256->128)
        gtok = nn.Dense(self.E)(jnp.concatenate([glob, econ_emb]))[None, :]          # v32: + econ_emb
        x = jnp.concatenate([tok, gtok], axis=0)                                    # (P+1,128)
        amask = jnp.concatenate([mask, jnp.ones((1,), bool)])

        # ---- edge bias (exp21, Graphormer-style): per-pair structural bias added INSIDE every
        # trunk attention layer. edge (P,P,E_edge) -> per-head scalar (P,P,H); padded with 0 on the
        # global-token row/col -> the bias flows into BOTH planet emb AND gtok -> board -> the VALUE
        # head sees the relational (arrival/eff/margin) info, not just the pointer. ONE shared
        # EdgeMLP, added each layer (cheap, standard structural-bias sharing). ----
        eb = nn.Dense(H)(nn.gelu(nn.Dense(32)(edge)))                               # (P,P,H); hidden 32 (6->32->H)
        eb = jnp.transpose(eb, (2, 0, 1))                                            # (H,P,P)
        edge_bias = jnp.zeros((H, Pp, Pp), eb.dtype).at[:, :P, :P].set(eb)           # gtok row/col = 0

        # ---- trunk: 3-layer pre-LN MHSA (+ edge bias) ----
        for _ in range(self.n_layers):
            xn = nn.LayerNorm()(x)
            qkv = nn.Dense(3 * self.E)(xn)
            qh, kh, vh = jnp.split(qkv, 3, axis=-1)
            qh = qh.reshape(Pp, H, d); kh = kh.reshape(Pp, H, d); vh = vh.reshape(Pp, H, d)
            sc2 = jnp.einsum('phd,qhd->hpq', qh, kh) / jnp.sqrt(float(d)) + edge_bias
            sc2 = jnp.where(jnp.broadcast_to(amask[None, None, :], (H, Pp, Pp)), sc2, -1e9)
            out = jnp.einsum('hpq,qhd->phd', jax.nn.softmax(sc2, axis=-1), vh).reshape(Pp, self.E)
            x = x + nn.Dense(self.E)(out)
            xn = nn.LayerNorm()(x)
            x = x + nn.Dense(self.E)(nn.gelu(nn.Dense(2 * self.E)(xn)))
        x = nn.LayerNorm()(x)
        emb = x[:P]
        gemb = x[P]

        # ---- board / ctx: ATTENTION-ONLY pooling over planet tokens (exp22 v3 2026-06-14: dropped the
        # mean+max pools per user -> keep only the learned attention pool; query from the global token
        # output, "the board summary decides which planets to look at").
        # board = [gtok ‖ attn] = 2E; ctx = [emb ‖ board] = 3E. ----
        bh, bd = self.pool_heads, self.E // self.pool_heads
        bq = nn.Dense(self.E)(gemb).reshape(bh, bd)                                  # (bh,bd) q from gtok
        bk = nn.Dense(self.E)(emb).reshape(P, bh, bd)
        bv = nn.Dense(self.E)(emb).reshape(P, bh, bd)
        bsc = jnp.einsum('hd,phd->hp', bq, bk) / jnp.sqrt(float(bd))                 # (bh,P)
        bsc = jnp.where(mask[None, :], bsc, -1e9)
        batt = jax.nn.softmax(bsc, axis=-1)
        board_att = jnp.einsum('hp,phd->hd', batt, bv).reshape(self.E)               # (E,)
        board = jnp.concatenate([gemb, board_att], axis=-1)                          # (2E,)

        # ---- pointer head: SELF = HOLD (always legal) ----
        # tgt = bilinear q·k  +  DIRECT edge term (exp21 option B). The trunk edge-bias above only
        # re-weights attention (out[s]=Σ attn[s,j]·V[j]; V[j] carries node value, NOT the (s,j) margin),
        # so the pointer can't reconstruct margin(s,j) from emb alone. A small per-pair EdgeMLP_p adds
        # the arrival/eff/margin signal straight onto the target logit for the action that uses it. The
        # margin's sign-flip + wide range map naturally to a large +/- additive bias; edge is pre-norm ~[-1,1]. ----
        q2 = nn.Dense(self.E)(emb)
        k2 = nn.Dense(self.E)(emb)
        tgt = (q2 @ k2.T) / jnp.sqrt(float(self.E))                                  # (P,P)
        ptr_edge = nn.Dense(1)(nn.gelu(nn.Dense(16)(edge)))[:, :, 0]                  # (P,P) EdgeMLP_p
        tgt = tgt + ptr_edge
        eye = jnp.eye(P, dtype=bool)
        legal = (reach & mask[None, :] & ~eye) | eye
        tgt = jnp.where(legal, tgt, -1e9)

        # ---- value ----
        v_out = nn.Dense(1)(nn.gelu(nn.Dense(self.E)(board)))[0]
        return tgt, emb, gemb, board, v_out


class LaunchGate(nn.Module):
    """v6 EXPLICIT LAUNCH GATE (2026-06-12): per-planet Bernoulli launch logit from ctx —
    'act or hold' becomes a first-class binary decision instead of competing with 47 targets
    inside the pointer softmax (v12 BaseHead pattern, minus the value output). The pointer's
    self-slot stays legal as a numerical no-op fallback, but HOLD probability mass lives here."""
    E: int = 64

    @nn.compact
    def __call__(self, ctx):
        return nn.Dense(1)(nn.gelu(nn.Dense(self.E)(ctx)))[:, 0]     # (P,) launch logits
