"""Render a Kaggle orbit-wars episode replay as an animated GIF, in the style of the
project's local replay viewer (viewer/viewer_core.js).

This bundle documents the ONE evaluation-phase game where our submission "Luca" (the
merge, sub 53993338) beat "Isaiah @ Tufa Labs" in a 2-player head-to-head:
    episode 84096506  ·  seed 432888502  ·  P1 Luca wins by full-board conquest at turn 143.

Files in this folder:
    episode_84096506_raw.json   the ORIGINAL replay, exactly as downloaded from Kaggle
                                (GET /api/v1/competitions/episodes/84096506/replay).
    winprob_84096506.json       per-turn RL win-confidence for seat 1 (Luca), = (V(s)+1)/2
                                from the 2p greedy value head. Regenerated below if absent.
    luca_vs_tufa_ep84096506.gif the rendered animation (this script's output).

The win-probability curve is computed with the deploy value head verbatim (2p/inference,
pure numpy, shared by the merge). It was verified to reproduce the stored _aux.winprob of a
local replay to within max|Δ|=0.00000, so this offline replay -> winprob is exact.

Layout (1040x880): board 720x720 + right info panel 320; below, two half-height charts
side by side (total ships | win probability). Playback = 4x (83 ms/turn, one frame/turn).

Run:  python make_replay_gif.py        (uses conda env kaggle-orbitwar: numpy + Pillow)
"""
import json
import math
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]                      # figures/luca_vs_tufa -> repo root
INFER_DIR = REPO / "2p" / "inference"       # deploy value head (numpy)
RAW = HERE / "episode_84096506_raw.json"
WP = HERE / "winprob_84096506.json"
OUT = HERE / "luca_vs_tufa_ep84096506.gif"

# ============================================================================
# 1) Win-probability (RL value head) — compute from the replay if not cached.
# ============================================================================
def compute_winprob(replay, me):
    """Per-turn (V(s)+1)/2 for `me`, replaying each turn's stored observation through the
    deploy primitives (state.update -> _obs_to_arr -> value_of). Kaggle sets `step` only in
    seat-0's obs, so inject it into seat>0's obs (mirrors eval.py)."""
    sys.path.insert(0, str(INFER_DIR))
    import rl_agent_greedy as G             # _obs_to_arr, AgentState, _W, R.value_of
    steps = replay["steps"]
    n = len(steps)
    state = G.AgentState()
    out = [None] * n
    for t in range(n):
        seat0 = steps[t][0]["observation"]
        obs = steps[t][me]["observation"]
        if obs.get("step") is None and seat0.get("step") is not None:
            obs = dict(obs); obs["step"] = seat0["step"]
        state.update(obs)
        arr, _ = G._obs_to_arr(obs, state)
        if arr["p_x"].shape[0] == 0:
            continue
        v = G.R.value_of(arr, G._W, me)
        out[t] = round(float((v + 1.0) / 2.0), 4)
    return out


def load_or_compute_winprob(replay, me=1):
    if WP.exists():
        return json.loads(WP.read_text())[str(me)]
    series = compute_winprob(replay, me)
    WP.write_text(json.dumps({str(me): series}))
    return series


# ============================================================================
# 2) Renderer (constants + drawing ported from viewer/viewer_core.js).
# ============================================================================
BOARD = 100.0
SUN_X, SUN_Y, SUN_R = 50.0, 50.0, 10.0
PAD = 4.0
COLORS = {0: (59, 130, 246), 1: (239, 68, 68), 2: (16, 185, 129), 3: (245, 158, 11), -1: (148, 163, 184)}
COMET = (167, 139, 250)
AMBER = (251, 191, 36)
BG = (5, 10, 21)
PANEL_BG = (11, 18, 32)
GRID = (30, 41, 59)
GRID_MID = (71, 85, 105)
MUTED = (100, 116, 139)
TXT = (226, 232, 240)
SUBTXT = (148, 163, 184)
MARKER = (34, 197, 94)

SS = 2                     # supersample for anti-aliasing (downscaled at the end)
BW = 720                   # board (viewer canvas is 720x720)
SIDE_W = 320
GW = BW + SIDE_W           # 1040
CHART_REGION_H = 160       # two half-height charts, side by side
GH = BW + CHART_REGION_H   # 880
CHART_W = GW // 2          # 520 each
scale = BW / (BOARD + PAD * 2)
FRAME_MS = 83              # 4x  ==  max(20, (1000/3)*0.25)
# GitHub raw serves files over ~10 MB as application/octet-stream, which browsers refuse to
# render as an image (so it won't show when hotlinked on Kaggle). Keep the GIF well under
# that by downscaling the output a little and using a smaller palette.
GIF_SCALE = 0.62          # output scale relative to GW x GH (keeps the file well under 10 MB)
GIF_COLORS = 128          # GIF palette size (fewer colors => smaller file)


def _font(bold, px):
    """Monospace to match the viewer; Consolas on Windows, graceful fallbacks elsewhere."""
    cands = [r"C:\Windows\Fonts\consolab.ttf"] if bold else [r"C:\Windows\Fonts\consola.ttf"]
    cands += ["DejaVuSansMono-Bold.ttf", "DejaVuSansMono.ttf", "courbd.ttf", "cour.ttf"]
    for c in cands:
        try:
            return ImageFont.truetype(c, int(px * SS))
        except Exception:
            continue
    return ImageFont.load_default()

f_ship = _font(1, 13); f_prod = _font(1, 11); f_pid = _font(1, 11); f_fleet = _font(1, 9)
f_title = _font(1, 16); f_sub = _font(0, 12); f_turn = _font(1, 16)
f_pstat = _font(0, 13); f_pstatb = _font(1, 13); f_fhdr = _font(0, 12); f_fitem = _font(0, 11)
f_axis = _font(0, 12); f_chdr = _font(1, 13)


def w2c(x, y):
    return (x + PAD) * scale * SS, (y + PAD) * scale * SS

def rgba(c, a):
    return (c[0], c[1], c[2], int(a * 255))

def text_center(dr, cx, cy, s, font, fill):
    b = dr.textbbox((0, 0), s, font=font)
    dr.text((cx - (b[2] - b[0]) / 2, cy - (b[3] - b[1]) / 2 - b[1]), s, font=font, fill=fill)

def text_mid_left(dr, x, cy, s, font, fill):
    b = dr.textbbox((0, 0), s, font=font)
    dr.text((x, cy - (b[3] - b[1]) / 2 - b[1]), s, font=font, fill=fill)


def build_sun_sprite():
    import numpy as np
    r = SUN_R * scale * SS
    size = int(math.ceil(r * 2)) + 2
    cx = cy = size / 2.0
    yy, xx = np.mgrid[0:size, 0:size]
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / r
    c0 = np.array([253, 224, 71]); c6 = np.array([251, 191, 36]); c1 = np.array([252, 211, 77])
    t = np.clip(dist, 0, 1); col = np.zeros((size, size, 3)); a = np.zeros((size, size))
    m1 = t <= 0.6; f1 = (t / 0.6)[m1][:, None]; col[m1] = c0 * (1 - f1) + c6 * f1; a[m1] = 1.0
    m2 = ~m1; f2 = ((t - 0.6) / 0.4)[m2][:, None]; col[m2] = c6 * (1 - f2) + c1 * f2
    a[m2] = (1 - ((t - 0.6) / 0.4))[m2]; a[dist > 1.0] = 0.0
    arr = np.dstack([col.astype("uint8"), (np.clip(a, 0, 1) * 255).astype("uint8")])
    return Image.fromarray(arr, "RGBA"), size


def draw_dashed_circle(dr, cx, cy, r, color, dashes=44, width=1):
    for i in range(dashes):
        if i % 2:
            continue
        a0 = 2 * math.pi * i / dashes; a1 = 2 * math.pi * (i + 1) / dashes
        dr.line([(cx + r * math.cos(a0), cy + r * math.sin(a0)),
                 (cx + r * math.cos(a1), cy + r * math.sin(a1))], fill=color, width=width)


def main():
    replay = json.loads(RAW.read_text())
    info = replay["info"]
    names = info.get("TeamNames") or ["P0", "P1"]
    seed = info.get("seed")
    rewards = replay["rewards"]
    steps = []
    for arr in replay["steps"]:
        obs = arr[0]["observation"]
        steps.append({"t": obs.get("step"), "planets": obs.get("planets", []), "fleets": obs.get("fleets", [])})
    N = len(steps)
    init_ids = set(ip[0] for ip in replay["steps"][0][0]["observation"]["initial_planets"])
    nA = 2
    wp_me = load_or_compute_winprob(replay, 1)

    ships_series = [[0] * N for _ in range(nA)]
    for t, st in enumerate(steps):
        for p in st["planets"]:
            o = p[1]
            if 0 <= o < nA:
                ships_series[o][t] += p[5]
        for f in st["fleets"]:
            o = f[1]
            if 0 <= o < nA:
                ships_series[o][t] += f[6]
    maxV = max(1, max(max(s) for s in ships_series))

    launch = {}
    for t, st in enumerate(steps):
        for f in st["fleets"]:
            launch.setdefault(f[0], t)

    sun_sprite, sun_size = build_sun_sprite()

    def is_comet(p):
        return p[0] not in init_ids

    def player_totals(st):
        ships = [0] * nA; prod = [0] * nA; planets = [0] * nA
        for p in st["planets"]:
            o = p[1]
            if 0 <= o < nA:
                ships[o] += p[5]; prod[o] += p[6]; planets[o] += 1
        for f in st["fleets"]:
            o = f[1]
            if 0 <= o < nA:
                ships[o] += f[6]
        return ships, prod, planets

    def render_board(base, dr, st):
        scx, scy = w2c(SUN_X, SUN_Y)
        base.alpha_composite(sun_sprite, (int(scx - sun_size / 2), int(scy - sun_size / 2)))
        sr = SUN_R * scale * SS
        draw_dashed_circle(dr, scx, scy, sr, rgba((252, 165, 165), 0.35), dashes=44, width=max(1, SS))
        for p in st["planets"]:
            pid, owner, x, y, radius, sh, prod = p[0], p[1], p[2], p[3], p[4], p[5], p[6]
            cx, cy = w2c(x, y); r = radius * scale * SS
            comet = is_comet(p)
            col = COMET if comet else COLORS.get(owner, COLORS[-1])
            dr.ellipse([cx - r, cy - r, cx + r, cy + r], fill=col,
                       outline=rgba((255, 255, 255), 0.30), width=max(1, SS // 2))
            text_center(dr, cx, cy, str(sh), f_ship, (255, 255, 255))
            if prod > 0 and not comet:
                text_center(dr, cx, cy + r + 9 * SS, f"+{prod}", f_prod, AMBER)
            text_center(dr, cx, cy - r - 7 * SS, f"p{pid}", f_pid, rgba((255, 255, 255), 0.55))
        for ft in st["fleets"]:
            fid, owner, x, y, angle, src, sh = ft[0], ft[1], ft[2], ft[3], ft[4], ft[5], ft[6]
            cx, cy = w2c(x, y); col = COLORS.get(owner, COLORS[-1])
            size = max(4, min(10, 4 + math.log(max(sh, 1)) * 0.7)) * (scale / 8) * SS
            ca, sa = math.cos(angle), math.sin(angle)
            pts = [(size * 1.5, 0), (-size * 0.7, -size * 0.7), (-size * 0.7, size * 0.7)]
            tp = [(cx + px * ca - py * sa, cy + px * sa + py * ca) for px, py in pts]
            dr.polygon(tp, fill=col, outline=rgba((255, 255, 255), 0.4))
            text_mid_left(dr, cx + size + 2 * SS, cy - size, str(sh), f_fleet, col)

    def render_side(dr, st, cur):
        x0 = BW * SS + 14 * SS
        y = 14 * SS
        dr.text((x0, y), "Orbit Wars · local replay", font=f_title, fill=TXT); y += 24 * SS
        dr.text((x0, y), f"Episode 84096506  ·  seed {seed}", font=f_sub, fill=MUTED); y += 17 * SS
        dr.text((x0, y), "Luca vs Isaiah @ Tufa Labs", font=f_sub, fill=SUBTXT); y += 24 * SS
        dr.text((x0, y), f"Turn {st['t']} / {N - 1}", font=f_turn, fill=TXT); y += 27 * SS
        ships, prod, planets = player_totals(st)
        for pid in range(nA):
            col = COLORS[pid]
            tag = "  (winner)" if rewards[pid] == max(rewards) and rewards.count(max(rewards)) == 1 else ""
            dr.text((x0, y), f"P{pid} {names[pid]}{tag}", font=f_pstatb, fill=col); y += 18 * SS
            dr.text((x0 + 12 * SS, y), f"{ships[pid]} ships · +{prod[pid]}/t · {planets[pid]} planets",
                    font=f_pstat, fill=col); y += 22 * SS
        wv = wp_me[cur]
        if wv is not None:
            dr.text((x0, y), "RL win-confidence (value head):", font=f_sub, fill=SUBTXT); y += 17 * SS
            dr.text((x0 + 12 * SS, y), f"P1 Luca {round(100 * wv)}%   ·   P0 {100 - round(100 * wv)}%",
                    font=f_pstatb, fill=COLORS[1]); y += 22 * SS
        y += 2 * SS
        dr.line([(x0, y), (BW * SS + (SIDE_W - 12) * SS, y)], fill=GRID, width=max(1, SS)); y += 8 * SS
        fl = sorted(st["fleets"], key=lambda a: (a[1], a[0]))
        dr.text((x0, y), f"Fleets in flight ({len(fl)})", font=f_fhdr, fill=SUBTXT); y += 19 * SS
        maxrows = int((BW * SS - y) / (15 * SS))
        for ft in fl[:maxrows]:
            fid, owner, x, yy2, angle, src, sh = ft[0], ft[1], ft[2], ft[3], ft[4], ft[5], ft[6]
            col = COLORS.get(owner, COLORS[-1]); deg = int(round(angle * 180 / math.pi))
            age = st["t"] - launch.get(fid, st["t"])
            dr.text((x0, y), f"P{owner} f{fid} · {sh} sh · p{src}→ {deg}° · {age}t", font=f_fitem, fill=col)
            y += 15 * SS
        if len(fl) > maxrows:
            dr.text((x0, y), f"  +{len(fl) - maxrows} more…", font=f_fitem, fill=MUTED)

    def line_chart(dr, ox, oy, Wc, Hc, header, series, colors, cur, ymax, pct=False, mid=False):
        dr.rectangle([ox, oy, ox + Wc, oy + Hc], fill=BG)
        dr.text((ox + 42 * SS, oy + 6 * SS), header, font=f_chdr, fill=SUBTXT)
        padL, padR, padT, padB = 42 * SS, 12 * SS, 26 * SS, 18 * SS
        plotW = Wc - padL - padR; plotH = Hc - padT - padB
        for i in range(5):
            yy = oy + padT + plotH * i / 4
            gcol = GRID_MID if (mid and i == 2) else GRID
            dr.line([(ox + padL, yy), (ox + Wc - padR, yy)], fill=gcol, width=max(1, SS))
            lab = f"{100 * (1 - i / 4):.0f}%" if pct else f"{ymax * (1 - i / 4):.0f}"
            text_mid_left(dr, ox + 4 * SS, yy, lab, f_axis, MUTED)
        xstep = max(1, N // 8)
        for t in range(0, N, xstep):
            xx = ox + padL + plotW * t / max(1, N - 1)
            dr.text((xx - 6 * SS, oy + Hc - 15 * SS), str(t), font=f_axis, fill=MUTED)
        for arr, col in zip(series, colors):
            run = []
            for t in range(N):
                v = arr[t]
                if v is None:
                    if len(run) > 1:
                        dr.line(run, fill=col, width=max(2, int(1.6 * SS)))
                    run = []
                    continue
                xx = ox + padL + plotW * t / max(1, N - 1)
                yy = oy + padT + plotH * (1 - min(max(v, 0), ymax) / ymax)
                run.append((xx, yy))
            if len(run) > 1:
                dr.line(run, fill=col, width=max(2, int(1.6 * SS)))
        xx = ox + padL + plotW * cur / max(1, N - 1)
        dr.line([(xx, oy + padT), (xx, oy + Hc - padB)], fill=MARKER, width=max(1, int(1.5 * SS)))
        if pct:
            v = series[0][cur]
            txt = f"t={cur}  P1={round(100 * v)}%" if v is not None else f"t={cur}"
        else:
            txt = f"t={cur}  " + "  ".join(f"P{i}={series[i][cur]}" for i in range(len(series)))
        tw = dr.textlength(txt, font=f_axis)
        dr.text((min(xx + 6 * SS, ox + Wc - padR - tw), oy + padT + 3 * SS), txt, font=f_axis, fill=MARKER)

    def render_frame(cur):
        img = Image.new("RGBA", (GW * SS, GH * SS), BG + (255,))
        dr = ImageDraw.Draw(img, "RGBA")
        dr.rectangle([BW * SS, 0, GW * SS, BW * SS], fill=PANEL_BG)
        dr.line([(BW * SS, 0), (BW * SS, BW * SS)], fill=GRID, width=max(1, SS))
        st = steps[cur]
        render_board(img, dr, st)
        render_side(dr, st, cur)
        line_chart(dr, 0, BW * SS, CHART_W * SS, CHART_REGION_H * SS, "Total ships over time",
                   ships_series, [COLORS[0], COLORS[1]], cur, maxV)
        line_chart(dr, CHART_W * SS, BW * SS, CHART_W * SS, CHART_REGION_H * SS,
                   "Win probability  (RL value head, P1 Luca)", [wp_me], [COLORS[1]], cur, 1.0, pct=True, mid=True)
        dr.line([(CHART_W * SS, BW * SS), (CHART_W * SS, GH * SS)], fill=GRID, width=max(1, SS))
        dr.line([(0, BW * SS), (GW * SS, BW * SS)], fill=GRID, width=max(1, SS))
        return img.convert("RGB").resize((GW, GH), Image.LANCZOS)

    frames = []
    for cur in range(N):
        frames.append(render_frame(cur))
        if cur % 20 == 0:
            print(f"rendered {cur}/{N}", flush=True)

    hold = max(1, int(1500 / FRAME_MS))            # pause ~1.5s on the final frame before looping
    durations = [FRAME_MS] * (len(frames) + hold - 1)
    final = frames + [frames[-1]] * (hold - 1)

    def _prep(f):                                  # downscale + quantize to keep the file small
        if GIF_SCALE != 1.0:
            f = f.resize((round(GW * GIF_SCALE), round(GH * GIF_SCALE)), Image.LANCZOS)
        return f.quantize(colors=GIF_COLORS, method=Image.MEDIANCUT, dither=Image.NONE)
    final_p = [_prep(f) for f in final]
    final_p[0].save(OUT, save_all=True, append_images=final_p[1:], duration=durations,
                    loop=0, optimize=True, disposal=2)
    import os
    ow, oh = final_p[0].size
    print("SAVED", OUT, "| size MB", round(os.path.getsize(OUT) / 1e6, 2),
          "| frames", len(final_p), f"| {ow}x{oh} | 4x {FRAME_MS}ms", flush=True)


if __name__ == "__main__":
    main()
