"""2p and 4p cross-evaluation standings tables.

Same Nature/Science style as the other figures (bundled Inter, booktabs rules,
10.25 in wide). Numbers are taken verbatim from the cross-eval logs:
  2p: 2p_cross_leaderboard.log  -> per-combo avg win-rate + vs-#1 / vs-#2.
      Ranks 1-15 shown; the 21 older weights (expB-expH) are collapsed to one row.
  4p: 4p_cross_leaderboard.log  -> the 10 held-out weights, P(1st..4th) + eturn,
      with the deployed top-2 MERGE (from the same board pool) as the top row.
Each row also carries its combos (pairings) and games. The deployed run is
labelled final@u<update>; the other experiments are relabelled expA@..expH@.

    python figures/make_cross_eval.py
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.textpath import TextPath

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # figures/ (script lives in figures/src/)
FDIR = os.path.join(HERE, "fonts")
FP_BODY = fm.FontProperties(fname=os.path.join(FDIR, "Inter-Light.ttf"))
FP_BOLD = fm.FontProperties(fname=os.path.join(FDIR, "Inter-SemiBold.ttf"))
FP_ITAL = fm.FontProperties(fname=os.path.join(FDIR, "Inter-LightItalic.ttf"))

CHAR = "#2b2b2b"; GRAY = "#7d7d7d"; RULE = "#333333"
RULE_LW = 1.0; RULE_LW_EDGE = 2.0
BODY = 12.0; HEAD = 12.0; FOOT = 10.0
TARGET_W = 10.25

MARGIN = 0.24; GUT = 0.30; GAP_RANK = 0.28; ROWH = 0.30
TOP_MARGIN = 0.16; TOPRULE_PAD = 0.09; HEADH = 0.34
PRE_BOTTOM = 0.12; FOOT_GAP = 0.05; FOOT_LH = 0.24; BOT_MARGIN = 0.16

# head = [rank, label, num1, num2, ...]; rows = (rank, label, [num cells...], bold?)
CROSS_2P = {
    "head": ["#", "2p checkpoint", "combos", "games", "win-rate", "vs #1", "vs #2"],
    "rows": [
        ("1",  "final@u55000", ["35", "13,440", "0.669", "—", "49%"], True),
        ("2",  "final@u53000", ["35", "13,440", "0.667", "50%", "—"], False),
        ("3",  "final@u54000", ["35", "13,440", "0.665", "51%", "50%"], False),
        ("4",  "final@u51000", ["35", "13,440", "0.657", "45%", "52%"], False),
        ("5",  "final@u52000", ["35", "13,440", "0.655", "47%", "46%"], False),
        ("6",  "final@u49000", ["35", "13,440", "0.649", "51%", "45%"], False),
        ("7",  "final@u48000", ["35", "13,440", "0.646", "46%", "46%"], False),
        ("8",  "final@u50000", ["35", "13,440", "0.636", "47%", "40%"], False),
        ("9",  "expA@u58000",  ["35", "13,440", "0.634", "46%", "47%"], False),
        ("10", "expA@u56000",  ["35", "13,440", "0.633", "36%", "42%"], False),
        ("11", "expA@u57000",  ["35", "13,440", "0.633", "41%", "48%"], False),
        ("12", "expA@u53000",  ["35", "13,440", "0.629", "49%", "45%"], False),
        ("13", "expA@u54000",  ["35", "13,440", "0.628", "44%", "40%"], False),
        ("14", "final@u46000", ["35", "13,440", "0.627", "40%", "46%"], False),
        ("15", "final@u45000", ["35", "13,440", "0.627", "41%", "43%"], False),
        ("16–36", "expB – expH  (21 weights)", ["35", "13,440", "0.48 – 0.24", "...", "..."], False),
    ],
    "foot": ["Per-pair average win-rate over the 36-weight pool (630 pairs, 483,840 agent-games).",
             "vs #1 / vs #2 = win-rate vs final@u55000 / u53000. final@u55000 (bold) deployed; expA–expH = the eight prior experiments (expB–expH collapsed)."],
}

CROSS_4P = {
    "head": ["#", "4p checkpoint", "combos", "games", "P(1st)", "P(2nd)", "P(3rd)", "P(4th)", "eturn"],
    "rows": [
        ("—",  "merge (u44000 + u39000)", ["120", "30,720", "0.460", "0.540", "0.000", "0.000", "—"], True),
        ("1",  "final@u44000", ["84", "21,504", "0.353", "0.239", "0.239", "0.168", "370"], True),
        ("2",  "final@u47000", ["84", "21,504", "0.340", "0.250", "0.215", "0.195", "365"], False),
        ("3",  "final@u46000", ["84", "21,504", "0.326", "0.231", "0.250", "0.193", "369"], False),
        ("4",  "final@u43000", ["84", "21,504", "0.320", "0.252", "0.246", "0.183", "374"], False),
        ("5",  "final@u39000", ["84", "21,504", "0.317", "0.248", "0.240", "0.195", "367"], True),
        ("6",  "final@u41000", ["84", "21,504", "0.313", "0.274", "0.216", "0.197", "368"], False),
        ("7",  "final@u38000", ["84", "21,504", "0.307", "0.273", "0.234", "0.187", "381"], False),
        ("8",  "expA@u67000",  ["84", "21,504", "0.081", "0.254", "0.291", "0.374", "353"], False),
        ("9",  "expA@u70000",  ["84", "21,504", "0.073", "0.255", "0.269", "0.403", "357"], False),
        ("10", "expA@u69000",  ["84", "21,504", "0.070", "0.224", "0.301", "0.405", "357"], False),
    ],
    "foot": ["P(1st–4th) and mean game length (eturn) on held-out boards (210 combos, 215,040 agent-games).",
             "the merge row is a 1-ply value ensemble of final@u44000 and final@u39000 (bold), which never placed below 2nd; expA is the prior 4p agent."],
}

TABLES = {"cross_2p": CROSS_2P, "cross_4p": CROSS_4P}


def _fp(base, size):
    p = base.copy(); p.set_size(size); return p


def tw(s, fp=FP_BODY, size=BODY):
    return 0.0 if not s else TextPath((0, 0), s, size=size, prop=fp).get_extents().width / 72.0


def draw(spec, out):
    head, rows, foot = spec["head"], spec["rows"], spec["foot"]
    ncol = len(head) - 2                                   # numeric columns
    total_w = TARGET_W

    w_rank = max(tw(head[0], FP_BODY, HEAD), max(tw(r[0], FP_BOLD if r[3] else FP_BODY) for r in rows))
    x_rank_r = MARGIN + w_rank
    x_lab = x_rank_r + GAP_RANK

    wcol = []
    for i in range(ncol):
        w = tw(head[i + 2], FP_BODY, HEAD)
        for _, _, vals, bold in rows:
            w = max(w, tw(vals[i], FP_BOLD if bold else FP_BODY))
        wcol.append(w)
    r_edge = [0.0] * ncol
    r_edge[ncol - 1] = total_w - MARGIN
    for i in range(ncol - 2, -1, -1):
        r_edge[i] = r_edge[i + 1] - wcol[i + 1] - GUT

    n = len(rows)
    H = (TOP_MARGIN + TOPRULE_PAD + HEADH + n * ROWH
         + PRE_BOTTOM + FOOT_GAP + len(foot) * FOOT_LH + BOT_MARGIN)

    fig = plt.figure(figsize=(total_w, H), dpi=200)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off()
    ax.set_xlim(0, total_w); ax.set_ylim(0, H)

    def rule(y, lw=RULE_LW):
        ax.plot([MARGIN, total_w - MARGIN], [y, y], color=RULE, lw=lw, solid_capstyle="butt", zorder=3)

    def txt(x, y, s, ha="left", size=BODY, fp=FP_BODY, color=CHAR):
        ax.text(x, y, s, ha=ha, va="center", color=color, fontproperties=_fp(fp, size), zorder=4)

    y = H - TOP_MARGIN
    rule(y, RULE_LW_EDGE)
    y -= TOPRULE_PAD
    txt(x_rank_r, y - HEADH / 2, head[0], ha="right", size=HEAD, color=GRAY)
    txt(x_lab, y - HEADH / 2, head[1], ha="left", size=HEAD, color=GRAY)
    for i in range(ncol):
        txt(r_edge[i], y - HEADH / 2, head[i + 2], ha="right", size=HEAD, color=GRAY)
    y -= HEADH
    rule(y)
    for rank, name, vals, bold in rows:
        fp = FP_BOLD if bold else FP_BODY
        txt(x_rank_r, y - ROWH / 2, rank, ha="right", fp=FP_BODY, color=GRAY)
        txt(x_lab, y - ROWH / 2, name, ha="left", fp=fp)
        for i in range(ncol):
            txt(r_edge[i], y - ROWH / 2, vals[i], ha="right", fp=fp)
        y -= ROWH
    y -= PRE_BOTTOM
    rule(y, RULE_LW_EDGE)
    y -= FOOT_GAP
    for ln in foot:
        ax.text(MARGIN, y - FOOT_LH / 2, ln, ha="left", va="center", color=GRAY,
                fontproperties=_fp(FP_ITAL, FOOT), zorder=4)
        y -= FOOT_LH

    fig.savefig(out, facecolor="white")
    plt.close(fig)
    print("wrote", out, f"({total_w:.2f} x {H:.2f} in)")


if __name__ == "__main__":
    for key, spec in TABLES.items():
        draw(spec, os.path.join(HERE, key + ".png"))
