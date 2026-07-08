"""Final imitation-learning models vs the public agents.

Nature/Science style (bundled Inter, booktabs rules), 10.25 in wide so the font
matches the feature/profile tables. Two blocks; the "2p"/"4p" tag sits in the
header row's first cell (not a separate title line, not a row label). 2p has no
"ours" column; the 4p block carries our own P(1st) in the ours column. No agent
legend / no reference links in the figure (text + footnotes carry those).

Numbers:
- 2p: best benchmarked 2p IL (exp-020 v9, same >=1500 data/recipe as warm-start v41);
  head-to-head win-rate, n=20.
- 4p: exp-024 IL u30000, one 64-game 4-way FFA. P(1st) sums to 1.0 across the four
  seats (ours 0.41); out-rank = P(our model ranks strictly above that opponent).

    python figures/make_il_vs_public.py
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

CHAR = "#2b2b2b"; GRAY = "#7d7d7d"; RULE = "#333333"
RULE_LW = 1.0; RULE_LW_EDGE = 2.0
BODY = 12.5; HEAD = 12.5
TARGET_W = 10.25

MARGIN = 0.24; GUT = 0.44; ROWH = 0.30
TOP_MARGIN = 0.18; TOPRULE_PAD = 0.09; HEADH = 0.36
HEADRULE_PAD = 0.05; BLOCK_GAP = 0.24
PRE_BOTTOM = 0.12; BOT_MARGIN = 0.16

SLOTS = ["Orbit-Lite", "Roman", "Shummingfang", "ours"]   # column order (ours only used by 4p)
BLOCKS = [
    {"tag": "2p", "cols": 3, "rows": [("win-rate", ["0.90", "0.90", "0.85"])]},
    {"tag": "4p", "cols": 4, "rows": [("P(1st)",   ["0.33", "0.00", "0.27", "0.41"]),
                                      ("out-rank", ["0.66", "0.91", "0.69"])]},
]


def _fp(base, size):
    p = base.copy(); p.set_size(size); return p


def tw(s, fp=FP_BODY, size=BODY):
    return 0.0 if not s else TextPath((0, 0), s, size=size, prop=fp).get_extents().width / 72.0


def draw(out):
    labels = [lab for b in BLOCKS for lab, _ in b["rows"]] + [b["tag"] for b in BLOCKS]
    w_lab = max(tw(s, FP_BOLD) for s in labels)
    total_w = TARGET_W
    x_lab = MARGIN
    x0 = x_lab + w_lab + GUT
    slot = ((total_w - MARGIN) - x0) / len(SLOTS)
    xc = [x0 + (i + 0.5) * slot for i in range(len(SLOTS))]

    H = TOP_MARGIN + TOPRULE_PAD + PRE_BOTTOM + BOT_MARGIN
    for bi, b in enumerate(BLOCKS):
        H += (BLOCK_GAP if bi > 0 else 0) + HEADH + HEADRULE_PAD + len(b["rows"]) * ROWH

    fig = plt.figure(figsize=(total_w, H), dpi=200)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off()
    ax.set_xlim(0, total_w); ax.set_ylim(0, H)

    def rule(y, lw=RULE_LW):
        ax.plot([MARGIN, total_w - MARGIN], [y, y], color=RULE, lw=lw, solid_capstyle="butt", zorder=3)

    def txt(x, y, s, ha="center", size=BODY, fp=FP_BODY, color=CHAR):
        ax.text(x, y, s, ha=ha, va="center", color=color, fontproperties=_fp(fp, size), zorder=4)

    y = H - TOP_MARGIN
    rule(y, RULE_LW_EDGE)
    y -= TOPRULE_PAD
    for bi, b in enumerate(BLOCKS):
        if bi > 0:
            y -= BLOCK_GAP
        txt(x_lab, y - HEADH / 2, b["tag"], ha="left", size=HEAD, fp=FP_BOLD, color=GRAY)   # first-column header (bold)
        for i in range(b["cols"]):
            txt(xc[i], y - HEADH / 2, SLOTS[i], ha="center", size=HEAD, color=GRAY,
                fp=(FP_BOLD if SLOTS[i] == "ours" else FP_BODY))
        y -= HEADH + HEADRULE_PAD
        rule(y)
        for lab, vals in b["rows"]:
            txt(x_lab, y - ROWH / 2, lab, ha="left", color=GRAY)
            for i, v in enumerate(vals):
                txt(xc[i], y - ROWH / 2, v, ha="center",
                    fp=(FP_BOLD if SLOTS[i] == "ours" else FP_BODY))
            y -= ROWH
    y -= PRE_BOTTOM
    rule(y, RULE_LW_EDGE)

    fig.savefig(out, facecolor="white")
    plt.close(fig)
    print("wrote", out, f"({total_w:.2f} x {H:.2f} in)")


if __name__ == "__main__":
    draw(os.path.join(HERE, "il_vs_public.png"))
