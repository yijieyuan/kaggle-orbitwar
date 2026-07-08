"""Final submissions on the official engine: merge vs greedy.

Same block style as make_il_vs_public.py (bundled Inter, booktabs rules, 10.25 in
wide). The "2p"/"4p" tag sits in the header row's first cell; merge (the deployed
score-carrier) is bold. Numbers from local_replays (eval.py, official kaggle engine):
  2p: merge vs greedy head-to-head, 100 games -> merge 0.71 / greedy 0.29.
  4p: each agent as the LONE seat against three copies of the other, 80 games each
      -> lone merge 0.31 (vs 3 greedy), lone greedy 0.12 (vs 3 merge); 0.25 = fair share.

    python figures/make_final_eval.py
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
BODY = 12.5; HEAD = 12.5; FOOT = 10.0
TARGET_W = 10.25

MARGIN = 0.24; GUT = 0.44; ROWH = 0.30
TOP_MARGIN = 0.18; TOPRULE_PAD = 0.09; HEADH = 0.36
HEADRULE_PAD = 0.05; BLOCK_GAP = 0.24
PRE_BOTTOM = 0.12; FOOT_GAP = 0.06; FOOT_LH = 0.24; BOT_MARGIN = 0.14

SLOTS = ["merge", "greedy"]
BOLD = {"merge"}
BLOCKS = [
    {"tag": "", "cols": 2, "rows": [
        ("2p win-rate",       ["0.71", "0.29"]),
        ("4p P(1st), 1 vs 3", ["0.31", "0.12"]),
    ]},
]
FOOTNOTE = ["2p: merge vs greedy head-to-head, 100 games. 4p: each agent as the lone seat against three copies of the other,",
            "80 games each, so 0.25 is the fair share (a lone merge over-shares, a lone greedy under-shares)."]


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

    H = TOP_MARGIN + TOPRULE_PAD + PRE_BOTTOM + FOOT_GAP + len(FOOTNOTE) * FOOT_LH + BOT_MARGIN
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
        txt(x_lab, y - HEADH / 2, b["tag"], ha="left", size=HEAD, fp=FP_BOLD, color=GRAY)
        for i in range(b["cols"]):
            txt(xc[i], y - HEADH / 2, SLOTS[i], ha="center", size=HEAD, color=GRAY,
                fp=(FP_BOLD if SLOTS[i] in BOLD else FP_BODY))
        y -= HEADH + HEADRULE_PAD
        rule(y)
        for lab, vals in b["rows"]:
            txt(x_lab, y - ROWH / 2, lab, ha="left", color=GRAY)
            for i, v in enumerate(vals):
                txt(xc[i], y - ROWH / 2, v, ha="center",
                    fp=(FP_BOLD if SLOTS[i] in BOLD else FP_BODY))
            y -= ROWH
    y -= PRE_BOTTOM
    rule(y, RULE_LW_EDGE)
    y -= FOOT_GAP
    for ln in FOOTNOTE:
        ax.text(MARGIN, y - FOOT_LH / 2, ln, ha="left", va="center", color=GRAY,
                fontproperties=_fp(FP_ITAL, FOOT), zorder=4)
        y -= FOOT_LH

    fig.savefig(out, facecolor="white")
    plt.close(fig)
    print("wrote", out, f"({total_w:.2f} x {H:.2f} in)")


if __name__ == "__main__":
    draw(os.path.join(HERE, "final_eval.png"))
