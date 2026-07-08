"""Reproducible model-profile table (parameters + FLOPs per OrbitNet block).

Same Nature/Science style as make_feature_tables.py (bundled Inter, booktabs rules,
no vertical lines/shading) and the SAME rendered width (10.25 in) so the font renders
at the identical visual size. Flat: one row per MLP / CNN / pooling block, each with a
light-gray description (name charcoal + detail gray). Numbers come from
scratchpad/orbitnet_breakdown_fine.py, which partitions the deployed 2p u55000 weights
(verified total 1,172,888 params) and counts forward-pass FLOPs at a representative
P = 32 planets.

    python figures/make_model_profile.py
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.textpath import TextPath
from matplotlib.offsetbox import TextArea, HPacker, AnnotationBbox

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # figures/ (script lives in figures/src/)
FDIR = os.path.join(HERE, "fonts")
FP_BODY = fm.FontProperties(fname=os.path.join(FDIR, "Inter-Light.ttf"))
FP_BOLD = fm.FontProperties(fname=os.path.join(FDIR, "Inter-SemiBold.ttf"))
FP_ITAL = fm.FontProperties(fname=os.path.join(FDIR, "Inter-LightItalic.ttf"))

CHAR = "#2b2b2b"   # charcoal body
DESC = "#7d7d7d"   # gray descriptions / footnote
RULE = "#333333"   # rules
RULE_LW = 1.0
RULE_LW_EDGE = 2.0

BODY = 12.5; HEAD = 12.5; FOOT = 10.5
TARGET_W = 10.25   # match the feature tables exactly

MARGIN = 0.24; GUT = 0.40; ROWH = 0.235
TOP_MARGIN = 0.18; TOPRULE_PAD = 0.09; HEADH = 0.36
PRE_TOTAL = 0.10; PRE_BOTTOM = 0.12; FOOT_GAP = 0.05; FOOT_H = 0.30; BOT_MARGIN = 0.18

# (name, description, params, params%, FLOPs, FLOPs%)
ROWS = [
    ("Planet forecast CNN",      "three convs over the 50×7 forecast, per planet", "13.5 k", "1.1%",  "42.9 M", "29.6%"),
    ("Planet attention pooling", "pools the forecast, per planet",                 "9.9 k",  "0.8%",  "27.1 M", "18.7%"),
    ("Planet ts-embed MLP",      "projects the pooled forecast",                   "12.5 k", "1.1%",  "0.8 M",  "0.5%"),
    ("Planet static MLP",        "encodes the static and rollout features",        "5.8 k",  "0.5%",  "0.4 M",  "0.2%"),
    ("Planet FiLM",              "the global vector scales and shifts the features","7.4 k", "0.6%",  "0.02 M", "0.01%"),
    ("Planet token MLP",         "fuses into the 128-d planet token",              "65.9 k", "5.6%",  "4.2 M",  "2.9%"),
    ("Global economy CNN",       "three convs over the 50×2 economy series",   "13.1 k", "1.1%",  "1.3 M",  "0.9%"),
    ("Global attention pooling", "pools the economy series",                       "10.2 k", "0.9%",  "0.8 M",  "0.6%"),
    ("Global econ-embed MLP",    "projects the pooled economy",                    "12.5 k", "1.1%",  "0.02 M", "0.02%"),
    ("Global gtok MLP",          "fuses into the global token",                    "11.9 k", "1.0%",  "0.02 M", "0.02%"),
    ("Main transformer",         "attention and FFN, ×6 layers, 128-d, 4 heads","795 k",  "67.8%", "55.4 M", "38.3%"),
    ("Edge-bias MLP",            "turns P×P×6 into a per-head attention bias",  "356",    "0.03%", "0.7 M",  "0.5%"),
    ("Pointer head",             "scores every planet as a target, with the edge bias", "33.2 k", "2.8%", "2.6 M", "1.8%"),
    ("Fraction head",            "sizes the launch from source, target and global","99.1 k", "8.5%",  "6.3 M",  "4.4%"),
    ("Value head",               "pools the tokens and reads a scalar value",      "82.6 k", "7.0%",  "2.2 M",  "1.5%"),
]
TOTAL = ("Total", "1.17 M", "100%", "144.9 M", "100%")
HEAD_LABELS = ["COMPONENT", "PARAMS", "%", "FLOPs", "%"]


def _fp(base, size):
    p = base.copy(); p.set_size(size); return p


def tw(s, fp=FP_BODY):
    if not s:
        return 0.0
    return TextPath((0, 0), s, size=BODY, prop=fp).get_extents().width / 72.0


def wnum(col, total_i):
    w = max(tw(HEAD_LABELS[col - 1]), max(tw(r[col + 1]) for r in ROWS))
    return max(w, tw(TOTAL[total_i], fp=FP_BOLD))


def draw(out):
    # numeric columns, flush right, at the fixed target width
    w_p, w_ps, w_f, w_fs = wnum(1, 1), wnum(2, 2), wnum(3, 3), wnum(4, 4)
    total_w = TARGET_W
    fs_r = total_w - MARGIN
    f_r = fs_r - w_fs - GUT
    ps_r = f_r - w_f - GUT
    p_r = ps_r - w_ps - GUT
    x_comp = MARGIN

    n = len(ROWS)
    H = (TOP_MARGIN + TOPRULE_PAD + HEADH + n * ROWH
         + PRE_TOTAL + ROWH
         + PRE_BOTTOM + FOOT_GAP + FOOT_H + BOT_MARGIN)

    fig = plt.figure(figsize=(total_w, H), dpi=200)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off()
    ax.set_xlim(0, total_w); ax.set_ylim(0, H)

    def rule(y, lw=RULE_LW):
        ax.plot([MARGIN, total_w - MARGIN], [y, y], color=RULE, lw=lw,
                solid_capstyle="butt", zorder=3)

    def cell(x, y, s, ha="right", size=BODY, fp=FP_BODY, color=CHAR):
        ax.text(x, y, s, ha=ha, va="center", color=color,
                fontproperties=_fp(fp, size), zorder=4)

    def namedesc(y, name, desc, fp_name=FP_BODY):
        a = TextArea(name, textprops=dict(fontproperties=_fp(fp_name, BODY), color=CHAR))
        parts = [a]
        if desc:
            parts.append(TextArea(desc, textprops=dict(fontproperties=_fp(FP_BODY, BODY), color=DESC)))
        box = HPacker(children=parts, align="baseline", pad=0, sep=9)
        ax.add_artist(AnnotationBbox(box, (x_comp, y), xycoords="data", frameon=False,
                                     box_alignment=(0, 0.5), zorder=4))

    def nums(y, vals, fp):
        for x, v in zip((p_r, ps_r, f_r, fs_r), vals):
            cell(x, y, v, ha="right", fp=fp)

    y = H - TOP_MARGIN
    rule(y, RULE_LW_EDGE)                                   # top rule (thick)
    y -= TOPRULE_PAD
    cell(x_comp, y - HEADH / 2, HEAD_LABELS[0], ha="left", size=HEAD)
    for x, s in zip((p_r, ps_r, f_r, fs_r), HEAD_LABELS[1:]):
        cell(x, y - HEADH / 2, s, ha="right", size=HEAD)
    y -= HEADH
    rule(y)                                                 # header rule

    for name, desc, *vals in ROWS:
        yc = y - ROWH / 2
        namedesc(yc, name, desc)
        nums(yc, vals, FP_BODY)
        y -= ROWH

    y -= PRE_TOTAL
    rule(y)                                                 # rule above total
    yc = y - ROWH / 2
    namedesc(yc, TOTAL[0], "", fp_name=FP_BOLD)
    nums(yc, TOTAL[1:], FP_BOLD)
    y -= ROWH

    y -= PRE_BOTTOM
    rule(y, RULE_LW_EDGE)                                   # bottom rule (thick)
    y -= FOOT_GAP
    foot = "FLOPs: one forward pass at P = 32 planets. Parameters are P-independent."
    ax.text(x_comp, y - FOOT_H / 2, foot, ha="left", va="center", color=DESC,
            fontproperties=_fp(FP_ITAL, FOOT), zorder=4)

    fig.savefig(out, facecolor="white")
    plt.close(fig)
    print("wrote", out, f"({total_w:.2f} x {H:.2f} in)")


if __name__ == "__main__":
    draw(os.path.join(HERE, "model_profile.png"))
