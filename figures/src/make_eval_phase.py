"""Evaluation-phase tables. Same Nature/Science style as the other figures
(bundled Inter, booktabs rules, 10.25 in wide).

Opponent set FIXED to the top-20 of the 2026-06-25 frozen tracked ranking
(tools/rank_tracker/raw/tracked.json), so it does not drift with the live board.
Win-rates are a LIVE snapshot frozen 2026-07-05 17:20 ET (eval phase still running;
regenerate for finals via scratchpad/gen_combined.py + gen_field2.py):

  eval_h2h  (draw_combined): head-to-head. One row per opponent SUBMISSION (each
      top-20 team has two, told apart by submission id), with my merge (53993338) and
      greedy (53993524) side by side as two column regions. Per region: 2p n = 2p games,
      2p win = my 2p win-rate, me 4p / opp 4p = my P(1st) vs the opponent's P(1st) in our
      shared 4p games (0.25 = even). Top row aggregates all top-20 opponent submissions.
  eval_field (draw_split): each top-20 team's two submissions, games played and overall
      2p win-rate / 4p P(1st) across the whole field (from submissions.csv).

    python figures/make_eval_phase.py
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

CHAR = "#2b2b2b"; GRAY = "#7d7d7d"; RULE = "#333333"; BAND = "#f4f4f4"
RULE_LW = 1.0; RULE_LW_EDGE = 2.0
BODY = 12.0; HEAD = 12.0; FOOT = 10.0
TARGET_W = 10.25

MARGIN = 0.24; GUT = 0.34; GAP_RANK = 0.26; GAP_SUB = 0.26; ROWH = 0.30
TOP_MARGIN = 0.16; TOPRULE_PAD = 0.09; HEADH = 0.34; REGIONH = 0.30
PRE_BOTTOM = 0.12; FOOT_GAP = 0.05; FOOT_LH = 0.24; BOT_MARGIN = 0.16
GUT_C = 0.26; REGION_GAP = 0.42

# ── Combined head-to-head data ──────────────────────────────────────────────────
# groups = (rank, team, [ (opp_sub_id, [merge 5 vals], [greedy 5 vals]), ... ], bold)
# each 5-val list = [2p n, 2p win, 4p n, my P(1st), opp P(1st)]; the draw slices to COL_IDX.
COL_IDX = [0, 1, 3, 4]                       # 2p n, 2p win, my 4p, opp 4p
COL_H = ["2p n", "2p win", "me 4p", "opp 4p"]
GROUPS = [
    ("—", "all top-20 opponents", [("", ["3,662", "44%", "21,581", "26%", "26%"], ["3,249", "41%", "18,579", "29%", "25%"])], True),
    ("1", "Isaiah @ Tufa Labs", [("53993189", ["11", "0%", "41", "15%", "44%"], ["3", "0%", "39", "10%", "44%"]), ("53993217", ["6", "17%", "58", "17%", "41%"], ["3", "0%", "36", "19%", "42%"])], False),
    ("2", "TonyK", [("53983137", ["26", "31%", "232", "14%", "49%"], ["16", "6%", "152", "14%", "50%"]), ("53983530", ["52", "12%", "451", "17%", "51%"], ["27", "30%", "219", "14%", "50%"])], False),
    ("3", "Hober Malloc", [("53982490", ["49", "35%", "387", "17%", "38%"], ["35", "14%", "257", "21%", "42%"]), ("53982496", ["72", "38%", "505", "21%", "36%"], ["29", "17%", "246", "25%", "36%"])], False),
    ("4", "Jake Will", [("53961520", ["34", "12%", "329", "22%", "28%"], ["27", "0%", "170", "22%", "22%"]), ("53972924", ["40", "12%", "272", "19%", "29%"], ["14", "7%", "179", "16%", "26%"])], False),
    ("5", "Felix M Neumann", [("53958390", ["89", "20%", "657", "25%", "21%"], ["42", "12%", "340", "25%", "23%"]), ("53974193", ["55", "16%", "418", "18%", "27%"], ["38", "16%", "261", "18%", "30%"])], False),
    ("6", "flg", [("53913042", ["103", "27%", "803", "20%", "27%"], ["64", "16%", "396", "19%", "28%"]), ("53975220", ["92", "25%", "556", "22%", "25%"], ["36", "14%", "322", "21%", "25%"])], False),
    ("7", "Audun Ljone Henriksen", [("53993640", ["183", "30%", "1,052", "26%", "31%"], ["119", "23%", "614", "25%", "34%"]), ("53993677", ["93", "39%", "648", "17%", "40%"], ["51", "22%", "345", "17%", "35%"])], False),
    ("8", "Ender", [("53979034", ["148", "16%", "819", "26%", "13%"], ["91", "14%", "531", "30%", "14%"]), ("53983642", ["138", "13%", "781", "26%", "9%"], ["82", "10%", "458", "25%", "11%"])], False),
    ("9", "Xiangyu Liu", [("53945598", ["194", "49%", "964", "26%", "35%"], ["158", "48%", "885", "31%", "30%"]), ("53980171", ["209", "56%", "1,038", "24%", "25%"], ["140", "51%", "803", "27%", "25%"])], False),
    ("10", "Boey", [("53986380", ["199", "33%", "1,002", "25%", "22%"], ["117", "21%", "725", "22%", "23%"]), ("53992481", ["203", "35%", "1,007", "21%", "23%"], ["104", "23%", "584", "28%", "20%"])], False),
    ("11", "moriiiiiiiiim", [("53986435", ["186", "56%", "1,016", "25%", "26%"], ["198", "46%", "1,063", "32%", "24%"]), ("53986703", ["87", "56%", "486", "27%", "21%"], ["110", "56%", "660", "30%", "23%"])], False),
    ("12", "Vadasz & Ascalon", [("53993924", ["135", "50%", "660", "32%", "20%"], ["163", "40%", "823", "33%", "20%"]), ("53993965", ["84", "58%", "585", "32%", "21%"], ["122", "43%", "654", "35%", "20%"])], False),
    ("13", "dragon warrior", [("53977843", ["110", "60%", "601", "27%", "27%"], ["139", "55%", "696", "32%", "27%"]), ("53977879", ["135", "63%", "644", "33%", "27%"], ["170", "50%", "846", "30%", "28%"])], False),
    ("15", "One Man Wrecking Machine", [("53993035", ["191", "68%", "928", "31%", "32%"], ["179", "56%", "910", "29%", "32%"]), ("53993719", ["126", "63%", "743", "27%", "32%"], ["180", "56%", "933", "29%", "34%"])], False),
    ("16", "M & J & M.ver2", [("53992808", ["169", "53%", "1,049", "29%", "21%"], ["155", "46%", "839", "29%", "23%"]), ("53993548", ["157", "56%", "853", "31%", "21%"], ["173", "43%", "876", "33%", "23%"])], False),
    ("17", "Orbit Goblins", [("53925121", ["36", "81%", "204", "36%", "25%"], ["55", "60%", "308", "32%", "23%"]), ("53993589", ["61", "66%", "386", "31%", "23%"], ["89", "51%", "522", "34%", "20%"])], False),
    ("18", "jonathan breitgand", [("53961919", ["16", "69%", "112", "35%", "14%"], ["25", "52%", "135", "37%", "19%"]), ("53964200", ["20", "85%", "92", "39%", "17%"], ["29", "76%", "110", "34%", "17%"])], False),
    ("19", "Azat Akhtyamov", [("53993443", ["68", "63%", "547", "31%", "20%"], ["117", "50%", "788", "35%", "17%"]), ("53993868", ["69", "46%", "461", "35%", "17%"], ["108", "51%", "610", "34%", "17%"])], False),
    ("20", "vkhydras", [("53993428", ["7", "86%", "80", "39%", "28%"], ["16", "81%", "115", "36%", "23%"]), ("53993440", ["9", "78%", "114", "29%", "26%"], ["25", "92%", "129", "26%", "36%"])], False),
]
FOOT_H2H = ["Head-to-head: my merge (53993338) and greedy (53993524) vs each top-20 opponent submission (2026-06-25 frozen top-20).",
            "2p n = 2p games; 2p win = my win-rate; me 4p / opp 4p = my vs the opponent's 4p P(1st) (0.25 = even). Top row = all. Final 2026-07-07."]

# ── Field table (per submission) data ───────────────────────────────────────────
HEAD_FIELD = ["#", "team", "sub", "2p games", "2p win", "4p games", "4p P(1st)"]
FIELD_GROUPS = [
    ("1", "Isaiah @ Tufa Labs", [("53993189", ["2,881", "89%", "3,800", "28%"]), ("53993217", ["3,113", "87%", "4,336", "28%"])], False),
    ("2", "TonyK", [("53983530", ["4,556", "41%", "8,485", "44%"]), ("53983137", ["4,509", "39%", "8,439", "43%"])], False),
    ("3", "Hober Malloc", [("53982496", ["4,497", "41%", "8,758", "42%"]), ("53982490", ["4,584", "39%", "8,914", "43%"])], False),
    ("4", "Jake Will", [("53961520", ["4,488", "71%", "8,254", "19%"]), ("53972924", ["4,426", "68%", "8,115", "22%"])], False),
    ("5", "Felix M Neumann", [("53974193", ["4,608", "62%", "9,226", "22%"]), ("53958390", ["4,494", "69%", "9,034", "19%"])], False),
    ("6", "flg", [("53975220", ["4,651", "58%", "9,258", "24%"]), ("53913042", ["4,588", "59%", "9,062", "25%"])], False),
    ("7", "Audun Ljone Henriksen", [("53993677", ["4,774", "42%", "9,821", "34%"]), ("53993640", ["4,472", "54%", "8,754", "29%"])], False),
    ("8", "Ender", [("53983642", ["4,518", "74%", "9,016", "11%"]), ("53979034", ["4,566", "71%", "8,942", "13%"])], False),
    ("9", "Xiangyu Liu", [("53980171", ["4,596", "47%", "9,088", "31%"]), ("53945598", ["4,743", "45%", "8,970", "34%"])], False),
    ("10", "Boey", [("53992481", ["4,688", "57%", "9,366", "22%"]), ("53986380", ["4,558", "59%", "9,144", "22%"])], False),
    ("11", "moriiiiiiiiim", [("53986435", ["4,716", "53%", "9,486", "26%"]), ("53986703", ["4,653", "55%", "9,257", "26%"])], False),
    ("12", "Vadasz & Ascalon", [("53993924", ["4,733", "58%", "9,180", "22%"]), ("53993965", ["4,645", "58%", "9,008", "23%"])], False),
    ("13", "dragon warrior", [("53977879", ["4,549", "46%", "9,197", "32%"]), ("53977843", ["4,644", "46%", "8,986", "33%"])], False),
    ("14", "Luca (ours)", [("53993338", ["4,749", "51%", "9,516", "28%"]), ("53993524", ["4,609", "49%", "9,124", "31%"])], True),
    ("15", "One Man Wrecking Machine", [("53993035", ["4,525", "45%", "8,832", "33%"]), ("53993719", ["4,447", "44%", "8,815", "35%"])], False),
    ("16", "M & J & M.ver2", [("53992808", ["4,568", "56%", "9,034", "24%"]), ("53993548", ["4,522", "55%", "8,920", "26%"])], False),
    ("17", "Orbit Goblins", [("53993589", ["4,647", "53%", "9,183", "25%"]), ("53925121", ["4,559", "53%", "8,829", "27%"])], False),
    ("18", "jonathan breitgand", [("53961919", ["4,514", "53%", "8,647", "25%"]), ("53964200", ["4,500", "51%", "8,267", "26%"])], False),
    ("19", "Azat Akhtyamov", [("53993868", ["4,545", "56%", "8,869", "24%"]), ("53993443", ["4,521", "56%", "8,925", "24%"])], False),
    ("20", "vkhydras", [("53993440", ["4,560", "43%", "8,446", "33%"]), ("53993428", ["4,443", "44%", "8,226", "32%"])], False),
]
FOOT_FIELD = ["Each team's two submissions: games played and overall 2p win-rate / 4p P(1st) across the field. Top-20 by the 2026-06-25 frozen ranking.",
              "Luca (bold, rank 14) is our team; 0.25 = even 4p share. Final 2026-07-07."]


def _fp(base, size):
    p = base.copy(); p.set_size(size); return p


def tw(s, fp=FP_BODY, size=BODY):
    return 0.0 if not s else TextPath((0, 0), s, size=size, prop=fp).get_extents().width / 72.0


def _check_foot(foot):
    usable = TARGET_W - 2 * MARGIN
    for ln in foot:
        w = tw(ln, FP_ITAL, FOOT)
        if w > usable:
            print(f"  !! footnote overflows ({w:.2f} > {usable:.2f} in): {ln[:60]}...")


def _fig(total_w, H):
    fig = plt.figure(figsize=(total_w, H), dpi=200)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off()
    ax.set_xlim(0, total_w); ax.set_ylim(0, H)
    return fig, ax


def draw_split(head, groups, foot, out):
    """Grouped table: rank + team on the first sub-row of each team, then one row per
    submission ([id] + numerics). Alternate teams banded."""
    ncol = len(head) - 2
    total_w = TARGET_W

    w_rank = max([tw(head[0], FP_BODY, HEAD)] + [tw(g[0], FP_BOLD if g[3] else FP_BODY) for g in groups])
    x_rank_r = MARGIN + w_rank
    x_lab = x_rank_r + GAP_RANK

    def cells(sub):
        return [sub[0]] + sub[1]
    wcol = []
    for i in range(ncol):
        w = tw(head[i + 2], FP_BODY, HEAD)
        for _, _, subs, bold in groups:
            fp = FP_BOLD if bold else FP_BODY
            for sub in subs:
                w = max(w, tw(cells(sub)[i], fp))
        wcol.append(w)
    r_edge = [0.0] * ncol
    r_edge[ncol - 1] = total_w - MARGIN
    for i in range(ncol - 2, -1, -1):
        r_edge[i] = r_edge[i + 1] - wcol[i + 1] - GUT

    n = sum(len(g[2]) for g in groups)
    H = (TOP_MARGIN + TOPRULE_PAD + HEADH + n * ROWH
         + PRE_BOTTOM + FOOT_GAP + len(foot) * FOOT_LH + BOT_MARGIN)
    fig, ax = _fig(total_w, H)

    def rule(y, x0=MARGIN, x1=total_w - MARGIN, lw=RULE_LW):
        ax.plot([x0, x1], [y, y], color=RULE, lw=lw, solid_capstyle="butt", zorder=3)

    def txt(x, y, s, ha="left", size=BODY, fp=FP_BODY, color=CHAR):
        ax.text(x, y, s, ha=ha, va="center", color=color, fontproperties=_fp(fp, size), zorder=4)

    y = H - TOP_MARGIN
    rule(y, lw=RULE_LW_EDGE)
    y -= TOPRULE_PAD
    txt(x_rank_r, y - HEADH / 2, head[0], ha="right", size=HEAD, color=GRAY)
    txt(x_lab, y - HEADH / 2, head[1], ha="left", size=HEAD, color=GRAY)
    for i in range(ncol):
        txt(r_edge[i], y - HEADH / 2, head[i + 2], ha="right", size=HEAD, color=GRAY)
    y -= HEADH
    rule(y)
    for gi, (rank, name, subs, bold) in enumerate(groups):
        fp = FP_BOLD if bold else FP_BODY
        gh = len(subs) * ROWH
        if not bold and gi % 2 == 1:
            ax.add_patch(plt.Rectangle((MARGIN, y - gh), total_w - 2 * MARGIN, gh,
                                       facecolor=BAND, edgecolor="none", zorder=1))
        for si, sub in enumerate(subs):
            if si == 0:
                txt(x_rank_r, y - ROWH / 2, rank, ha="right", fp=FP_BODY, color=GRAY)
                txt(x_lab, y - ROWH / 2, name, ha="left", fp=fp)
            for i, c in enumerate(cells(sub)):
                txt(r_edge[i], y - ROWH / 2, c, ha="right", fp=fp)
            y -= ROWH
        if rank == "—":          # separator only under the aggregate row
            rule(y)
    y -= PRE_BOTTOM
    rule(y, lw=RULE_LW_EDGE)
    y -= FOOT_GAP
    for ln in foot:
        ax.text(MARGIN, y - FOOT_LH / 2, ln, ha="left", va="center", color=GRAY,
                fontproperties=_fp(FP_ITAL, FOOT), zorder=4)
        y -= FOOT_LH
    fig.savefig(out, facecolor="white"); plt.close(fig)
    print("wrote", out, f"({total_w:.2f} x {H:.2f} in)")


def draw_combined(groups, foot, out):
    """Head-to-head: rank + team + opp submission id, then two column regions
    (merge / greedy) each showing COL_H over shared opponent-submission rows."""
    total_w = TARGET_W
    k = len(COL_IDX); ncol = 2 * k

    w_rank = max([tw("#", FP_BODY, HEAD)] + [tw(g[0], FP_BOLD if g[3] else FP_BODY) for g in groups])
    w_team = max([tw("team", FP_BODY, HEAD)] + [tw(g[1], FP_BOLD if g[3] else FP_BODY) for g in groups])
    subids = [s[0] for g in groups for s in g[2]]
    w_sub = max([tw("sub", FP_BODY, HEAD)] + [tw(x) for x in subids])
    x_rank_r = MARGIN + w_rank
    x_team = x_rank_r + GAP_RANK
    x_sub = x_team + w_team + GAP_SUB

    def cv(sub, region, ci):
        return (sub[1] if region == 0 else sub[2])[COL_IDX[ci]]
    wcol = [tw(COL_H[i % k], FP_BODY, HEAD) for i in range(ncol)]
    for _, _, subs, bold in groups:
        fp = FP_BOLD if bold else FP_BODY
        for sub in subs:
            for region in (0, 1):
                for ci in range(k):
                    idx = region * k + ci
                    wcol[idx] = max(wcol[idx], tw(cv(sub, region, ci), fp))
    r_edge = [0.0] * ncol
    r_edge[ncol - 1] = total_w - MARGIN
    for i in range(ncol - 2, -1, -1):
        gap = REGION_GAP if (i + 1) == k else GUT_C
        r_edge[i] = r_edge[i + 1] - wcol[i + 1] - gap

    left0 = r_edge[0] - wcol[0]; right0 = r_edge[k - 1]
    left1 = r_edge[k] - wcol[k]; right1 = r_edge[ncol - 1]
    if x_sub + w_sub > left0 - 0.05:
        print(f"  !! left block overlaps numerics (x_sub_end={x_sub + w_sub:.2f} > {left0 - 0.05:.2f})")

    n = sum(len(g[2]) for g in groups)
    H = (TOP_MARGIN + TOPRULE_PAD + REGIONH + HEADH + n * ROWH
         + PRE_BOTTOM + FOOT_GAP + len(foot) * FOOT_LH + BOT_MARGIN)
    fig, ax = _fig(total_w, H)

    def rule(y, x0=MARGIN, x1=total_w - MARGIN, lw=RULE_LW):
        ax.plot([x0, x1], [y, y], color=RULE, lw=lw, solid_capstyle="butt", zorder=3)

    def txt(x, y, s, ha="left", size=BODY, fp=FP_BODY, color=CHAR):
        ax.text(x, y, s, ha=ha, va="center", color=color, fontproperties=_fp(fp, size), zorder=4)

    y = H - TOP_MARGIN
    rule(y, lw=RULE_LW_EDGE)
    y -= TOPRULE_PAD
    # region-label row + cmidrules
    txt((left0 + right0) / 2, y - REGIONH / 2, "merge", ha="center", size=HEAD, fp=FP_BOLD, color=GRAY)
    txt((left1 + right1) / 2, y - REGIONH / 2, "greedy", ha="center", size=HEAD, fp=FP_BOLD, color=GRAY)
    rule(y - REGIONH, x0=left0, x1=right0)
    rule(y - REGIONH, x0=left1, x1=right1)
    y -= REGIONH
    # column-header row
    txt(x_rank_r, y - HEADH / 2, "#", ha="right", size=HEAD, color=GRAY)
    txt(x_team, y - HEADH / 2, "team", ha="left", size=HEAD, color=GRAY)
    txt(x_sub, y - HEADH / 2, "sub", ha="left", size=HEAD, color=GRAY)
    for i in range(ncol):
        txt(r_edge[i], y - HEADH / 2, COL_H[i % k], ha="right", size=HEAD, color=GRAY)
    y -= HEADH
    rule(y)
    for gi, (rank, name, subs, bold) in enumerate(groups):
        fp = FP_BOLD if bold else FP_BODY
        gh = len(subs) * ROWH
        if not bold and gi % 2 == 1:
            ax.add_patch(plt.Rectangle((MARGIN, y - gh), total_w - 2 * MARGIN, gh,
                                       facecolor=BAND, edgecolor="none", zorder=1))
        for si, sub in enumerate(subs):
            if si == 0:
                txt(x_rank_r, y - ROWH / 2, rank, ha="right", fp=FP_BODY, color=GRAY)
                txt(x_team, y - ROWH / 2, name, ha="left", fp=fp)
            txt(x_sub, y - ROWH / 2, sub[0], ha="left", fp=fp, color=GRAY)
            for region in (0, 1):
                for ci in range(k):
                    idx = region * k + ci
                    txt(r_edge[idx], y - ROWH / 2, cv(sub, region, ci), ha="right", fp=fp)
            y -= ROWH
        if rank == "—":          # separator only under the aggregate row
            rule(y)
    y -= PRE_BOTTOM
    rule(y, lw=RULE_LW_EDGE)
    y -= FOOT_GAP
    for ln in foot:
        ax.text(MARGIN, y - FOOT_LH / 2, ln, ha="left", va="center", color=GRAY,
                fontproperties=_fp(FP_ITAL, FOOT), zorder=4)
        y -= FOOT_LH
    fig.savefig(out, facecolor="white"); plt.close(fig)
    print("wrote", out, f"({total_w:.2f} x {H:.2f} in)")


if __name__ == "__main__":
    _check_foot(FOOT_H2H); _check_foot(FOOT_FIELD)
    draw_combined(GROUPS, FOOT_H2H, os.path.join(HERE, "eval_h2h.png"))
    draw_split(HEAD_FIELD, FIELD_GROUPS, FOOT_FIELD, os.path.join(HERE, "eval_field.png"))
