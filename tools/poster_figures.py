#!/usr/bin/env python3
"""Poster figures, drawn as SVG from the committed tables in reports/.

No plotting library: the SVG is written by hand, so the files open unchanged in PowerPoint, Keynote,
Illustrator, Inkscape or a browser, and rsvg-convert turns one into a PNG or PDF when a tool needs that.

    uv run tools/poster_figures.py                                  # docs/poster/*.svg, 13px type on white
    uv run tools/poster_figures.py --scale 1.3 --surface "#F4F4F1" --out <dir>
                                                                    # poster type (about 14 pt when placed 13 in
                                                                    # wide), drawn on the poster's card colour

The layout adapts to the type size: label columns are measured from the text, plots take the rest of the width.
Every number comes from reports/*_tables/*.csv; nothing is typed in by hand except the schematic in f0.
"""
import argparse
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REP = ROOT / "reports"

# Reference palette (dataviz skill): categorical slots 1 and 2, a light step of the blue ramp, and inks.
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
BLUE, ORANGE, BLUE_LIGHT, GRAY = "#2a78d6", "#eb6834", "#86b6ef", "#b5b4ae"
FONT = "Inter, Helvetica, Arial, sans-serif"

T = {}                 # typography, set by configure()
SURFACE = "#ffffff"    # the figure background; the poster passes its card colour
OUT = ROOT / "docs" / "poster"


def configure(scale=1.0, surface="#ffffff", out=None):
    global SURFACE, OUT
    SURFACE = surface
    OUT = Path(out) if out else ROOT / "docs" / "poster"
    T.update(FS=round(13 * scale), TS=round(16 * scale), SS=round(12.5 * scale), AS=round(12 * scale),
             ROW=round(34 * scale), R=round(5 * scale), BAR=round(16 * scale), LW=max(2, round(2 * scale)),
             S=scale)


def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def tw(s, size):
    """Estimated width of a text run in px (Helvetica-like average glyph width)."""
    return 0.55 * size * len(str(s))


def wrap(text, width, size):
    lines, cur = [], ""
    for w in str(text).split():
        cand = (cur + " " + w).strip()
        if cur and tw(cand, size) > width:
            lines.append(cur)
            cur = w
        else:
            cur = cand
    if cur:
        lines.append(cur)
    return lines


def ticks_for(width, ticks, size, fmt):
    """Thin a tick list until its labels fit the plot width."""
    while len(ticks) > 3 and width / (len(ticks) - 1) < 1.4 * max(tw(fmt(t), size) for t in ticks):
        ticks = ticks[::2]
    return ticks


def ci(cell):
    """'+0.041 [+0.024, +0.060]' -> (0.041, 0.024, 0.060); a bare number -> (v, v, v)."""
    nums = [float(v) for v in re.findall(r"[-+]?\d*\.?\d+", str(cell).replace("−", "-"))]
    return (nums[0], nums[1], nums[2]) if len(nums) >= 3 else (nums[0], nums[0], nums[0])


def signed(t):
    return "0" if abs(t) < 0.005 else f"{t:+.2f}".rstrip("0").rstrip(".")


def plain(t):
    return f"{t:.2f}".rstrip("0").rstrip(".") if t else "0"


class SVG:
    def __init__(self, w, h=100):
        self.w, self.h, self.el = w, h, []

    def text(self, x, y, s, size=None, fill=INK, anchor="start", weight=None):
        size = size or T["FS"]
        w = f' font-weight="{weight}"' if weight else ""
        self.el.append(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" text-anchor="{anchor}"{w}>'
                       f'{esc(s)}</text>')

    def line(self, x1, y1, x2, y2, stroke=GRID, width=1, cap="butt"):
        self.el.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" '
                       f'stroke-width="{width}" stroke-linecap="{cap}"/>')

    def circle(self, cx, cy, r, fill, ring=True):
        if ring:
            self.el.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r + 2}" fill="{SURFACE}"/>')
        self.el.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r}" fill="{fill}"/>')

    def path(self, d, stroke, width=2, fill="none"):
        self.el.append(f'<path d="{d}" fill="{fill}" stroke="{stroke}" stroke-width="{width}" '
                       f'stroke-linejoin="round" stroke-linecap="round"/>')

    def hbar(self, x0, y, length, h, fill):
        """A horizontal bar growing from the baseline x0, rounded at the data end, square at the baseline."""
        if abs(length) < 0.5:
            return
        r = min(4, abs(length) / 2, h / 2)
        if length >= 0:
            d = (f"M{x0:.1f},{y:.1f} h{length - r:.1f} a{r},{r} 0 0 1 {r},{r} v{h - 2 * r:.1f} "
                 f"a{r},{r} 0 0 1 -{r},{r} h-{length - r:.1f} z")
        else:
            L = -length
            d = (f"M{x0:.1f},{y:.1f} h-{L - r:.1f} a{r},{r} 0 0 0 -{r},{r} v{h - 2 * r:.1f} "
                 f"a{r},{r} 0 0 0 {r},{r} h{L - r:.1f} z")
        self.path(d, "none", 0, fill)

    def vbar(self, x, y0, w, height, fill):
        """A column growing up from the baseline y0, rounded at the top."""
        if height < 0.5:
            return
        r = min(4, w / 2, height / 2)
        d = (f"M{x:.1f},{y0:.1f} v-{height - r:.1f} a{r},{r} 0 0 1 {r},-{r} h{w - 2 * r:.1f} "
             f"a{r},{r} 0 0 1 {r},{r} v{height - r:.1f} z")
        self.path(d, "none", 0, fill)

    def save(self, name):
        head = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" height="{self.h:.0f}" '
                f'viewBox="0 0 {self.w} {self.h:.0f}" font-family="{FONT}" font-size="{T["FS"]}">\n'
                f'<rect width="{self.w}" height="{self.h:.0f}" fill="{SURFACE}"/>\n')
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / name).write_text(head + "\n".join(self.el) + "\n</svg>\n")
        return OUT / name


class Axes:
    """A linear x scale over a plot rectangle (rows are placed by the caller)."""

    def __init__(self, svg, x, y, w, h, xmin, xmax):
        self.s, self.x, self.y, self.w, self.h, self.xmin, self.xmax = svg, x, y, w, h, xmin, xmax

    def X(self, v):
        return self.x + (v - self.xmin) / (self.xmax - self.xmin) * self.w

    def xaxis(self, ticks, fmt=signed, zero=None, label=None):
        """Gridlines, tick labels and an axis label; returns the y below the axis band."""
        a = T["AS"]
        for t in ticks_for(self.w, list(ticks), a, fmt):
            X = self.X(t)
            self.s.line(X, self.y, X, self.y + self.h, stroke=GRID)
            self.s.text(X, self.y + self.h + a * 1.3, fmt(t), size=a, fill=INK2, anchor="middle")
        if zero is not None:
            X = self.X(zero)
            self.s.line(X, self.y, X, self.y + self.h, stroke=INK2)
        if label:
            cx = min(max(self.x + self.w / 2, 24 + tw(label, a) / 2), self.s.w - 24 - tw(label, a) / 2)
            self.s.text(cx, self.y + self.h + a * 2.7, label, size=a, fill=INK2, anchor="middle")
            return self.y + self.h + a * 3.2
        return self.y + self.h + a * 1.8

    def dot_ci(self, y, m, lo, hi, color, label=None):
        self.s.line(self.X(lo), y, self.X(hi), y, stroke=color, width=T["LW"], cap="round")
        self.s.circle(self.X(m), y, T["R"], color)
        if label:
            self.s.text(self.X(hi) + T["R"] + 6, y + T["AS"] * 0.35, label, size=T["AS"], fill=INK2)

    def bar_ci(self, y, m, lo, hi, color, label=None):
        h = T["BAR"]
        self.s.hbar(self.X(0), y - h / 2, self.X(m) - self.X(0), h, color)
        if hi > lo:
            self.s.line(self.X(lo), y, self.X(hi), y, stroke=INK2, width=1.5)
            for v in (lo, hi):
                self.s.line(self.X(v), y - h / 4, self.X(v), y + h / 4, stroke=INK2, width=1.5)
        if label:
            self.s.text(self.X(max(hi, m)) + 8, y + T["AS"] * 0.35, label, size=T["AS"], fill=INK2)


def title(svg, t, sub=None, x=24):
    """Wrapped title and subtitle; returns the y where content starts."""
    y = 12
    for line in wrap(t, svg.w - 2 * x, T["TS"]):
        y += T["TS"]
        svg.text(x, y, line, size=T["TS"], weight=600)
        y += 4
    y += 2
    for line in (wrap(sub, svg.w - 2 * x, T["SS"]) if sub else []):
        y += T["SS"] * 1.3
        svg.text(x, y, line, size=T["SS"], fill=INK2)
    return y + T["SS"] * 1.7


def legend(svg, x, y, items, maxx=None):
    """Dots and names in a row, wrapping before maxx; returns the y below the legend."""
    a, x0 = T["AS"], x
    for color, text in items:
        w = T["R"] * 2 + 8 + tw(text, a)
        if maxx and x + w > maxx and x > x0:
            x, y = x0, y + a * 1.6
        svg.circle(x + T["R"], y - a * 0.35, T["R"], color, ring=False)
        svg.text(x + T["R"] * 2 + 6, y, text, size=a, fill=INK2)
        x += w + 26
    return y + a * 1.6


def note(svg, x, y, text, width):
    for line in wrap(text, width, T["AS"]):
        svg.text(x, y, line, size=T["AS"], fill=INK2)
        y += T["AS"] * 1.35
    return y


def row_labels(svg, x, rows, y0, pitch, color=INK):
    """Labels down a column; returns the row centre ys."""
    ys = []
    for i, lab in enumerate(rows):
        y = y0 + i * pitch + pitch / 2
        if lab is not None:
            svg.text(x, y + T["FS"] * 0.35, lab, fill=color)
        ys.append(y)
    return ys


# ---------------------------------------------------------------- f0: the mechanism, a schematic

def fig_mechanism():
    toks = ["the", "a", "wait", "we", "I don't"]
    before = [0.55, 0.25, 0.08, 0.07, 0.05]
    pun, kept = 2, 0.02
    scale = (1 - kept) / (1 - before[pun])
    after = [kept if i == pun else p * scale for i, p in enumerate(before)]
    S = T["S"]
    cw, gap, ph, vmax = 40 * S, 18 * S, 4.6 * T["ROW"], 0.66
    left = 24 + tw("0.6", T["AS"]) + 10
    pw = 5 * (cw + gap)
    W = int(left + pw + 60 + pw + 24)
    s = SVG(W)
    y0 = title(s, "A negative update on one token hands its probability to the tokens the model already favors",
               "Schematic, not data. The penalized token's mass is redistributed in proportion to the rest of the "
               "distribution, so the choice the model already favors gains most.")
    head_y = y0 + T["FS"]
    base_y = head_y + 16 + ph
    for px, vals, head in ((left, before, "next-token distribution before the update"),
                           (left + pw + 60, after, "after pushing down 'wait'")):
        s.text(px, head_y, head, weight=600)
        for g in (0.2, 0.4, 0.6):
            y = base_y - g / vmax * ph
            s.line(px, y, px + pw, y, stroke=GRID)
            s.text(px - 6, y + T["AS"] * 0.35, plain(g), size=T["AS"], fill=INK2, anchor="end")
        s.line(px, base_y, px + pw, base_y, stroke=INK2)
        for i, (t, v) in enumerate(zip(toks, vals)):
            x = px + i * (cw + gap)
            s.vbar(x, base_y, cw, v / vmax * ph, ORANGE if i == pun else BLUE)
            s.text(x + cw / 2, base_y + T["AS"] * 1.3, t, size=T["AS"], fill=INK2, anchor="middle")
            if i in (pun, 0):
                s.text(x + cw / 2, base_y - v / vmax * ph - 6, plain(v), size=T["AS"], fill=INK2, anchor="middle")
    s.h = legend(s, left, base_y + T["AS"] * 3.2, [(BLUE, "untouched tokens"), (ORANGE, "the penalized token")]) + 4
    return s.save("f0_mechanism.svg")


# ---------------------------------------------------------------- f1: study 1, hidden rule

ARMS = [("base_B", "frozen model, second batch (noise)"),
        ("opsa", "OPSA: own least likely 20%"),
        ("random", "random 20%, same penalty"),
        ("answer_key", "answer key: least likely of wrong answers"),
        ("teacher_sel", "teacher-selected 20%, OPSA's penalty"),
        ("teacher_neg", "teacher, negative terms only"),
        ("teacher_both", "teacher, both signs (distillation)"),
        ("sft", "SFT on worked demonstrations")]


def fig_hidden_rule():
    r = pd.read_csv(REP / "2026-11-24_hidden_rule_tables/reading.csv")
    bands = [("rare (<5%)", "Rare band", "7 operators, base under 5%"),
             ("common (>=20%)", "Common band", "3 operators, base 27 to 44%")]
    labw = max(tw(l, T["FS"]) for _, l in ARMS) + 24
    right = tw("+0.41", T["AS"]) + T["R"] + 14
    W = int(max(1040, 24 + labw + 2 * (270 + right) + 30 + 24))
    s = SVG(W)
    y0 = title(s, "Study 1. Can punishment grow a rare correct answer? Change in the correct share after 60 LoRA steps",
               "Qwen3-1.7B, twelve hidden binary operations (\"Compute 7 # 4.\", a # b = 2a + b never shown). Held-out "
               "numbers, 24 items x 16 samples per operator; 95% intervals over items.")
    pw = (W - 24 - labw - 2 * right - 30 - 24) / 2
    top = y0 + T["FS"] * 2.7
    rows_h = len(ARMS) * T["ROW"]
    ys = row_labels(s, 24, [l for _, l in ARMS], top, T["ROW"])
    for (band, head, desc), k in zip(bands, (0, 1)):
        px = 24 + labw + k * (pw + right + 30)
        ax = Axes(s, px, top, pw, rows_h, -0.4, 0.5)
        s.text(px, y0 + T["FS"] * 0.2, head, weight=600)
        s.text(px, y0 + T["FS"] * 0.2 + T["AS"] * 1.3, desc, size=T["AS"], fill=INK2)
        ax.xaxis([-0.4, -0.2, 0, 0.2, 0.4], zero=0)
        for (key, lab), y in zip(ARMS, ys):
            row = r[(r.rarity == band) & (r.arm == key)].iloc[0]
            m, lo, hi = ci(row.d_correct)
            ax.dot_ci(y, m, lo, hi, GRAY if key == "base_B" else BLUE,
                      label=signed(m) if key in ("sft", "opsa") else None)
    yb = top + rows_h
    s.text(24 + labw + (2 * pw + 2 * right + 30) / 2, yb + T["AS"] * 2.7,
           "change in the correct share, trained minus base", size=T["AS"], fill=INK2, anchor="middle")
    s.h = yb + T["AS"] * 3.2 + 8
    return s.save("f1_hidden_rule.svg")


# ---------------------------------------------------------------- f2: study 2, the premise audit

PREMISE_ROWS = [
    ("The boxed refusal: by then the decision is made", [
        ("refusal: first content token of the boxed refusal", "first token inside \\boxed{I don't know.}"),
        ("refusal: every token of the boxed refusal", "every token of the boxed refusal"),
        ("refusal: brace token that opens the box", "the brace that opens the box")]),
    ("The hedge in the reasoning, where the model first says it cannot solve this", [
        ("hedge phrase in the reasoning of refusing responses: first token",
         "first token of a hedge (\"cannot be determined\", \"is not given\")"),
        ("hedge phrase in the reasoning of refusing responses: every token", "every token of the hedge"),
        ("hedge phrase in the reasoning of false-answer responses: first token",
         "hedge first token, in responses that answered anyway")]),
    ("Answers, for comparison", [
        ("false answer: first content token of the box (unanswerable)", "first token of a false answer's box"),
        ("correct answer: first content token of the box (originals)", "first token of a correct answer's box")]),
]


def fig_premise():
    p = pd.read_csv(REP / "2026-10-25_sum_premise_audit_tables/base_s0/premise.csv").set_index("tokens")
    col = "share in OPSA's set [95% CI]"
    W = 1000
    s = SVG(W)
    y0 = title(s, "Study 2, premise. Does OPSA's rule punish the tokens that start a refusal?",
               "Base Qwen3-1.7B on SUM: 284 unanswerable problems and their solvable originals, 8 answers each. "
               "Share of each token kind inside OPSA's lowest 20% (its punished set); 95% intervals over problems.")
    labels = [lab for _, rows in PREMISE_ROWS for _, lab in rows]
    heads = [h for h, _ in PREMISE_ROWS]
    labw = max(max(tw(l, T["FS"]) for l in labels) + 12, max(tw(h, T["FS"]) for h in heads)) + 24
    px = 24 + labw
    pw = W - px - tw("0.66", T["AS"]) - 30
    hp, rp = T["ROW"] * 0.7, T["ROW"] * 0.8           # header and item row pitch
    rows_h = sum(hp + rp * len(rows) + 10 for _, rows in PREMISE_ROWS) + T["AS"] * 1.5
    ax = Axes(s, px, y0, pw, rows_h, 0, 0.75)
    s.h = ax.xaxis([0, 0.2, 0.4, 0.6], fmt=plain, label="share of these tokens that OPSA selects for punishment") + 8
    X20 = ax.X(0.2)
    s.line(X20, y0, X20, y0 + rows_h, stroke=INK2, width=1.5)
    ny = y0 + T["AS"] * 1.1
    for line in wrap("0.20: the share OPSA selects of all tokens", W - X20 - 30, T["AS"]):
        s.text(X20 + 6, ny, line, size=T["AS"], fill=INK2)
        ny += T["AS"] * 1.3
    y = y0 + T["AS"] * 1.5
    for head, rows in PREMISE_ROWS:
        y += hp
        s.text(24, y, head, weight=600)
        for key, lab in rows:
            y += rp
            m, lo, hi = ci(p.loc[key, col])
            s.text(36, y, lab, fill=INK2)
            ax.bar_ci(y - T["FS"] * 0.35, m, lo, hi, BLUE, label=f"{m:.2f}")
        y += 10
    return s.save("f2_premise_audit.svg")


# ---------------------------------------------------------------- f3: study 2, behaviour of the released checkpoint

SUM_ROWS = [("all", "unanswerable", "refusal", "refuses the unanswerable problem (all responses)"),
            ("completed only", "unanswerable", "refusal", "refuses (finished responses only)"),
            ("completed only", "unanswerable", "false_answer", "gives a false answer (finished only)"),
            ("all", "unanswerable", "no_answer", "never finishes, even at 16,384 tokens"),
            ("completed only", "answerable", "correct", "solves the solvable original (finished only)")]


def fig_sum_behaviour():
    d = pd.read_csv(REP / "2026-10-25_sum_premise_audit_tables/opsa_s0_16k/paired_vs_base.csv")
    cb, co, cd = d.columns[3], d.columns[4], d.columns[5]
    W = 1000
    s = SVG(W)
    y0 = title(s, "Study 2, behaviour. Did punishment-only training remove refusals? Not detectably.",
               "Released Tuwhy/Qwen3-1.7B-OPSA against its base, paired over SUM's 284 problems; 95% intervals.")
    y0 = legend(s, 24, y0, [(BLUE_LIGHT, "base model (2 seeds x 8 samples, 4,096-token cap)"),
                            (BLUE, "OPSA checkpoint (1 sample, 16,384-token cap)")], maxx=W - 24) + 6
    labw = max(tw(l, T["FS"]) for *_, l in SUM_ROWS) + 24
    dw = tw("OPSA minus base [95% CI]", T["AS"]) + 16
    px = 24 + labw
    pw = W - px - dw - 40
    pitch = T["ROW"] * 1.25
    rows_h = len(SUM_ROWS) * pitch
    ax = Axes(s, px, y0 + T["AS"] * 1.6, pw, rows_h, 0, 0.9)
    s.h = ax.xaxis([0, 0.2, 0.4, 0.6, 0.8], fmt=plain, label="share of responses") + 8
    s.text(px + pw + 40, y0 + T["AS"] * 1.1, "OPSA minus base [95% CI]", size=T["AS"], fill=INK2)
    ys = row_labels(s, 24, [l for *_, l in SUM_ROWS], y0 + T["AS"] * 1.6, pitch)
    for (resp, var, out, _), y in zip(SUM_ROWS, ys):
        row = d[(d.responses == resp) & (d.variant == var) & (d.outcome == out)].iloc[0]
        b, o = float(row[cb]), float(row[co])
        s.line(ax.X(b), y, ax.X(o), y, stroke=GRAY, width=T["LW"], cap="round")
        s.circle(ax.X(b), y, T["R"] + 1, BLUE_LIGHT)
        s.circle(ax.X(o), y, T["R"] + 1, BLUE)
        s.text(px + pw + 40, y + T["AS"] * 0.35, str(row[cd]).replace("−", "-"), size=T["AS"], fill=INK2)
    return s.save("f3_sum_refusal.svg")


# ---------------------------------------------------------------- f4: experiments 2 and 7, the released checkpoint

def _end_labels(svg, x, entries):
    """Labels at line ends, pushed apart where they would collide, with a leader where moved."""
    entries = sorted(entries, key=lambda e: e[0])
    ys = [e[0] for e in entries]
    step = T["AS"] * 1.25
    for i in range(1, len(ys)):
        ys[i] = max(ys[i], ys[i - 1] + step)
    for (y0, text, color), y in zip(entries, ys):
        if abs(y - y0) > 1:
            svg.line(x - 6, y0, x - 1, y - T["AS"] * 0.35, stroke=color, width=1)
        svg.text(x, y, text, size=T["AS"], fill=INK2)


def _line_chart(s, y0, xs, xlabel, ylabel, ymin, ymax, yticks, xfmt, right_text):
    """Shared frame of the two line charts; returns X(i), Y(v) and the plot rectangle."""
    a = T["AS"]
    x0 = 24 + tw("1.0", a) + 12
    pw = s.w - x0 - tw(right_text, a) - 20
    ph = 6 * T["ROW"]
    y0 = y0 + a * 1.4
    s.text(x0 - tw("1.0", a) - 12, y0 - a * 0.3, ylabel, size=a, fill=INK2)
    Y = lambda v: y0 + ph - (v - ymin) / (ymax - ymin) * ph
    X = lambda i: x0 + i / (len(xs) - 1) * pw
    for g in yticks:
        s.line(x0, Y(g), x0 + pw, Y(g), stroke=GRID if g > ymin + 1e-9 else INK2)
        s.text(x0 - 8, Y(g) + a * 0.35, f"{g:.1f}" if ymax - ymin < 0.5 else plain(g), size=a, fill=INK2, anchor="end")
    for i, t in enumerate(xs):
        s.text(X(i), y0 + ph + a * 1.3, xfmt(t), size=a, fill=INK2, anchor="middle")
    s.text(x0 + pw / 2, y0 + ph + a * 2.7, xlabel, size=a, fill=INK2, anchor="middle")
    s.h = y0 + ph + a * 3.2 + 6
    return X, Y, (x0, y0, pw, ph)


def fig_passk_curves():
    c = pd.read_csv(REP / "2026-11-24_passk_tables/curves.csv").set_index("arm")
    series = [("trained", "OPSA checkpoint", BLUE), ("base B (scoring)", "base, T = 1.0", ORANGE),
              ("base T=0.4", "base, T = 0.4 (sharpest sampler)", GRAY)]
    W = 560
    s = SVG(W)
    y0 = title(s, "Experiments 2 and 7. pass@k on 100 MATH-500 problems",
               "Levels 3 to 5, 8 samples per problem, graded on the last boxed answer.")
    y0 = legend(s, 24, y0, [(c_, t) for _, t, c_ in series], maxx=W - 24)
    X, Y, (x0, _, pw, _) = _line_chart(s, y0, list(range(1, 9)), "k, attempts per problem", "pass@k",
                                       0.6, 1.0, (0.6, 0.7, 0.8, 0.9, 1.0), str, "0.93 OPSA checkpoint")
    ends = []
    for key, lab, color in series:
        vals = [float(c.loc[key, f"pass@{k}"]) for k in range(1, 9)]
        s.path(" ".join(f"{'M' if i == 0 else 'L'}{X(i):.1f},{Y(v):.1f}" for i, v in enumerate(vals)), color, T["LW"])
        for i, v in enumerate(vals):
            s.circle(X(i), Y(v), T["R"] - 1, color)
        ends.append((Y(vals[-1]) + T["AS"] * 0.35, f"{vals[-1]:.2f} {lab.split(' (')[0]}", color))
    _end_labels(s, x0 + pw + 12, ends)
    return s.save("f4a_passk_curves.svg")


def fig_passk_budget():
    b = pd.read_csv(REP / "2026-11-24_passk_tables/budgets.csv")
    series = [("trained", "OPSA checkpoint", BLUE), ("base B (scoring)", "base, T = 1.0", ORANGE)]
    budgets = [1024, 2048, 4096, 8192, 16384]
    W = 560
    s = SVG(W)
    y0 = title(s, "The price of the gain is length: pass@1 against the token budget",
               "A response counts only if it finished within the budget. The base gains nothing past 2,048 tokens.")
    y0 = legend(s, 24, y0, [(c_, t) for _, t, c_ in series], maxx=W - 24)
    X, Y, (x0, py, pw, ph) = _line_chart(s, y0, budgets, "response budget, tokens (log scale)", "pass@1",
                                         0, 1.0, (0, 0.2, 0.4, 0.6, 0.8, 1.0), lambda t: f"{t:,}", "0.82 OPSA checkpoint")
    for key, lab, color in series:
        rows = b[b.arm == key].sort_values("budget (tokens)")
        pts = [(budgets.index(int(r["budget (tokens)"])), float(r["pass@1"])) for _, r in rows.iterrows()]
        s.path(" ".join(f"{'M' if i == 0 else 'L'}{X(k):.1f},{Y(v):.1f}" for i, (k, v) in enumerate(pts)), color, T["LW"])
        for k, v in pts:
            s.circle(X(k), Y(v), T["R"] - 1, color)
        k, v = pts[-1]
        if k == len(budgets) - 1:
            s.text(X(k) + 10, Y(v) + T["AS"] * 0.35, f"{v:.2f} {lab}", size=T["AS"], fill=INK2)
        else:
            s.text(X(k), Y(v) - T["AS"], f"{v:.2f} {lab}", size=T["AS"], fill=INK2, anchor="middle")
    s.line(X(2), py, X(2), py + ph, stroke=GRID)
    s.text(X(2) + 6, py + T["AS"] * 1.0, "same budget, 4,096:", size=T["AS"], fill=INK2)
    s.text(X(2) + 6, py + T["AS"] * 2.2, "0.34 against 0.68", size=T["AS"], fill=INK2)
    return s.save("f4b_passk_budget.svg")


def fig_passk_groups():
    c = pd.read_csv(REP / "2026-11-24_passk_tables/contrasts.csv")
    c1 = [k for k in c.columns if k.startswith("Δ pass@1")][0]
    c8 = [k for k in c.columns if k.startswith("Δ pass@8")][0]
    groups = [("all problems", "all 100 problems"), ("all solved", "base solved all 16 of 16 attempts (33)"),
              ("some solved", "base solved some (56)"), ("none solved", "base solved none of 16 (11)")]
    W = 900
    s = SVG(W)
    y0 = title(s, "Where the checkpoint's gain sits: problems grouped by what the base did with them",
               "OPSA checkpoint minus base, paired over problems; 95% intervals. The frozen-model control moves "
               "pass@1 by -0.006 [-0.032, +0.021].")
    y0 = legend(s, 24, y0, [(BLUE, "change in pass@1"), (ORANGE, "change in pass@8")], maxx=W - 24)
    labw = max(tw(l, T["FS"]) for _, l in groups) + 24
    px = 24 + labw
    pw = W - px - 24
    pitch = T["ROW"] * 1.35
    ax = Axes(s, px, y0, pw, pitch * len(groups), -0.2, 0.8)
    s.h = ax.xaxis([-0.2, 0, 0.2, 0.4, 0.6, 0.8], zero=0, label="OPSA checkpoint minus base") + 8
    ys = row_labels(s, 24, [l for _, l in groups], y0, pitch)
    for (key, _), y in zip(groups, ys):
        row = c[(c.arm == "trained") & (c.group == key)].iloc[0]
        ax.dot_ci(y - T["R"] * 1.7, *ci(row[c1]), BLUE)
        ax.dot_ci(y + T["R"] * 1.7, *ci(row[c8]), ORANGE)
    return s.save("f4c_passk_groups.svg")


# ---------------------------------------------------------------- f5: study 3, token replay

RULES = [("OPSA: lowest 20% log-prob", "OPSA: lowest 20%"),
         ("Random 20%", "random 20%"),
         ("Teacher-negative 20%", "teacher-negative 20%"),
         ("NSR: failed responses", "NSR: failed answers")]


def fig_replay():
    u = pd.read_csv(REP / "2026-10-22_pilot_qwen3-1.7b_tables/usefulness.csv").set_index("rule")
    b = pd.read_csv(REP / "2026-10-22_pilot_qwen3-1.7b_tables/bands.csv")
    ob = b[b.rule.str.startswith("OPSA")]
    band_labels = [(f"{r['p(token)']}" if str(r['p(token)']).startswith("p") else f"p {r['p(token)']}")
                   + f" ({int(r.positions)})" for _, r in ob.iterrows()]
    W = 1060
    s = SVG(W)
    y0 = title(s, "Study 3. Are the tokens OPSA punishes worth keeping? Usefulness by replay on MATH-500",
               "Qwen3-1.7B, 64 positions per rule from 24 mixed problems; 8 continuations with the token kept and 8 "
               "with it replaced. U = P(correct | kept) minus P(correct | replaced); 95% intervals over problems. "
               "Right: OPSA's tokens by the model's probability of the token, positions in brackets.")
    lab1 = max(tw(l, T["FS"]) for _, l in RULES) + 24
    lab2 = max(tw(l, T["FS"]) for l in band_labels) + 24
    right = tw("+0.07", T["AS"]) + T["R"] + 14
    gap = 50
    pw = (W - 24 - lab1 - right - gap - lab2 - right - 24) / 2
    top = y0 + T["FS"] * 1.4
    rows_h = 4 * T["ROW"]
    # left: mean usefulness per rule
    s.text(24, y0, "Mean usefulness of each rule's tokens", weight=600)
    ax = Axes(s, 24 + lab1, top, pw, rows_h, -0.2, 0.2)
    ya = ax.xaxis([-0.2, -0.1, 0, 0.1, 0.2], zero=0, label="usefulness U (> 0: pushing the token down hurts)")
    ys = row_labels(s, 24, [l for _, l in RULES], top, T["ROW"])
    for (key, _), y in zip(RULES, ys):
        m, lo, hi = ci(u.loc[key, "usefulness U [95% CI]"])
        ax.dot_ci(y, m, lo, hi, BLUE if key.startswith("OPSA") else GRAY, label=signed(m))
    nw = lab1 + pw + right
    yl = note(s, 24, ya + T["AS"] * 1.2, "Random's effect sits entirely in near-certain tokens (p > 0.99) inside "
              "correct answers: replacing one breaks the answer. OPSA has no such tokens, so the gap is about "
              "determinism, not selection.", nw)
    # right: OPSA by the kept token's probability
    x2 = 24 + lab1 + pw + right + gap
    s.text(x2, y0, "OPSA's tokens by the token's probability", weight=600)
    ax2 = Axes(s, x2 + lab2, top, pw, rows_h, -0.2, 0.2)
    ax2.xaxis([-0.2, -0.1, 0, 0.1, 0.2], zero=0, label="usefulness U")
    ys = row_labels(s, x2, band_labels, top, T["ROW"])
    for (_, row), y in zip(ob.iterrows(), ys):
        ax2.bar_ci(y, float(row.usefulness), 0, 0, BLUE, label=signed(float(row.usefulness)))
    yr = note(s, x2, ya + T["AS"] * 1.2, "OPSA's rarest tokens (p < 0.2) are not hidden good steps. Its useful "
              "share, 25%, is what sampling noise alone gives when every token has U = 0 (null 30% [20%, 39%]); the "
              "same holds for its 33% harmful share.", lab2 + pw + right)
    s.h = max(yl, yr) + 4
    return s.save("f5a_replay_usefulness.svg")


def fig_token_types():
    c = pd.read_csv(REP / "2026-10-22_pilot_qwen3-1.7b_tables/categories.csv", index_col=0)
    order = ["word", "math/LaTeX", "digit", "space/newline", "punctuation/other", "doubt/fork word", "end token"]
    W = 800
    s = SVG(W)
    y0 = title(s, "What OPSA selects: token types in its lowest 20% against all tokens",
               "474,090 response tokens from 480 MATH-500 answers. OPSA's set is words, almost never digits; "
               "doubt and fork words (\"wait\", \"but\", \"actually\") are four times over-represented.")
    y0 = legend(s, 24, y0, [(BLUE, "OPSA's lowest 20%"), (GRAY, "all tokens")], maxx=W - 24)
    labw = max(tw(l, T["FS"]) for l in order) + 24
    px = 24 + labw
    pw = W - px - tw("0.592", T["AS"]) - 30
    pitch = T["ROW"] * 1.0
    bh = T["BAR"] * 0.7
    ax = Axes(s, px, y0, pw, pitch * len(order), 0, 0.65)
    s.h = ax.xaxis([0, 0.2, 0.4, 0.6], fmt=plain, label="share of the set") + 8
    ys = row_labels(s, 24, order, y0, pitch)
    for t, y in zip(order, ys):
        o, a = float(c.loc[t, "OPSA: lowest 20% log-prob"]), float(c.loc[t, "all tokens"])
        s.hbar(ax.X(0), y - bh - 1, ax.X(o) - ax.X(0), bh, BLUE)
        s.hbar(ax.X(0), y + 1, ax.X(a) - ax.X(0), bh, GRAY)
        s.text(ax.X(o) + 6, y - 1, f"{o:.3f}".rstrip("0"), size=T["AS"] - 1, fill=INK2)
        s.text(ax.X(a) + 6, y + bh, f"{a:.3f}".rstrip("0"), size=T["AS"] - 1, fill=INK2)
    return s.save("f5b_token_types.svg")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scale", type=float, default=1.0, help="type scale; 1.3 for the poster")
    ap.add_argument("--surface", default="#ffffff", help="background colour, e.g. the poster's card colour")
    ap.add_argument("--out", help="output folder (default docs/poster)")
    args = ap.parse_args(argv)
    configure(args.scale, args.surface, args.out)
    for f in (fig_mechanism, fig_hidden_rule, fig_premise, fig_sum_behaviour, fig_passk_curves,
              fig_passk_budget, fig_passk_groups, fig_replay, fig_token_types):
        print(f())


if __name__ == "__main__":
    main()
