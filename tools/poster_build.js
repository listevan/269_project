#!/usr/bin/env node
// The CS 269 poster, 48 x 36 inches landscape, as a PowerPoint file.
//
//   NODE_PATH=<dir with pptxgenjs> node tools/poster_build.js      # writes docs/poster/poster_48x36.pptx
//
// Figures come from docs/poster/png/ (tools/poster_figures.py writes the SVGs; rsvg-convert makes the PNGs).
// Text sizes are poster sizes: title 60 pt, panel titles 34 pt, body 22 pt, setup lines and the table 19 to 20 pt.
// Heights are estimated from character counts (Calibri), so the layout prints each column's bottom edge; check
// the rendered poster after any text change.

const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");

const ROOT = path.resolve(__dirname, "..");
const PNG = path.join(ROOT, "docs", "poster", "png");
const OUT = path.join(ROOT, "docs", "poster", "poster_48x36.pptx");
const APPLY_THEME = process.env.PPTX_APPLY_THEME; // path to the pptx skill's scripts/apply_theme.js, optional

const THEME = {
  name: "Punishment poster",
  headFontFace: "Calibri",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "0B0B0B", lt1: "FFFFFF", dk2: "52514E", lt2: "F4F4F1",
    accent1: "2A78D6", accent2: "EB6834", accent3: "86B6EF", accent4: "B5B4AE", accent5: "1BAF7A", accent6: "6250D6",
    hlink: "2A78D6", folHlink: "6250D6",
  },
};

const W = 48, H = 36, MARGIN = 1.0, GAP = 0.6;
const COLS = 3;
const COL_W = (W - 2 * MARGIN - (COLS - 1) * GAP) / COLS;
const PAD = 0.45;            // inside a panel card
const INNER = COL_W - 2 * PAD;
const SP = 0.22;             // space between items inside a panel

function pngRatio(file) {    // width / height, read from the PNG header
  const b = fs.readFileSync(path.join(PNG, file));
  return b.readUInt32BE(16) / b.readUInt32BE(20);
}

const pres = new pptxgen();
pres.defineLayout({ name: "POSTER", width: W, height: H });
pres.layout = "POSTER";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.author = "Steven Li";
pres.title = "Can Punishment Alone Teach a Model Something New, and What Does It Take Away?";
const C = pres.SchemeColor;

pres.defineSlideMaster({
  title: "POSTER_FRAME",
  background: { color: C.background1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: MARGIN, y: 0.65, w: W - 2 * MARGIN, h: 1.1,
      fontSize: 60, bold: true, color: C.text1, margin: 0, valign: "top", align: "left" } } },
    { placeholder: { options: { name: "byline", type: "body", x: MARGIN, y: 1.8, w: W - 2 * MARGIN, h: 0.55,
      fontSize: 26, color: C.text2, margin: 0, valign: "top", align: "left" } } },
  ],
});

// ---------------------------------------------------------------- measuring text

const CHAR_EM = 0.47;        // average Calibri glyph width as a fraction of the point size
const LINE = 1.2;            // line height as a multiple of the point size
function lines(text, widthIn, pt) {
  const cpl = Math.max(8, Math.floor((widthIn * 72) / (CHAR_EM * pt)));
  return text.split("\n").reduce((n, p) => n + Math.max(1, Math.ceil(p.length / cpl)), 0);
}
function textHeight(text, widthIn, pt) {
  return (lines(text, widthIn, pt) * pt * LINE) / 72;
}

// ---------------------------------------------------------------- a panel: measure first, then draw a card behind

class Panel {
  constructor(number, title) {
    this.number = number;
    this.title = title;
    this.ops = [];
  }
  text(text, pt, opts = {}) {
    this.ops.push({ kind: "text", text, pt, opts });
    return this;
  }
  bullets(items, pt, opts = {}) {
    this.ops.push({ kind: "bullets", items, pt, opts });
    return this;
  }
  fig(file, widthIn, opts = {}) {
    this.ops.push({ kind: "fig", file, widthIn, ratio: pngRatio(file), opts });
    return this;
  }
  figs(files) {
    this.ops.push({ kind: "figs", files, ratio: Math.min(...files.map(pngRatio)) });
    return this;
  }
  table(rows, colW, pt, name) {
    this.ops.push({ kind: "table", rows, colW, pt, name });
    return this;
  }
  measure() {
    const titlePt = 34;
    let h = PAD + Math.max(0.7, textHeight(this.title, INNER - 0.95, titlePt)) + SP;
    for (const op of this.ops) {
      if (op.kind === "text") h += textHeight(op.text, INNER, op.pt) + SP;
      else if (op.kind === "bullets") {
        for (const it of op.items) h += textHeight(it, INNER - 0.45, op.pt) + (op.opts.gap ?? 0.1);
        h += SP - (op.opts.gap ?? 0.1);
      } else if (op.kind === "fig") h += op.widthIn / op.ratio + SP;
      else if (op.kind === "figs") h += (INNER - 0.4) / 2 / op.ratio + SP;
      else if (op.kind === "table") h += op.rows.reduce((s, r) => s + r.h, 0) + SP;
    }
    return h - SP + PAD;
  }
  draw(slide, x, y) {
    const h = this.measure();
    slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x, y, w: COL_W, h, rectRadius: 0.25, fill: { color: C.background2 }, line: { color: C.background2, width: 0 },
      objectName: `panel ${this.number} card`,
    });
    let cy = y + PAD;
    const titlePt = 34;
    const titleH = Math.max(0.7, textHeight(this.title, INNER - 0.95, titlePt));
    slide.addShape(pres.shapes.OVAL, { x: x + PAD, y: cy + 0.02, w: 0.68, h: 0.68, fill: { color: C.accent1 },
      line: { color: C.accent1, width: 0 }, objectName: `panel ${this.number} badge` });
    slide.addText(String(this.number), { x: x + PAD, y: cy + 0.02, w: 0.68, h: 0.68, fontSize: 26, bold: true,
      color: C.background1, align: "center", valign: "middle", margin: 0, isTextBox: true,
      objectName: `panel ${this.number} number` });
    slide.addText(this.title, { x: x + PAD + 0.95, y: cy, w: INNER - 0.95, h: titleH, fontSize: titlePt, bold: true,
      color: C.text1, margin: 0, valign: "top", isTextBox: true, objectName: `panel ${this.number} title` });
    cy += titleH + SP;
    for (const op of this.ops) {
      if (op.kind === "text") {
        const th = textHeight(op.text, INNER, op.pt);
        slide.addText(op.text, { x: x + PAD, y: cy, w: INNER, h: th, fontSize: op.pt, color: op.opts.color ?? C.text1,
          margin: 0, valign: "top", isTextBox: true, italic: !!op.opts.italic, objectName: op.opts.name ?? "text" });
        cy += th + SP;
      } else if (op.kind === "bullets") {
        const gap = op.opts.gap ?? 0.1;
        let th = 0;
        for (const it of op.items) th += textHeight(it, INNER - 0.45, op.pt) + gap;
        const runs = op.items.map((t, i) => ({ text: t, options: { bullet: { indent: 18 }, breakLine: i < op.items.length - 1,
          paraSpaceAfter: gap * 72 } }));
        slide.addText(runs, { x: x + PAD, y: cy, w: INNER, h: th, fontSize: op.pt, color: C.text1, margin: 0,
          valign: "top", isTextBox: true, objectName: op.opts.name ?? "bullets" });
        cy += th + SP - gap;
      } else if (op.kind === "fig") {
        const fh = op.widthIn / op.ratio;
        slide.addImage({ path: path.join(PNG, op.file), x: x + PAD + (INNER - op.widthIn) / 2, y: cy, w: op.widthIn, h: fh,
          objectName: op.file.replace(".png", "") });
        cy += fh + SP;
      } else if (op.kind === "figs") {
        const fw = (INNER - 0.4) / 2, fh = fw / op.ratio;
        op.files.forEach((f, i) => slide.addImage({ path: path.join(PNG, f), x: x + PAD + i * (fw + 0.4), y: cy, w: fw, h: fh,
          objectName: f.replace(".png", "") }));
        cy += fh + SP;
      } else if (op.kind === "table") {
        // a cell is a string, or { text, fill } to tint it (fill in hex, since table fills are hex-only)
        const rows = op.rows.map((r, i) => r.cells.map((c) => {
          const cell = typeof c === "string" ? { text: c } : c;
          return { text: cell.text, options: {
            bold: i === 0 || !!cell.bold, fontSize: op.pt, color: i === 0 ? C.text2 : C.text1, valign: "top",
            fill: { color: cell.fill ?? (i === 0 ? C.background2 : C.background1) }, margin: 0.05 } };
        }));
        const th = op.rows.reduce((s, r) => s + r.h, 0);
        slide.addTable(rows, { x: x + PAD, y: cy, w: INNER, colW: op.colW, rowH: op.rows.map((r) => r.h),
          border: { type: "solid", color: THEME.colors.lt2, pt: 1.5 }, autoPage: false, objectName: op.name ?? "table" });
        cy += th + SP;
      }
    }
    return y + h;
  }
}

// ---------------------------------------------------------------- content

const question = new Panel(1, "The question")
  .text("Post-training normally rewards the behaviours we want. A few recent methods get their improvements from " +
    "punishment alone. On-Policy Self-Adaptation (OPSA) pushes down the least likely tokens in a model's own answers, " +
    "with no teacher, reward or reference answer. Punishing an unlikely token moves its mass to the choices the model " +
    "already favors, so these methods should mainly reinforce existing habits. The behaviours post-training aims at " +
    "are the rare ones: following an unfamiliar rule, dropping a confident wrong answer, admitting that a problem " +
    "cannot be solved.", 22, { name: "question intro" })
  .bullets([
    "RQ1. Can teacher-free, negative-only training increase initially rare correct responses, and which rare good " +
    "behaviours does it suppress along the way?",
    "RQ2. What do token selection, teacher information, positive feedback and exposure to useful actions each add?",
  ], 22, { name: "research questions" })
  .fig("f0_mechanism.png", INNER);

const armRows = [
  { cells: ["Arm", "Pushes down", "Uses"], h: 0.42 },
  { cells: ["OPSA", "its own least likely 20% of tokens; entropy-weighted penalty, -0.5 to -1.0", "nothing"], h: 0.72 },
  { cells: ["Random", "the same number of tokens drawn at random, same penalty", "nothing; tests whether the selection matters"], h: 0.72 },
  { cells: ["Answer key", "the least likely tokens of answers the key marks wrong", "the answer key"], h: 0.72 },
  { cells: ["Teacher-selected", "the 20% the informed teacher likes least, OPSA's penalty", "a teacher: the same model with the rule in its prompt"], h: 0.72 },
  { cells: ["Teacher, negative only", "every token the teacher likes less than the student", "a teacher"], h: 0.72 },
  { cells: ["Teacher, both signs", "rewards and penalties: on-policy distillation", "a teacher; shows what positive feedback adds"], h: 0.72 },
  { cells: ["SFT", "nothing: trains on worked demonstrations", "demonstrations; shows the task is learnable"], h: 0.72 },
];
const methods = new Panel(2, "Methods: seven training arms, one model")
  .table(armRows, [3.1, 6.2, INNER - 9.3], 19, "arms table")
  .text("Qwen3-1.7B throughout, thinking off, T = 1.0, no top-p or top-k; 95% bootstrap intervals over problems or " +
    "items. Every study carries a frozen-model control, the base model resampled with another seed, as its noise floor. " +
    "OPSA's selector follows its released code: per pool of 16 responses, the lowest 20% of log-probabilities. All runs " +
    "on one AMD RX 9070 XT (16 GB): vLLM on ROCm for sampling, transformers with LoRA for training.", 20,
    { color: C.text2, name: "setup" });

const limits = new Panel(7, "Limits and next")
  .bullets([
    "One model size (1.7B), one seed per training arm, one learning rate chosen after OPSA collapsed at 1e-4, 60 LoRA " +
    "steps. \"The rare band does not move\" is a statement about that budget.",
    "The teacher is the same 1.7B model with the rule in its prompt; its per-token signal correlates 0.76 with length.",
    "The replay pilot is sized for large gaps only: 64 positions per rule, 8 + 8 continuations.",
    "On SUM the checkpoint was sampled once per problem at a 16k cap, the base eight times at 4k; 16% of its responses " +
    "are still unfinished.",
    "Next: force the rule into some rollouts and see whether on-policy feedback keeps it; study 1's adapters on SUM; " +
    "a second seed of OPSA and the both-sign teacher.",
  ], 20, { name: "limits" });

// Study 1 by rarity band: one row per arm, the change in the correct share with its 95% interval, from the
// committed reading table. A cell is tinted when the interval excludes zero (green above, orange below).
function readCsv(file) {
  const lines = fs.readFileSync(file, "utf8").trim().split("\n");
  const split = (line) => [...line.matchAll(/("([^"]*)"|[^,]*)(,|$)/g)].slice(0, -1).map((m) => m[2] ?? m[1]);
  const head = split(lines[0]);
  return lines.slice(1).map((l) => Object.fromEntries(split(l).map((v, i) => [head[i], v])));
}
const reading = readCsv(path.join(ROOT, "reports", "2026-11-24_hidden_rule_tables", "reading.csv"));
const BAND_ARMS = [["base_B", "Frozen model (noise floor)"], ["opsa", "OPSA"], ["random", "Random"],
  ["answer_key", "Answer key"], ["teacher_sel", "Teacher-selected"], ["teacher_neg", "Teacher, negative only"],
  ["teacher_both", "Teacher, both signs"], ["sft", "SFT"]];
const BANDS = [["rare (<5%)", "Rare\n7 operators, base under 5%"], ["uncommon (5-20%)", "Uncommon\n2 operators, base 5 to 20%"],
  ["common (>=20%)", "Common\n3 operators, base 27 to 44%"]];
function bandCell(arm, band) {
  const row = reading.find((r) => r.arm === arm && r.rarity === band);
  const m = row.d_correct.match(/^([+-][\d.]+) \[([+-][\d.]+), ([+-][\d.]+)\]$/);
  const lo = parseFloat(m[2]), hi = parseFloat(m[3]);
  return { text: row.d_correct, bold: lo > 0 || hi < 0, fill: lo > 0 ? "DDF3E8" : hi < 0 ? "FBE3DA" : undefined };
}
const bandRows = [
  { cells: ["Change in the correct share\n(tint: interval excludes zero)", ...BANDS.map((b) => b[1])], h: 0.72 },
  ...BAND_ARMS.map(([arm, label]) => ({ cells: [label, ...BANDS.map((b) => bandCell(arm, b[0]))], h: 0.42 })),
];
const BAND_COLW = [3.6, (INNER - 3.6) / 3, (INNER - 3.6) / 3, (INNER - 3.6) / 3];

const study1 = new Panel(3, "Study 1. Learning a hidden rule: no negative-only arm grew a rare correct answer")
  .text("\"Compute 7 # 4.\" with a # b = 2a + b never shown. Twelve operators spanning base correct rates 0 to 44%; " +
    "LoRA r = 16, 60 steps at lr 2e-5, 128 prompts per step; evaluated on held-out numbers, 24 items x 16 samples per " +
    "operator.", 20, { color: C.text2, name: "study 1 setup" })
  .fig("f1_hidden_rule.png", 10.0)
  .table(bandRows, BAND_COLW, 19, "study 1 by band")
  .bullets([
    "Rare band: every negative-only arm sits inside the noise floor; SFT on worked examples gains +0.28. The task is " +
    "learnable; the on-policy signals do not reach it.",
    "Common band: OPSA, random and both teacher-negative arms lose 10 to 30 points; the answer key arm gains 15. " +
    "OPSA's rollout entropy falls 0.245 to 0.097 while random's never moves: the selection is what sharpens.",
    "The both-sign teacher converges on one definition, a + b, for 111 of 128 final rollouts: the correct definition " +
    "never appears in the student's own samples, so its positive half has nothing to push up. SFT differs in that its " +
    "targets contain the answer.",
  ], 22, { name: "study 1 findings" });

const study3 = new Panel(6, "Study 3. Token replay: what OPSA punishes is neither useful nor harmful")
  .fig("f5a_replay_usefulness.png", 12.5)
  .bullets([
    "OPSA's tokens: U = -0.027 [-0.075, +0.011], damage p(1-p)U = -0.0053 [-0.0132, +0.0007]. Its 25% useful share " +
    "equals the noise floor (null 30% [20%, 39%]): per-token labels need more continuations, not more positions.",
    "Random tokens look useful (+0.068) only through near-certain tokens in correct answers (p > 0.99): replacing one " +
    "breaks the answer. OPSA's set is 59% words and 2% digits, with doubt words four times over-represented.",
  ], 22, { name: "study 3 findings" });

const study2 = new Panel(5, "Study 2. Side effects on refusals: the premise holds, the behaviour barely moves")
  .fig("f2_premise_audit.png", 11.0)
  .bullets([
    "OPSA's rule punishes the hedge that starts a refusal 58% of the time against a 20% base rate; the boxed refusal " +
    "itself only 2%, because by then it is near-certain. The second base seed reproduces every share within 0.02.",
  ], 22, { name: "premise finding" })
  .fig("f3_sum_refusal.png", 12.0)
  .bullets([
    "The released checkpoint keeps its refusals: among finished responses 0.351 to 0.315, -0.037 [-0.088, +0.014]; " +
    "the raw drop (-0.084) is entirely responses that never finish (+0.125). A drop of the size reward-based RL " +
    "produced on this data (0.30 to 0.08) is excluded. It hedges in loops: 31 hedge phrases per refusing response " +
    "against 3.9.",
  ], 22, { name: "behaviour finding" });

const checkpoint = new Panel(4, "The released checkpoint on MATH-500: better at first tries and at retries, paid for in length")
  .figs(["f4a_passk_curves.png", "f4b_passk_budget.png"])
  .bullets([
    "pass@1 0.676 to 0.816 (+0.140 [+0.094, +0.193]), pass@8 0.87 to 0.93 (+0.060 [+0.000, +0.120]); Sharpening " +
    "Tax(8) = -0.053 [-0.169, +0.068], not positive. Problems the base sometimes solved gain +0.21 pass@1 with pass@8 " +
    "unchanged (consistency); the 11 it never solved gain +0.455 pass@8, and 3 of 100 stay unsolved by the base after " +
    "88 attempts (coverage). No base temperature comes close.",
    "Mean answer 7,541 tokens against 1,026; 18% unfinished at 16,384; reflection markers 14.6 per 1k tokens against " +
    "1.8. The dominant wrong answer's mass falls 0.127 to 0.007, but no-answer rises 0.03 to 0.24: failures turn from " +
    "confident wrong answers into answers that never finish.",
  ], 22, { name: "checkpoint findings" });

// ---------------------------------------------------------------- the slide

const slide = pres.addSlide({ masterName: "POSTER_FRAME" });
slide.addText("Can Punishment Alone Teach a Model Something New, and What Does It Take Away?", { placeholder: "title" });
slide.addText("Steven Li  ·  UCLA MSCS  ·  CS 269, Fall 2026  ·  negative-only post-training (OPSA and its relatives) on Qwen3-1.7B",
  { placeholder: "byline" });

// the answer, with the four takeaways
const calloutY = 2.55, calloutH = 2.5;
slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: MARGIN, y: calloutY, w: W - 2 * MARGIN, h: calloutH, rectRadius: 0.25,
  fill: { color: C.accent3, transparency: 75 }, line: { color: C.accent3, transparency: 75, width: 0 }, objectName: "answer card" });
slide.addText("Punishment-only training cannot aim: it sharpens whatever the model already does most.", {
  x: MARGIN + PAD, y: calloutY + 0.3, w: W - 2 * MARGIN - 2 * PAD, h: 0.6, fontSize: 30, bold: true, color: C.text1,
  margin: 0, valign: "top", isTextBox: true, objectName: "the answer" });
const takeaways = [
  "1  It did not teach a hidden rule, with or without a teacher or an answer key; SFT on worked examples did.",
  "2  OPSA's rule does punish the hedging that starts a refusal, yet the trained checkpoint keeps its refusals.",
  "3  The released checkpoint gains at first tries and at retries, with no Sharpening Tax, paid for with seven times the tokens.",
  "4  The tokens OPSA punishes are neither useful nor harmful on average; the selection is about words and doubt, not correctness.",
];
const tkW = (W - 2 * MARGIN - 2 * PAD - 0.6) / 2;
takeaways.forEach((t, i) => {
  slide.addText(t, { x: MARGIN + PAD + (i % 2) * (tkW + 0.6), y: calloutY + 1.05 + Math.floor(i / 2) * 0.72, w: tkW, h: 0.7,
    fontSize: 22, color: C.text1, margin: 0, valign: "top", isTextBox: true, objectName: `takeaway ${i + 1}` });
});

// three columns
const top = calloutY + calloutH + 0.5;
const colX = (i) => MARGIN + i * (COL_W + GAP);
const bottoms = [];
let y = top;
y = question.draw(slide, colX(0), y) + GAP;
y = methods.draw(slide, colX(0), y) + GAP;
y = limits.draw(slide, colX(0), y);
bottoms.push(y);
y = top;
y = study1.draw(slide, colX(1), y) + GAP;
y = checkpoint.draw(slide, colX(1), y);
bottoms.push(y);
y = top;
y = study2.draw(slide, colX(2), y) + GAP;
y = study3.draw(slide, colX(2), y);
bottoms.push(y);

slide.addNotes("Poster for CS 269. Figures: docs/poster/*.svg, drawn by tools/poster_figures.py from the tables in reports/.");

(async () => {
  await pres.writeFile({ fileName: OUT });
  if (APPLY_THEME) {
    const { applyTheme } = require(APPLY_THEME);
    await applyTheme(OUT, THEME);
  }
  console.log(`wrote ${OUT}`);
  console.log(`columns start at y = ${top.toFixed(2)} in; bottoms: ${bottoms.map((b) => b.toFixed(2)).join(", ")} in; page height ${H} in, margin ${MARGIN}`);
})();
