# Poster plan, drafted 2026-11-25

Working material for the CS 269 poster. Every number below is copied from the four reports in `reports/`; the
figures in this folder are drawn from the committed CSV tables by `tools/poster_figures.py` (rerun it after any
table changes; `rsvg-convert -w 2400 f1_hidden_rule.svg -o f1.png` gives a raster copy).

## The one-line answer

Punishment-only training cannot aim. It sharpens whatever the model already does most, teaches nothing the
model never produces on its own, and the tokens it punishes turn out neither useful nor harmful. The released
OPSA checkpoint does get better, but it buys the gain with seven times the tokens, not a sharper first guess, and
it keeps its refusals.

## Title and header

- Title (Steven's own, from the Abstract tab): Can Punishment Alone Teach a Model Something New, and What Does It
  Take Away?
- Steven Li, CS 269, UCLA, Fall 2026.
- All runs on one AMD RX 9070 XT (16 GB); Qwen3-1.7B throughout, thinking off, T = 1.0, 95% bootstrap
  intervals, and a frozen-model control (the base model resampled) in every study as the noise floor.

## Layout, three columns

| Column | Panels |
|---|---|
| 1 | 1 Question and mechanism (f0). 2 Methods: the arms table and setup. 7 Limits and next. |
| 2 | 3 Study 1, the hidden rule (f1, the headline). 4 The released checkpoint, pass@k and the token budget (f4a, f4b). |
| 3 | 5 Study 2, SUM premise and behaviour (f2, f3). 6 Study 3, token replay (f5a). |

The four takeaways sit in the header callout under the one-line answer, so there is no separate takeaways panel;
f4c and f5b are not on the poster (their numbers are in the text).

Built as PowerPoint, 48 x 36 in landscape: `docs/poster/poster_48x36.pptx`, by `tools/poster_build.js` (pptxgenjs)
from `docs/poster/png/` (poster-type renders of the figures). `tools/poster_make.sh` rebuilds everything from the
tables. Type: Calibri; title 60 pt, panel titles 34 pt, body 22 pt, table and setup lines 19 to 20 pt; figure
labels about 14 pt at print size.

## Panel 1. Question and mechanism

Text (condense the abstract's first paragraph): post-training normally rewards the behaviours we want, but a
few recent methods get their improvements from punishment alone. OPSA pushes down the least likely tokens in a
model's own answers with no teacher, reward or reference answer. Punishing an unlikely token moves its mass to
the choices the model already favors, so these methods should mainly reinforce existing habits. The behaviours
post-training aims at are the rare ones: following an unfamiliar rule, dropping a confident wrong answer,
admitting a problem cannot be solved.

- RQ1: can teacher-free, negative-only training increase initially rare correct responses, and which rare good
  behaviours does it suppress along the way?
- RQ2: what do token selection, teacher information, positive feedback and exposure to useful actions each add?

Figure: `f0_mechanism.svg` (schematic, labelled as such).

## Panel 2. Methods: the arms

| Arm | What it pushes down | Information used |
|---|---|---|
| OPSA | the model's own least likely 20% of tokens, entropy-weighted penalty -0.5 to -1.0 | none |
| Random | the same number of tokens drawn at random, same penalty | none; tests whether the selection matters |
| Answer key (NSR-style) | least likely tokens of responses the key marks wrong | the answer key |
| Teacher-selected | the 20% the informed teacher likes least, OPSA's penalty | a teacher (the same model with the definition in its prompt) |
| Teacher, negative only | every token the teacher likes less than the student | a teacher |
| Teacher, both signs | on-policy distillation: rewards and penalties | a teacher; shows what positive feedback adds |
| SFT | (positive only) worked demonstrations | demonstrations; shows the task is learnable |

OPSA's selector follows its released code: per pool of 16 responses, the lowest 20% of recomputed log-probs.

## Panel 3. Study 1, the hidden rule (headline)

Claim: no negative-only rule grew a correct answer that started rare, with or without a teacher. SFT on worked
examples did, so the task is learnable; the on-policy signals do not reach it.

Setup: "Compute 7 # 4." with a # b = 2a + b never shown; 12 operators spanning base correct rates 0 to 44%;
LoRA r = 16, 60 steps at lr 2e-5, 128 prompts per step; evaluation on held-out numbers, 24 items x 16 samples per
operator.

| Change in the correct share, held-out numbers | all 12 operators | rare band (7 ops, base < 5%) | common band (3 ops, base 27 to 44%) |
|---|---|---|---|
| base model, second batch (noise floor) | +0.004 [-0.006, +0.013] | +0.001 [-0.005, +0.007] | +0.003 [-0.030, +0.036] |
| OPSA | -0.050 [-0.065, -0.034] | -0.003 [-0.012, +0.007] | -0.209 [-0.260, -0.156] |
| random tokens, same penalty | -0.027 [-0.039, -0.015] | +0.000 [-0.006, +0.007] | -0.102 [-0.142, -0.062] |
| answer key | +0.041 [+0.024, +0.060] | -0.004 [-0.010, +0.002] | +0.155 [+0.099, +0.213] |
| teacher-selected, OPSA's penalty | -0.073 [-0.096, -0.050] | +0.022 [+0.002, +0.044] | -0.277 [-0.342, -0.207] |
| teacher, negative only | -0.094 [-0.112, -0.075] | -0.004 [-0.015, +0.009] | -0.302 [-0.363, -0.238] |
| teacher, both signs | +0.071 [+0.053, +0.090] | +0.012 [-0.006, +0.034] | +0.268 [+0.215, +0.319] |
| SFT demonstrations | +0.345 [+0.321, +0.368] | +0.280 [+0.255, +0.305] | +0.414 [+0.352, +0.474] |

Base correct share 0.118 overall, 0.012 in the rare band, 0.374 in the common band. The one on-policy entry
above the noise in the rare band (teacher-selected, +0.022) is a coincidence of operands: that arm writes
a + b + 1 for every symbol, which equals a + 2b whenever b = 1.

Side facts worth a line each:
- OPSA's rollout entropy falls 0.245 to 0.097 over training; random's never moves (0.24). The selection is what
  sharpens; the penalty alone mostly disrupts (random's no-answer share 0.30).
- Teacher, both signs converges on one definition, a + b, for 111 of 128 final rollouts: the hidden rule of one
  operator and the habit of another. Its positive half has nothing to push up, because the correct definition
  never appears in the student's own samples.
- Before training, OPSA's own rule singles out the rare correct answer: the most-punished decile of base
  responses is the most correct (14.4% against 4.8% in the least punished).
- At lr 1e-4 OPSA collapsed: every rollout hit the 512-token cap by step 24 and none answered. The released
  checkpoint's length growth, at speed.
- Following a definition stated in the prompt survives every on-policy arm (0.995 to 1.000); SFT drops to 0.87.

Figures: `f1_hidden_rule.svg` (dot and interval, rare and common bands). The per-operator scatter
`reports/2026-11-24_hidden_rule_change_by_rarity.svg` (one panel per arm, log-spaced base rate) can sit beside it
or replace it if there is room; it shows that every gain or loss comes from operators the base could already read.

## Panel 4. Study 2, SUM: what punishment removes

Claim, premise: OPSA's rule does assign negative updates to the tokens that start a refusal, not at the boxed
"I don't know", which is near-certain by then, but at the hedge in the reasoning where the model first writes
that something cannot be determined.

| Token kind (base model, SUM) | share inside OPSA's punished set (base rate 0.20) |
|---|---|
| first token inside \boxed{I don't know.} | 0.021 [0.011, 0.034] |
| the brace that opens the box | 0.586 [0.546, 0.627] |
| first token of a hedge phrase in refusing responses | 0.583 [0.554, 0.613] |
| every token of the hedge | 0.520 [0.502, 0.539] |
| hedge first token in responses that answered anyway | 0.656 [0.606, 0.705] |
| first token of a correct answer's box | 0.007 [0.002, 0.012] |

The second base seed reproduces every share within 0.02. At the brace after \boxed, the probability of the
refusal path is 0.51 in refusals, 0.01 in false answers and 0.0007 in correct answers: the box records a decision
taken earlier.

Claim, behaviour: the released checkpoint keeps its refusals within a few points of the base. A drop of the size
reward-based RL produced (0.30 to 0.08 in the SUM paper) is excluded.

| Paired over 284 problems | base | OPSA checkpoint (16,384-token cap) | difference [95% CI] |
|---|---|---|---|
| refusal on unanswerable, all responses | 0.348 | 0.264 | -0.084 [-0.126, -0.041] |
| refusal, finished responses only | 0.351 | 0.315 | -0.037 [-0.088, +0.014] |
| false answer, finished only | 0.649 | 0.685 | +0.037 [-0.014, +0.086] |
| never finishes | 0.009 | 0.134 | +0.125 [+0.088, +0.162] |
| correct on the solvable original, finished only | 0.528 | 0.769 | +0.241 [+0.197, +0.289] |

The raw refusal drop is entirely responses that never finished. The checkpoint's own responses hedge in loops:
about 31 hedge phrases per refusing response against 3.9 for the base, at median token probability 1.0 (base
0.636), so a further OPSA update would move little.

Figures: `f2_premise_audit.svg`, `f3_sum_refusal.svg`.

## Panel 5. The released checkpoint on MATH-500 (experiments 2 and 7)

Claim: the checkpoint is better at first tries and at retries, so its gain is not pure sharpening; it is bought
with seven times the tokens and eight times the self-checking per token.

| 100 problems, levels 3 to 5 | base (T = 1.0) | OPSA checkpoint | difference [95% CI] |
|---|---|---|---|
| pass@1 | 0.676 | 0.816 | +0.140 [+0.094, +0.193] |
| pass@8 | 0.87 | 0.93 | +0.060 [+0.000, +0.120], p = 0.065 |
| Sharpening Tax(8) | | | -0.053 [-0.169, +0.068] (not positive) |
| mean tokens | 1,026 | 7,541 | |
| still unfinished at 16,384 tokens | 0.022 (at 4,096) | 0.177 | |
| reflection markers per 1k tokens | 1.82 | 14.64 | |
| mass on the base's dominant wrong answer (67 problems) | 0.127 | 0.007 | -0.119 [-0.164, -0.078] |
| no-answer share | 0.03 | 0.239 | |

By group (what base batch A did with the problem over 16 attempts): all solved (33) +0.008 pass@1, +0.000
pass@8; some solved (56) +0.210 pass@1 with pass@8 unchanged (the consistency signature of sharpening); none
solved (11) +0.182 pass@1 and +0.455 [+0.182, +0.727] pass@8. Three problems remain unsolved by the base after 88
attempts and their derivations hold up; about 3 of 100 problems became solvable.

Sampling alone does not do this: the base's pass@1 stays 0.662 to 0.679 across T = 1.0, 0.8, 0.6, 0.4, and the
coldest sampler shows the sharpening signature (pass@8 down to 0.84) that the checkpoint lacks. At a matched
4,096-token budget the checkpoint's pass@1 is 0.344 against the base's 0.676; it needs the full 16,384 to reach
0.816, while the base gains nothing past 2,048.

Figures: `f4a_passk_curves.svg`, `f4b_passk_budget.svg`, `f4c_passk_groups.svg`.

## Panel 6. Study 3, token replay: is what OPSA punishes worth keeping?

Claim: pushing down OPSA's tokens is locally harmless, and at 8 + 8 continuations single tokens cannot be
labelled useful or harmful. What OPSA selects is clear: words and doubt words, almost never digits.

| Rule (64 positions each, MATH-500 levels 3 to 5) | usefulness U [95% CI] | share U > 0 (null band) | damage p(1-p)U |
|---|---|---|---|
| OPSA, lowest 20% log-prob | -0.027 [-0.075, +0.011] | 0.25 (0.24 [0.14, 0.34]) | -0.0053 [-0.0132, +0.0007] |
| random 20% | +0.068 [+0.008, +0.128] | 0.42 (0.25 [0.16, 0.36]) | +0.0007 [-0.0005, +0.0020] |
| teacher-negative 20% | -0.018 [-0.050, +0.020] | 0.19 (0.22 [0.12, 0.31]) | -0.0012 [-0.0057, +0.0024] |
| NSR, every token of a failed answer | +0.002 [-0.034, +0.034] | 0.28 (0.23 [0.14, 0.33]) | -0.0003 [-0.0033, +0.0025] |

Random's usefulness sits entirely in near-certain tokens (p > 0.99) inside correct answers (U = +0.174 on 18
positions); random tokens below p = 0.8 have U = 0.000 together. So the OPSA-minus-random gap, -0.096
[-0.174, -0.019], is a statement about deterministic tokens, not about OPSA's selection. Within OPSA's set the
rarest band (p < 0.2, 14 positions) is neutral (U = -0.009); only the 0.5 to 0.8 band shows a push helping
(U = -0.094, 20 positions, 85% in failed answers). Sign agreement between two kept passes is 64%: per-token
usefulness is noise at this size.

Token types: OPSA's set is 59% words and 2% digits (all tokens: 33% and 17%); doubt and fork words 2.2%
against 0.5%; 58% of OPSA's tokens are also teacher-negative tokens. The instrument passed its wiring gates
(HF and vLLM log-probs agree to 0.008; replay reproduces the rollout state, r = 0.78; second kept pass within
binomial noise).

Figures: `f5a_replay_usefulness.svg`, `f5b_token_types.svg`.

## Panel 7. Takeaways

1. Punishment alone did not teach a hidden rule, with or without a teacher or an answer key. Every negative-only
   arm stayed at the noise floor on rare operators and lost 10 to 30 points on common ones; SFT gained 28 points
   on the rare ones. What separates SFT from the rest is not the sign of the update but that its targets contain
   the answer.
2. The mechanism-level premise about refusals held (OPSA's rule punishes the hedging that starts a refusal at
   three times the base rate), but the behaviour-level effect is small or absent in the released checkpoint.
3. The released checkpoint's gain is real at first tries and retries, with no Sharpening Tax, and is paid for
   with seven times the tokens; its failures turn from confident wrong answers into answers that never finish.
4. The tokens OPSA punishes are neither useful nor harmful on average; its selection is about words and doubt,
   not about correctness.

## Panel 8. Limits and next

- One model size (1.7B), one seed per training arm, one learning rate chosen after the 1e-4 collapse, 60 LoRA
  steps. "The rare band does not move" is a statement about that budget.
- The teacher is the same 1.7B model with the definition in its prompt; its signal correlates 0.76 with length.
- Replay pilot: 64 positions per rule, 8 + 8 continuations; sized to see large gaps only.
- SUM comparison: the checkpoint sampled once per problem at a 16k cap, the base eight times at 4k; 16% of the
  checkpoint's responses still unfinished.
- The released checkpoint's true base (hybrid or -Base) not confirmed by weight distance.
- Next: the action-exposure diagnostic (force the rule into some rollouts and see whether on-policy feedback keeps
  it); study 1's adapters on SUM; a second seed of OPSA and teacher-both (about 2 GPU-hours).

## Figure inventory

| File | Panel | Source tables |
|---|---|---|
| f0_mechanism.svg | 1 | schematic, toy numbers |
| f1_hidden_rule.svg | 3 | 2026-11-24_hidden_rule_tables/reading.csv |
| f2_premise_audit.svg | 4 | 2026-10-25_sum_premise_audit_tables/base_s0/premise.csv |
| f3_sum_refusal.svg | 4 | 2026-10-25_sum_premise_audit_tables/opsa_s0_16k/paired_vs_base.csv |
| f4a_passk_curves.svg | 5 | 2026-11-24_passk_tables/curves.csv |
| f4b_passk_budget.svg | 5 | 2026-11-24_passk_tables/budgets.csv |
| f4c_passk_groups.svg | 5 | 2026-11-24_passk_tables/contrasts.csv |
| f5a_replay_usefulness.svg | 6 | 2026-10-22_pilot_qwen3-1.7b_tables/usefulness.csv, bands.csv |
| f5b_token_types.svg | 6 | 2026-10-22_pilot_qwen3-1.7b_tables/categories.csv |
| reports/2026-11-24_hidden_rule_change_by_rarity.svg | 3 (optional) | by_split and per-operator tables |

Colors: blue #2a78d6 and orange #eb6834 (two series), light blue #86b6ef for "before" in the dumbbell, gray
#b5b4ae for context series. Fonts: Inter or Helvetica; the SVG text stays editable in PowerPoint and Inkscape.

## Open items

1. Open the .pptx in PowerPoint and check the text fit once with the real Calibri (the preview here used a
   substitute font); the text boxes carry about 15% slack.
2. Uncommitted work in the repo as of 2026-11-25: the 2026-10-22 analysis changes (signal gates 4 and 5, the
   share_null and bands tables, the rebuilt pilot report), `docs/`, and this folder. Tests pass (67 passed, 4
   skipped).
3. Re-check novelty against the fork github.com/duterscmy/On-Policy-Self-Adaptation before submission.
4. Optional runs if time allows: a second seed of OPSA and teacher-both; study 1's adapters through the SUM
   pipeline; 32 + 32 continuations at the replay positions.
