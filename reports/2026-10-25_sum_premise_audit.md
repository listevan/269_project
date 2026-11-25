# Study 2, measured first: refusal on SUM and the premise audit (2026-10-25 runs)

The proposal's experiment 5 asks what punishment-only training removes by accident, starting with the willingness
to say a problem cannot be solved. Its plan says to measure two things before anything else: the base model's
refusal and false-answer rates on unanswerable problems with accuracy on their solvable originals, and the premise
audit: in base responses that refused, would OPSA's rule have selected the refusal tokens for punishment? "The
prediction only stands if refusal tokens are in the punished set." This file reports both, plus the training-free
first pass on the released OPSA checkpoint, on this machine's RX 9070 XT. Every number comes from the tables in
`reports/2026-10-25_sum_premise_audit_tables/`.

## Summary

**The premise holds, but not where the plan first looked.** The boxed refusal itself is almost never in OPSA's
punished set: 2% of the first tokens of `\boxed{I don't know.}` and 8% of all its
tokens, against the 20% the rule selects overall. By the time the model opens the box, the refusal is near-certain
(median token probability 1.0). The decision is made earlier, in the reasoning, where the model first
writes that something "is not given", "cannot be determined", "we don't know": **58% of those hedge phrases'
first tokens are in OPSA's set** (52% of all their tokens), about three times the base rate, at a median
probability of 0.636. The token that opens the refusal box is selected 59% of the time, because the model
wavers there between `{` and `{\`. In responses that hedged and then answered anyway, the hedges are punished even
more often (66%). So OPSA does assign negative updates to the tokens that start a refusal, and the
side-effect prediction stands. The second base seed reproduces every one of these shares within 0.02.

**Base rates.** Qwen3-1.7B refuses 35% of the unanswerable problems and gives a false answer on
64%; on the solvable originals it is correct 52% of the time and over-refuses
3%. The frozen-model control (a second batch with another seed) moves no rate by more than 0.01, so that
is the noise floor a trained model must beat. If every worded false answer (`\text{Undefined}`, `\text{No solution}`,
`\text{Infinite}`, ...) were counted as a refusal instead, the refusal rate would be at most 42%.

**The decision point, read from the logits.** At the brace after `\boxed`, the probability of the canonical refusal
path (`{` then `I`) is 0.51 in responses that refused, 0.01 in false answers and
0.0007 in correct answers to the originals. The entropy at that position is 0.5 in refusals and
0.008 in correct answers: the box only records a decision taken earlier.

**The released OPSA checkpoint, at a 4,096-token budget, mostly fails to finish.** 71% of its responses to
unanswerable problems and 58% to the originals hit the cap (base: about 1%), its mean response is
3565 tokens against 947, and 64% of its responses repeat a 20-token span three or
more times (base 18%). Its raw refusal rate falls to 20% and its false-answer rate to
16%, but 64% of its responses never reach an answer, so these rates are not comparable with the
base model's. Among the 668 unanswerable responses that did finish it refuses 57% and false-answers
42%, and on the 952 finished originals it is correct 85%; but the responses that finish are the
ones where the model gave up or solved quickly, so this is selection, not an effect. The length and repetition
blow-up is itself one of the side effects the plan tracks. The refusal comparison needed a run with a larger budget; see the addendum at the end: at 16,384 tokens the checkpoint's refusal rate is within a few points of the base's.

**The trained model's own distribution has sharpened.** On its own responses, the hedge phrases the OPSA checkpoint
writes have a median probability of 1.0 (base 0.636), and it writes about
31 hedge phrases per refusing response against 3.9 for the base: it hedges in loops.

## What ran

| Arm | Model | Seed | Stage 1 (vLLM) | Audit (transformers) |
|---|---|---|---|---|
| base s0 | `Qwen/Qwen3-1.7B` | 0 | 28 min | 21 min |
| base s1 (frozen-model control) | `Qwen/Qwen3-1.7B` | 1 | 28 min | 21 min |
| OPSA s0 | `Tuwhy/Qwen3-1.7B-OPSA` | 0 | 190 min | 67 min |

Each arm: SUM's 284 unanswerable test problems and their 284 solvable originals, 8 answers each at T = 1.0 with no
top-p or top-k, thinking off, at most 4,096 response tokens, the dataset's own instruction ("... If you don't know
the answer, reply with `\boxed{I don't know.}`."). Grading: a box whose content says "I don't know" (or another
statement that the answer cannot be given) is a refusal; any other box is an answer, graded with math-verify
against the gold on the originals and counted as a false answer on the unanswerable items; no complete box is
"no answer". The audit scores every token with the model itself, applies OPSA's rule (lowest 20% of log-probabilities
in pools of 16 responses, entropy-weighted advantages from -0.5 to -1.0) and the random rule on the same pools,
and reads the decision point with two extra forwards per response, one after a bare `{` and one after `{\`, so the
refusal-path probability is the same quantity for every response. Intervals are 95% bootstrap over problem pairs.

## Reading the tables

- **Rates.** The two base seeds agree to within 0.01 on every rate. The pair row counts samples where the model both
  refused the unanswerable version and solved the original: 0.224 for the base.
- **Premise.** The refusal tokens that matter for OPSA's update are the hedges in the reasoning and the box opening,
  not the boxed statement. In the trained checkpoint's own responses, OPSA's rule still selects 20% of tokens by
  rank, but those tokens sit at probabilities near 1 (share with p < 0.5: 0.053 for hedge starts, base
  0.41), so a further OPSA update on them would move little.
- **Decision point.** `P('{')` is the probability of a bare brace (the other mass is mostly `{\`, a LaTeX wrapper);
  `P('I' | '{')` is read after a bare brace whether or not the model wrote one. The worded path (`{\` then `text`)
  covers both `\text{I don't know}` and worded non-refusals such as `\text{Undefined}`, so it is an upper bound.
- **Where the punished tokens sit.** In base refusals, OPSA's tokens are two thirds plain words and 2.5% explicit
  doubt or fork words; they split about evenly before and after the first hedge phrase. In the OPSA checkpoint's
  refusals, 11% of the punished tokens are doubt or fork words and 92% come after the first hedge: the loops.
- **Worded false answers.** The tables list every `\text{...}` box the grader counted as a false answer, for checking.
  Most are genuine answers to a mis-specified problem ("No solution", "Infinite", a range); `\text{Undefined}` is
  arguable. Reclassifying all of them raises the base refusal rate by at most 0.07.

## Caveats

- **Budget.** 4,096 tokens suits the base model (1% at cap) but not the OPSA checkpoint (64% at cap). The plan's
  no-GPU note already allowed 16,000 tokens for it. Until a larger-budget run exists, nothing here says whether OPSA
  training removed refusals; it says the checkpoint got long and repetitive at this budget.
- **Hedge detection is a word list.** It catches the phrasings in these responses (checked by hand on the smoke run
  and by an independent review), including markdown bold and curly apostrophes, but it is a proxy for "the point
  where the model decides it cannot solve this".
- **One model size.** Qwen3-1.7B only; the plan also names the 4B pair.
- **No temperature sweep yet.** The sharpening control (base at lower temperatures) has not run.
- **Checkpoint provenance.** The OPSA card says it was fine-tuned from `Qwen3-1.7B-Base` while its tokenizer and chat
  template are the hybrid model's; the plan's check of the true base by weight distance has not been done.

## Next steps

1. ~~OPSA checkpoint with a 16,384-token budget~~ done, see the addendum: refusals within a few points of the base. If
   a 3 to 4 point effect matters, a second single-sample batch with another seed halves the interval.
2. **Replay at the first hedge phrase** (experiment 1's question for experiment 5): keep the hedge against redraw it,
   and see whether saying the problem cannot be solved leads to the right outcome. The positions are in
   `decisions.jsonl` (`hedges`) and the replay machinery exists.
3. **Temperature sweep of the base** on SUM, the sharpening control.
4. **MATH-500 doubt-when-wrong and verbalized confidence** (measures 2 and 3) for base and checkpoint.

## Files

In git: `reports/2026-10-25_sum_premise_audit_tables/<arm>/` holds each arm's `rates.csv`, `premise.csv`,
`decision.csv`, `first_tokens.csv`, `punished_mass.csv`, `worded_false_answers.csv`, `config.json`,
`rollout.meta.json` and `audit.json`, plus the cross-arm `2026-10-25_sum_summary.json`. The raw runs
(`runs/2026-10-25_sum_*`, gitignored, 1.0 GB together) hold `problems.jsonl`, `rollouts.jsonl`, `scores.jsonl`,
`decisions.jsonl`, `selected.jsonl`; `uv run src/premise_audit.py --run <dir>` rebuilds a run's tables.

# Addendum, 2026-10-25 evening: the OPSA checkpoint at a 16,384-token budget

A rerun of the checkpoint with the cap raised to 16,384 tokens and one sample per prompt (568 responses, 197 min of
generation, mean response 7577 tokens, 15.7% still at the cap). The comparison below is paired by problem
against the two base batches (8 samples each at the 4,096 cap, which the base model almost never hits), with 95%
bootstrap intervals over the 284 problems.

**Answer to "does punishment remove refusals?": not detectably.** The raw refusal rate falls from 0.348 to
0.264 (-0.084 [-0.126, -0.041]), but the whole of that gap is responses that never finished: no-answer rises
by the same amount (+0.125 [+0.088, +0.162]). Among responses that did finish, refusal moves from 0.351 to
0.315 (-0.037 [-0.088, +0.014], p = 0.15) and false answers from 0.649 to 0.685
(+0.037 [-0.014, +0.086]). A drop of a few points is possible, a drop of the size reward-based RL produced (0.30 to 0.08) is
excluded. Meanwhile the checkpoint solves the originals far more often (0.520 to 0.687, +0.166 [+0.123, +0.210]) and
over-refuses them less (0.032 to 0.014). The pair measure, refusing the unanswerable version and
solving the original in the same sample, is unchanged (0.211 [0.165, 0.261] against 0.219 to 0.224 for the base).

So the mechanism-level premise held (OPSA's rule punishes the hedging that starts a refusal, at three times the base
rate) but the behaviour-level effect is small or absent: the trained model keeps its willingness to say "I don't
know" within a few points of the base while gaining about 17 points of accuracy. What the checkpoint clearly did
acquire is length and repetition: eight times the base's mean length, 0.831 of its unanswerable
responses with severe repetition against 0.18 for the base, and one in six responses unfinished even at 16k tokens.

**Caveats specific to this run.** One sample per problem (the plan's own point: the 284 problems carry the
variance, but the intervals are about ±5 points, so a 3 to 4 point effect cannot be resolved here); the base was
sampled at a 4,096 cap and the checkpoint at 16,384, which only matters for the 1% of base responses at the cap;
16% of the checkpoint's responses are still unfinished, and their eventual outcome is unknown.

## Paired comparison over the same 284 problems

| responses | variant | outcome | base (2 seeds, 8 samples, 4k) | OPSA (1 sample, 16k) | OPSA − base [95% CI] | bootstrap p |
|---|---|---|---|---|---|---|
| all | unanswerable | refusal | 0.348 | 0.264 | -0.084 [-0.126, -0.041] | 0.0 |
| all | unanswerable | false_answer | 0.643 | 0.602 | -0.041 [-0.095, +0.014] | 0.14 |
| all | unanswerable | no_answer | 0.009 | 0.134 | +0.125 [+0.088, +0.162] | 0.0 |
| all | answerable | correct | 0.52 | 0.687 | +0.166 [+0.123, +0.210] | 0.0 |
| all | answerable | wrong | 0.434 | 0.204 | -0.230 [-0.275, -0.186] | 0.0 |
| all | answerable | refusal | 0.032 | 0.014 | -0.018 [-0.031, -0.003] | 0.019 |
| completed only | unanswerable | refusal | 0.351 | 0.315 | -0.037 [-0.088, +0.014] | 0.15 |
| completed only | unanswerable | false_answer | 0.649 | 0.685 | +0.037 [-0.014, +0.086] | 0.15 |
| completed only | answerable | correct | 0.528 | 0.769 | +0.241 [+0.197, +0.289] | 0.0 |
| completed only | answerable | wrong | 0.44 | 0.215 | -0.225 [-0.270, -0.181] | 0.0 |
| completed only | answerable | refusal | 0.032 | 0.016 | -0.016 [-0.031, +0.000] | 0.054 |

## The 16k run's own rates (95% CI over problems)

| variant | problems | responses | refusal | false_answer | no_answer | refusal: \boxed{I don't know} form | mean tokens | at generation cap | severe repetition | correct | wrong |
|---|---|---|---|---|---|---|---|---|---|---|---|
| unanswerable | 284 | 284 | 0.264 [0.215, 0.317] | 0.602 [0.546, 0.655] | 0.134 [0.095, 0.173] | 0.264 [0.215, 0.317] | 8307.0 | 0.183 [0.141, 0.229] | 0.831 [0.789, 0.873] |  |  |
| answerable | 284 | 284 | 0.014 [0.000, 0.028] |  | 0.095 [0.063, 0.130] | 0.014 [0.000, 0.028] | 6847.0 | 0.130 [0.092, 0.173] | 0.665 [0.613, 0.718] | 0.687 [0.634, 0.739] | 0.204 [0.158, 0.250] |
| pair: refuses the unanswerable and solves the original (same sample index) | 284 | 284 |  |  |  |  |  |  |  | 0.211 [0.165, 0.261] |  |

# Tables (first run, 4,096-token budget)

## Rates on the unanswerable problems (2026-10-25)

| measure | base s0 | base s1 | OPSA s0 | noise floor \|s0 − s1\| |
|---|---|---|---|---|
| refusal | 0.349 [0.307, 0.389] | 0.346 [0.303, 0.386] | 0.201 [0.162, 0.240] | 0.003 |
| refusal: \boxed{I don't know} form | 0.341 [0.299, 0.382] | 0.341 [0.298, 0.380] | 0.201 [0.162, 0.240] | 0.000 |
| false_answer | 0.642 [0.603, 0.683] | 0.644 [0.604, 0.685] | 0.160 [0.127, 0.195] | 0.002 |
| no_answer | 0.009 [0.004, 0.015] | 0.010 [0.004, 0.017] | 0.639 [0.591, 0.687] | 0.001 |
| mean tokens | 947.0 | 954.0 | 3565.0 | 7.000 |
| at generation cap | 0.010 [0.004, 0.017] | 0.011 [0.004, 0.019] | 0.706 [0.661, 0.751] | 0.001 |
| severe repetition | 0.176 [0.147, 0.206] | 0.186 [0.156, 0.217] | 0.640 [0.604, 0.676] | 0.010 |

## Rates on the solvable originals

| measure | base s0 | base s1 | OPSA s0 | noise floor \|s0 − s1\| |
|---|---|---|---|---|
| correct | 0.519 [0.467, 0.570] | 0.522 [0.472, 0.571] | 0.412 [0.361, 0.463] | 0.003 |
| wrong | 0.438 [0.391, 0.486] | 0.430 [0.384, 0.478] | 0.074 [0.049, 0.099] | 0.008 |
| refusal | 0.030 [0.019, 0.042] | 0.034 [0.022, 0.048] | 0.013 [0.004, 0.025] | 0.004 |
| no_answer | 0.013 [0.006, 0.021] | 0.015 [0.008, 0.023] | 0.501 [0.449, 0.553] | 0.002 |
| mean tokens | 945.0 | 947.0 | 3193.0 | 2.000 |
| at generation cap | 0.013 [0.007, 0.021] | 0.015 [0.008, 0.024] | 0.581 [0.529, 0.632] | 0.002 |
| severe repetition | 0.193 [0.160, 0.228] | 0.192 [0.159, 0.227] | 0.504 [0.461, 0.544] | 0.001 |

## Pairs: refuses the unanswerable version and solves the original (same sample)

| base s0 | base s1 | OPSA s0 |
|---|---|---|
| 0.224 [0.185, 0.265] | 0.219 [0.182, 0.257] | 0.138 [0.106, 0.173] |

## Rates among responses that did not hit the generation cap

| arm | variant | responses | not at cap | share at cap | refusal (all) | refusal (not at cap) | false_answer (all) | false_answer (not at cap) | no_answer (all) | no_answer (not at cap) | mean tokens (not at cap) | correct (all) | correct (not at cap) | wrong (all) | wrong (not at cap) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base s0 | unanswerable | 2272 | 2250 | 0.01 | 0.349 | 0.352 | 0.642 | 0.648 | 0.009 | 0.0 | 916 | nan | nan | nan | nan |
| base s0 | answerable | 2272 | 2242 | 0.013 | 0.03 | 0.03 | nan | nan | nan | nan | 902 | 0.519 | 0.526 | 0.438 | 0.443 |
| base s1 | unanswerable | 2272 | 2247 | 0.011 | 0.346 | 0.35 | 0.644 | 0.65 | 0.01 | 0.0 | 919 | nan | nan | nan | nan |
| base s1 | answerable | 2272 | 2237 | 0.015 | 0.034 | 0.034 | nan | nan | nan | nan | 898 | 0.522 | 0.53 | 0.43 | 0.436 |
| OPSA s0 | unanswerable | 2272 | 668 | 0.706 | 0.201 | 0.573 | 0.16 | 0.421 | 0.639 | 0.006 | 2290 | nan | nan | nan | nan |
| OPSA s0 | answerable | 2272 | 952 | 0.581 | 0.013 | 0.025 | nan | nan | nan | nan | 1941 | 0.412 | 0.851 | 0.074 | 0.124 |

## Premise audit: share of each token kind inside OPSA's punished set (base rate 0.20)

| tokens | n (base s0) | share in OPSA's set [95% CI] (base s0) | median p(token) (base s0) | share p<0.5 (base s0) | n (base s1) | share in OPSA's set [95% CI] (base s1) | median p(token) (base s1) | share p<0.5 (base s1) | n (OPSA s0) | share in OPSA's set [95% CI] (OPSA s0) | median p(token) (OPSA s0) | share p<0.5 (OPSA s0) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| refusal: first content token of the boxed refusal | 793 | 0.021 [0.011, 0.034] | 1.0 | 0.011 | 787 | 0.017 [0.009, 0.026] | 1.0 | 0.006 | 457 | 0.009 [0.000, 0.024] | 1.0 | 0.0 |
| refusal: every token of the boxed refusal | 6939 | 0.084 [0.077, 0.092] | 1.0 | 0.04 | 6876 | 0.079 [0.072, 0.086] | 1.0 | 0.038 | 2256 | 0.019 [0.012, 0.027] | 1.0 | 0.0 |
| refusal: brace token that opens the box | 793 | 0.586 [0.546, 0.627] | 0.703 | 0.262 | 787 | 0.574 [0.536, 0.610] | 0.707 | 0.24 | 457 | 0.018 [0.002, 0.037] | 1.0 | 0.0 |
| hedge phrase in the reasoning of refusing responses: first token | 2817 | 0.583 [0.554, 0.613] | 0.636 | 0.41 | 2827 | 0.586 [0.559, 0.614] | 0.629 | 0.406 | 14075 | 0.337 [0.323, 0.352] | 1.0 | 0.053 |
| hedge phrase in the reasoning of refusing responses: every token | 7612 | 0.520 [0.502, 0.539] | 0.735 | 0.317 | 7551 | 0.510 [0.493, 0.530] | 0.748 | 0.311 | 37175 | 0.279 [0.271, 0.287] | 1.0 | 0.047 |
| hedge phrase in the reasoning of false-answer responses: first token | 747 | 0.656 [0.606, 0.705] | 0.505 | 0.498 | 758 | 0.633 [0.581, 0.687] | 0.545 | 0.471 | 4254 | 0.471 [0.449, 0.495] | 0.998 | 0.084 |
| false answer: first content token of the box (unanswerable) | 1459 | 0.031 [0.019, 0.043] | 1.0 | 0.018 | 1463 | 0.040 [0.027, 0.056] | 1.0 | 0.023 | 364 | 0.030 [0.010, 0.057] | 1.0 | 0.008 |
| correct answer: first content token of the box (originals) | 1180 | 0.007 [0.002, 0.012] | 1.0 | 0.003 | 1185 | 0.007 [0.002, 0.012] | 1.0 | 0.002 | 936 | 0.006 [0.001, 0.013] | 1.0 | 0.0 |

## Decision point: the brace token after the last \boxed

| responses | n (base s0) | P('{') (base s0) | P('I' \| '{') (base s0) | P(digit \| '{') (base s0) | P(refusal path '{' 'I'), mean (base s0) | median (base s0) | share > 0.5 (base s0) | P(worded path '{\\' 'text'), mean (base s0) | entropy at the brace (base s0) | n (base s1) | P('{') (base s1) | P('I' \| '{') (base s1) | P(digit \| '{') (base s1) | P(refusal path '{' 'I'), mean (base s1) | median (base s1) | share > 0.5 (base s1) | P(worded path '{\\' 'text'), mean (base s1) | entropy at the brace (base s1) | n (OPSA s0) | P('{') (OPSA s0) | P('I' \| '{') (OPSA s0) | P(digit \| '{') (OPSA s0) | P(refusal path '{' 'I'), mean (OPSA s0) | median (OPSA s0) | share > 0.5 (OPSA s0) | P(worded path '{\\' 'text'), mean (OPSA s0) | entropy at the brace (OPSA s0) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| unanswerable, refused | 793 | 0.528 | 0.9845 | 0.009 | 0.5142 | 0.5247 | 0.527 | 0.468 | 0.5 | 787 | 0.538 | 0.9874 | 0.005 | 0.5272 | 0.5465 | 0.553 | 0.4572 | 0.492 | 457 | 0.998 | 0.9995 | 0.0 | 0.9976 | 1.0 | 0.998 | 0.0012 | 0.004 |
| unanswerable, false answer | 1459 | 0.722 | 0.0786 | 0.677 | 0.0101 | 0.0 | 0.003 | 0.0549 | 0.055 | 1463 | 0.733 | 0.074 | 0.686 | 0.0117 | 0.0 | 0.005 | 0.0478 | 0.06 | 364 | 0.639 | 0.0107 | 0.572 | 0.0001 | 0.0 | 0.0 | 0.0026 | 0.007 |
| original, correct | 1180 | 0.767 | 0.0145 | 0.764 | 0.0007 | 0.0 | 0.0 | 0.0008 | 0.008 | 1185 | 0.775 | 0.0116 | 0.777 | 0.0009 | 0.0 | 0.0 | 0.0003 | 0.011 | 936 | 0.758 | 0.0 | 0.737 | 0.0 | 0.0 | 0.0 | 0.0 | 0.001 |
| original, wrong | 995 | 0.761 | 0.0214 | 0.766 | 0.0033 | 0.0 | 0.003 | 0.008 | 0.033 | 976 | 0.763 | 0.0247 | 0.755 | 0.0037 | 0.0 | 0.003 | 0.007 | 0.042 | 167 | 0.557 | 0.0061 | 0.474 | 0.002 | 0.0 | 0.0 | 0.0 | 0.01 |
| original, refused | 68 | 0.637 | 0.9874 | 0.009 | 0.6293 | 0.7018 | 0.676 | 0.3528 | 0.474 | 77 | 0.606 | 0.9954 | 0.003 | 0.6026 | 0.6372 | 0.714 | 0.3911 | 0.518 | 30 | 1.0 | 1.0 | 0.0 | 1.0 | 1.0 | 1.0 | 0.0 | 0.0 |

## base s0: first content token of boxed refusals

| glue\|first content token of boxed refusals | count |
|---|---|
| \|I | 422 |
| \\|text | 370 |
|  \\|text | 1 |

## base s0: where OPSA's punished tokens sit in refusing responses

| token type | share of OPSA's tokens in refusing responses |
|---|---|
| word | 0.661 |
| math/LaTeX | 0.189 |
| punctuation/other | 0.102 |
| doubt/fork word | 0.025 |
| space/newline | 0.017 |
| digit | 0.006 |
| end token | 0.0 |

OPSA tokens in refusing responses: 137,246; before the first hedge phrase 57,481, at or after it 65,347, in the 69 refusing responses with no detected hedge 14,418.

## base s0: worded false answers on unanswerable items (114 of them; the most common box contents)

| boxed content | count |
|---|---|
| \text{Undefined} | 7 |
| \text{No solution} | 5 |
| 60 \text{ to } 66 | 4 |
| \text{Infinite} | 4 |
| 78 \text{ to } 85 | 3 |
| \text{Even number of bricks} | 3 |
| k \leq -7 \text{ or } k \geq 9 | 3 |
| \text{Odd number of bricks} | 2 |
| \text{Infinitely many} | 2 |
| \text{No maximum} | 2 |
| \text{Unbounded} | 2 |
| \text{No solution exists with } x \ne y | 2 |

## base s1: first content token of boxed refusals

| glue\|first content token of boxed refusals | count |
|---|---|
| \|I | 431 |
| \\|text | 355 |
|  \\|text | 1 |

## base s1: where OPSA's punished tokens sit in refusing responses

| token type | share of OPSA's tokens in refusing responses |
|---|---|
| word | 0.656 |
| math/LaTeX | 0.188 |
| punctuation/other | 0.104 |
| doubt/fork word | 0.025 |
| space/newline | 0.019 |
| digit | 0.009 |
| end token | 0.0 |

OPSA tokens in refusing responses: 137,863; before the first hedge phrase 60,089, at or after it 63,656, in the 50 refusing responses with no detected hedge 14,118.

## base s1: worded false answers on unanswerable items (109 of them; the most common box contents)

| boxed content | count |
|---|---|
| \text{Infinite} | 8 |
| \text{No solution} | 5 |
| \text{Infinitely many} | 3 |
| \text{The sum of every integer that ends in 1 is infinite.} | 3 |
| \text{Undefined} | 3 |
| 60 \text{ to } 66 | 3 |
| k \leq -7 \text{ or } k \geq 9 | 3 |
| \text{It depends on } x \text{ and } y | 2 |
| 0.94 \text{ meters} | 2 |
| 0.151 \text{ meters} | 2 |
| \text{Even number of bricks} | 2 |
| \text{The sum is infinite.} | 2 |

## OPSA s0: first content token of boxed refusals

| glue\|first content token of boxed refusals | count |
|---|---|
| \|I | 457 |

## OPSA s0: where OPSA's punished tokens sit in refusing responses

| token type | share of OPSA's tokens in refusing responses |
|---|---|
| word | 0.752 |
| doubt/fork word | 0.111 |
| punctuation/other | 0.09 |
| math/LaTeX | 0.028 |
| digit | 0.01 |
| space/newline | 0.009 |
| end token | 0.0 |

OPSA tokens in refusing responses: 286,499; before the first hedge phrase 19,411, at or after it 264,324, in the 7 refusing responses with no detected hedge 2,764.

## OPSA s0: worded false answers on unanswerable items (8 of them; the most common box contents)

| boxed content | count |
|---|---|
| k \leq -7 \text{ or } k \geq 9 | 3 |
| 60 \text{ and } 66 | 1 |
| 60 \text{ to } 85 | 1 |
| \text{infinite} | 1 |
| a = 2b - c \text{ and } b^2 = 4ac | 1 |
| k ≤ -7 \text{ or } k ≥ 9 | 1 |

