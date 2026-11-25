# Experiments 2 and 7: the released OPSA checkpoint against its base on MATH-500 (2026-11-24 run)

The supporting study of the proposal: does training improve immediate success, success with retries, or both,
and where do regressions occur? Base `Qwen/Qwen3-1.7B` against the released `Tuwhy/Qwen3-1.7B-OPSA`, on 100
MATH-500 problems of levels 3 to 5 (the seeded subset whose first 60 are the token-replay pilot's problems). Every
number below comes from `reports/2026-11-24_passk_tables/`; the raw rollouts stay in `runs/2026-11-24_passk/`.

## Summary

**OPSA's checkpoint is better at both first tries and retries, so its gain is not pure sharpening.** Pass@1 rises
from 0.676 to 0.816 (+0.140 [+0.094, +0.193]) and pass@8 from
0.87 to 0.93 (+0.060 [+0.000, +0.120], p = 0.065). The Sharpening Tax is not positive:
Tax(8) = -0.053 [-0.169, +0.068], with the checkpoint's scalability score S(8) = 0.24 against the base's
0.187, so the trained model relies on retries at least as much as the base. A frozen-model control, two base batches
against each other, moves pass@1 by -0.006 [-0.032, +0.021] and pass@8 by -0.004 [-0.039, +0.028]: that is the noise floor.

**Where the gain sits.** Problems are grouped by what the first base batch (16 attempts) did with them.

| Group (base batch A, 16 attempts) | problems | Δ pass@1 | Δ pass@8 |
|---|---|---|---|
| all 16 solved | 33 | +0.008 [-0.011, +0.034] | +0.000 [+0.000, +0.000] |
| some solved | 56 | +0.210 [+0.134, +0.286] | +0.018 [-0.054, +0.089] |
| none solved | 11 | +0.182 [+0.034, +0.364] | +0.455 [+0.182, +0.727] |

In the middle group the checkpoint solves the same problems more reliably but not more of them: a 21-point gain in
pass@1 with pass@8 unchanged, which is the consistency signature of sharpening. In the never-solved group it also
solves problems the base did not: pass@8 goes from 0.091 to 0.545 on 11 problems.

**Coverage audit.** 5 problems were solved by the checkpoint and by neither base batch (0 of 24 attempts). With
64 extra base samples each, 2 of them turned out base-solvable and are not new. 3 remain unsolved by the base after 88
attempts, and their derivations hold up on reading: the octagon's area by placing the vertices and applying the
shoelace formula (`geometry/561`, gold 7/2, solved 5 of 8 times); p(8) through q(x) = (x² − 1)p(x) − x, a degree-7
polynomial with known roots (`intermediate_algebra/1197`, gold 3/56, 2 of 8); and the period-10 recurrence
(`intermediate_algebra/90`, gold 898, 7 of 8). Against this, the two base batches "discover" 1 problem of each
other's 100, and the checkpoint "lost" 2 problems the base had solved against a noise baseline of 3. So about 3 of 100
problems became solvable, with the plan's caveat that 88 base attempts is not the 1,024 it asks for before calling a
capability new.

**The dominant wrong answer nearly vanishes.** On the 67 problems where the base has a most-common wrong answer, that
answer carries 0.127 of the base's attempts and 0.007 of the checkpoint's (-0.119 [-0.164, -0.078]). Distinct wrong answers
per problem fall from 2.75 to 0.19, but unfinished responses rise from 0.03 to 0.239: the checkpoint's failures
turned from confident wrong answers into answers that never finish.

**Sampling alone does not do this.** The base's pass@1 stays between 0.662 and 0.679 across T = 1.0, 0.8, 0.6 and 0.4; no
temperature approaches the checkpoint's 0.816. The closest, T = 0.4 (chosen on half the problems and compared on the
other half), trails it by +0.130 [+0.060, +0.205] in pass@1 and +0.100 [+0.000, +0.200] in pass@8. The cold sampler shows the sharpening signature the
checkpoint lacks: its pass@8 drops to 0.84 and its S(8) to 0.16.

**The price is length.** The checkpoint's answers average 7541 tokens against 1026, and 0.177 of them are
still unfinished at 16,384 tokens. At a matched 4,096-token budget its pass@1 is 0.344 against the base's 0.676;
it needs 8,192 tokens to reach 0.623 and the full 16,384 for 0.816, while the base gains nothing past 2,048. Per
thousand tokens it carries 14.64 reflection markers ("wait", "but", "let me check") against the base's 1.82, and
9.97 against 1.24 on the strict count of sentence-initial ones. Some 20-token span repeats three or more times
in 0.76 of its responses (base 0.259), but almost none are degenerate loops: repeats of one span cover a fifth of the
response in 0.001 of them.

**Reading, by the plan's rules.** The gain is both kinds: consistency on problems the base sometimes solved, and
coverage on a few it never did, with no positive Sharpening Tax. It is bought with seven times the tokens and eight
times the self-checking per token, which fits the OPSA paper's own mechanism claim that punishing unlikely tokens
at high-entropy forks makes "wait" and "but" more frequent, and it fits the two earlier results here: OPSA's rule
punishes the hedging tokens where a refusal starts, yet the trained model kept its refusals and became far more
deliberate. What punishment-only training added, on this evidence, is deliberation length rather than a sharpened
first guess.

## What ran

| Arm | Model | T | Samples | Cap (tokens) | Seed | Generation |
|---|---|---|---|---|---|---|
| base A (grouping) | Qwen3-1.7B | 1.0 | 16 | 4,096 | 10 | 11 min |
| base B (scoring, frozen-model control) | Qwen3-1.7B | 1.0 | 8 | 4,096 | 11 | 6 min |
| base sweep | Qwen3-1.7B | 0.8 / 0.6 / 0.4 | 8 each | 4,096 | 12 to 14 | 18 min |
| trained | Qwen3-1.7B-OPSA | 1.0 | 8 | 16,384 | 15 | 277 min |
| coverage audit | Qwen3-1.7B on 5 problems | 1.0 | 64 | 4,096 | 20 | 9 min |

All arms on the same 100 problems (problem seed 0), thinking off, no top-p or top-k, graded by math-verify on the
last boxed answer. pass@k is the unbiased per-problem estimator averaged over problems; S(K) and the Tax follow the
Sharpening Tax paper with K = 8; intervals are paired bootstraps over problems. The driver is `run_passk.py`, the
analysis `src/passk.py`.

## Caveats

- **Eight samples, not 32.** S(8) is not the paper's S(32), though the plan notes the tax at 8 predicts the tax at
  32. Pass@8 with 8 samples is the any-success indicator and has the most variance.
- **Eleven problems in the never-solved group.** The coverage result rests on them and on three audited problems.
- **Budget.** The checkpoint was run at 16,384 tokens and the base at 4,096. The base never uses more than about
  2,000 (its pass@1 is flat past that budget), so the comparison does not short-change it, but the checkpoint's gain
  cannot be separated from its length without a trained model that writes short answers.
- **The repetition flag** is a proxy; the heavy-repetition rate shows the long answers are not loops.
- **One model size, one checkpoint**, whose true base (hybrid or `-Base`) the plan says to confirm by weight distance.

## Next

The two main studies still open are experiment 3, the hidden-rule study, which is the only one that can show
punishment teaching something new under controlled training, and a 32-sample rerun of this analysis if the exact
Sharpening Tax matters for the write-up. Both are runnable on this machine; experiment 3 needs the training loop.

# Tables

## Per arm and group (groups from base batch A: all / some / none of 16 attempts solved)

| arm | group (by base batch A) | problems | pass@1 | pass@8 | S(8) | mean tokens | at cap | repetition (a 20-gram x3) | heavy repetition (>=20% of tokens) | reflection markers / 1k tokens | strict reflection / 1k tokens |
|---|---|---|---|---|---|---|---|---|---|---|---|
| base A (grouping) | all problems | 100 | 0.67 | 0.866 | 0.196 | 1033 | 0.021 | 0.261 | 0.0 | 1.65 | 1.15 |
| base A (grouping) | all solved | 33 | 1.0 | 1.0 |  | 439 | 0.0 | 0.057 | 0.0 | 0.56 | 0.24 |
| base A (grouping) | some solved | 56 | 0.607 | 0.958 | 0.294 | 1126 | 0.009 | 0.283 | 0.0 | 1.38 | 0.96 |
| base A (grouping) | none solved | 11 | 0.0 | 0.0 | 0.0 | 2346 | 0.142 | 0.756 | 0.0 | 2.94 | 2.12 |
| base B (scoring) | all problems | 100 | 0.676 | 0.87 | 0.187 | 1026 | 0.022 | 0.259 | 0.001 | 1.82 | 1.24 |
| base B (scoring) | all solved | 33 | 0.989 | 1.0 | 0.19 | 446 | 0.0 | 0.045 | 0.0 | 0.68 | 0.29 |
| base B (scoring) | some solved | 56 | 0.623 | 0.946 | 0.26 | 1133 | 0.013 | 0.288 | 0.002 | 1.49 | 1.03 |
| base B (scoring) | none solved | 11 | 0.011 | 0.091 | 0.046 | 2220 | 0.136 | 0.75 | 0.0 | 3.37 | 2.38 |
| trained | all problems | 100 | 0.816 | 0.93 | 0.24 | 7541 | 0.177 | 0.76 | 0.001 | 14.64 | 9.97 |
| trained | all solved | 33 | 0.996 | 1.0 | 0.143 | 2880 | 0.0 | 0.451 | 0.0 | 13.93 | 9.1 |
| trained | some solved | 56 | 0.833 | 0.964 | 0.268 | 8719 | 0.161 | 0.897 | 0.002 | 14.99 | 10.27 |
| trained | none solved | 11 | 0.193 | 0.545 | 0.211 | 15529 | 0.795 | 0.989 | 0.0 | 14.02 | 9.58 |
| base T=0.8 | all problems | 100 | 0.662 | 0.87 | 0.197 | 1041 | 0.022 | 0.287 | 0.003 | 1.57 | 1.1 |
| base T=0.8 | all solved | 33 | 0.996 | 1.0 | 0.143 | 444 | 0.0 | 0.064 | 0.0 | 0.7 | 0.38 |
| base T=0.8 | some solved | 56 | 0.592 | 0.946 | 0.277 | 1114 | 0.009 | 0.308 | 0.002 | 1.27 | 0.9 |
| base T=0.8 | none solved | 11 | 0.023 | 0.091 | 0.027 | 2457 | 0.159 | 0.852 | 0.011 | 2.72 | 1.96 |
| base T=0.6 | all problems | 100 | 0.674 | 0.88 | 0.223 | 1033 | 0.024 | 0.289 | 0.005 | 1.45 | 1.02 |
| base T=0.6 | all solved | 33 | 0.985 | 1.0 | 0.153 | 456 | 0.0 | 0.053 | 0.004 | 0.7 | 0.37 |
| base T=0.6 | some solved | 56 | 0.621 | 0.964 | 0.315 | 1135 | 0.016 | 0.326 | 0.007 | 1.3 | 0.92 |
| base T=0.6 | none solved | 11 | 0.011 | 0.091 | 0.046 | 2242 | 0.136 | 0.807 | 0.0 | 2.29 | 1.67 |
| base T=0.4 | all problems | 100 | 0.679 | 0.84 | 0.16 | 1045 | 0.037 | 0.279 | 0.009 | 1.35 | 0.98 |
| base T=0.4 | all solved | 33 | 0.992 | 1.0 | 0.143 | 454 | 0.0 | 0.049 | 0.0 | 0.48 | 0.18 |
| base T=0.4 | some solved | 56 | 0.621 | 0.875 | 0.204 | 1102 | 0.029 | 0.312 | 0.016 | 1.1 | 0.79 |
| base T=0.4 | none solved | 11 | 0.034 | 0.182 | 0.074 | 2527 | 0.193 | 0.795 | 0.0 | 2.37 | 1.85 |

## pass@k curves

| arm | pass@1 | pass@2 | pass@3 | pass@4 | pass@5 | pass@6 | pass@7 | pass@8 |
|---|---|---|---|---|---|---|---|---|
| base A (grouping) | 0.67 | 0.762 | 0.802 | 0.825 | 0.841 | 0.852 | 0.86 | 0.866 |
| base B (scoring) | 0.676 | 0.774 | 0.814 | 0.835 | 0.847 | 0.856 | 0.864 | 0.87 |
| trained | 0.816 | 0.865 | 0.883 | 0.895 | 0.905 | 0.914 | 0.922 | 0.93 |
| base T=0.8 | 0.662 | 0.758 | 0.804 | 0.831 | 0.847 | 0.858 | 0.865 | 0.87 |
| base T=0.6 | 0.674 | 0.762 | 0.803 | 0.829 | 0.848 | 0.862 | 0.873 | 0.88 |
| base T=0.4 | 0.679 | 0.757 | 0.791 | 0.809 | 0.82 | 0.829 | 0.835 | 0.84 |

## Matched budgets: a response counts only if it finished within B tokens

| arm | budget (tokens) | pass@1 | pass@8 |
|---|---|---|---|
| base A (grouping) | 1024 | 0.562 | 0.73 |
| base A (grouping) | 2048 | 0.656 | 0.835 |
| base A (grouping) | 4096 | 0.67 | 0.866 |
| base B (scoring) | 1024 | 0.559 | 0.74 |
| base B (scoring) | 2048 | 0.661 | 0.84 |
| base B (scoring) | 4096 | 0.676 | 0.87 |
| trained | 1024 | 0.056 | 0.21 |
| trained | 2048 | 0.154 | 0.3 |
| trained | 4096 | 0.344 | 0.55 |
| trained | 8192 | 0.623 | 0.75 |
| trained | 16384 | 0.816 | 0.93 |
| base T=0.8 | 1024 | 0.55 | 0.72 |
| base T=0.8 | 2048 | 0.644 | 0.84 |
| base T=0.8 | 4096 | 0.662 | 0.87 |
| base T=0.6 | 1024 | 0.557 | 0.72 |
| base T=0.6 | 2048 | 0.659 | 0.87 |
| base T=0.6 | 4096 | 0.674 | 0.88 |
| base T=0.4 | 1024 | 0.571 | 0.7 |
| base T=0.4 | 2048 | 0.667 | 0.83 |
| base T=0.4 | 4096 | 0.679 | 0.84 |

## Contrasts against base batch B (paired over problems)

| arm | against | group | problems | Δ pass@1 [95% CI] | p (pass@1) | Δ pass@8 [95% CI] | p (pass@8) | Tax(8) = S_ref - S_arm [95% CI] |
|---|---|---|---|---|---|---|---|---|
| base A (grouping) | base B (scoring) | all problems | 100 | -0.006 [-0.032, +0.021] | 0.665 | -0.004 [-0.039, +0.028] | 0.813 | -0.009 [-0.061, +0.049] |
| base A (grouping) | base B (scoring) | all solved | 33 | +0.011 [+0.000, +0.034] | 0.699 | +0.000 [+0.000, +0.000] | 1.0 | n/a |
| base A (grouping) | base B (scoring) | some solved | 56 | -0.016 [-0.059, +0.027] | 0.479 | +0.012 [-0.029, +0.062] | 0.648 | -0.034 [-0.101, +0.027] |
| base A (grouping) | base B (scoring) | none solved | 11 | -0.011 [-0.034, +0.000] | 0.722 | -0.091 [-0.273, +0.000] | 0.722 | +0.046 [+0.000, +0.141] |
| trained | base B (scoring) | all problems | 100 | +0.140 [+0.094, +0.193] | 0.0 | +0.060 [+0.000, +0.120] | 0.065 | -0.053 [-0.169, +0.068] |
| trained | base B (scoring) | all solved | 33 | +0.008 [-0.011, +0.034] | 0.756 | +0.000 [+0.000, +0.000] | 1.0 | +0.048 [+0.048, +0.048] |
| trained | base B (scoring) | some solved | 56 | +0.210 [+0.134, +0.286] | 0.0 | +0.018 [-0.054, +0.089] | 0.828 | -0.008 [-0.172, +0.148] |
| trained | base B (scoring) | none solved | 11 | +0.182 [+0.034, +0.364] | 0.002 | +0.455 [+0.182, +0.727] | 0.002 | -0.165 [-0.305, -0.040] |
| base T=0.8 | base B (scoring) | all problems | 100 | -0.014 [-0.046, +0.018] | 0.407 | +0.000 [-0.050, +0.050] | 1.0 | -0.009 [-0.083, +0.064] |
| base T=0.8 | base B (scoring) | all solved | 33 | +0.008 [+0.000, +0.023] | 0.699 | +0.000 [+0.000, +0.000] | 1.0 | +0.048 [+0.048, +0.048] |
| base T=0.8 | base B (scoring) | some solved | 56 | -0.031 [-0.089, +0.025] | 0.269 | +0.000 [-0.071, +0.071] | 1.0 | -0.017 [-0.109, +0.069] |
| base T=0.8 | base B (scoring) | none solved | 11 | +0.011 [-0.034, +0.068] | 0.929 | +0.000 [-0.273, +0.273] | 1.0 | +0.019 [-0.084, +0.141] |
| base T=0.6 | base B (scoring) | all problems | 100 | -0.003 [-0.033, +0.029] | 0.913 | +0.010 [-0.020, +0.040] | 0.821 | -0.036 [-0.090, +0.022] |
| base T=0.6 | base B (scoring) | all solved | 33 | -0.004 [-0.027, +0.019] | 0.864 | +0.000 [+0.000, +0.000] | 1.0 | +0.037 [+0.031, +0.048] |
| base T=0.6 | base B (scoring) | some solved | 56 | -0.002 [-0.056, +0.049] | 0.939 | +0.018 [-0.036, +0.071] | 0.771 | -0.056 [-0.137, +0.032] |
| base T=0.6 | base B (scoring) | none solved | 11 | +0.000 [+0.000, +0.000] | 1.0 | +0.000 [+0.000, +0.000] | 1.0 | +0.000 [+0.000, +0.000] |
| base T=0.4 | base B (scoring) | all problems | 100 | +0.003 [-0.030, +0.034] | 0.903 | -0.030 [-0.080, +0.020] | 0.318 | +0.027 [-0.047, +0.102] |
| base T=0.4 | base B (scoring) | all solved | 33 | +0.004 [-0.011, +0.023] | 0.904 | +0.000 [+0.000, +0.000] | 1.0 | +0.048 [+0.048, +0.048] |
| base T=0.4 | base B (scoring) | some solved | 56 | -0.002 [-0.058, +0.058] | 0.974 | -0.071 [-0.161, +0.000] | 0.134 | +0.056 [-0.041, +0.150] |
| base T=0.4 | base B (scoring) | none solved | 11 | +0.023 [+0.000, +0.057] | 0.22 | +0.091 [+0.000, +0.273] | 0.702 | -0.028 [-0.141, +0.058] |

## The matched temperature

| choice | temperature | trained pass@1 (tuning half) | base T=0.4 pass@1 (tuning half) |
|---|---|---|---|
| held out: T chosen on odd problems, reported on even | 0.4 | 0.797 | 0.652 |
| in sample: T chosen on all problems | 0.4 | 0.816 | 0.679 |

| arm | against | group | problems | Δ pass@1 [95% CI] | p (pass@1) | Δ pass@8 [95% CI] | p (pass@8) | Tax(8) = S_ref - S_arm [95% CI] | problems_used |
|---|---|---|---|---|---|---|---|---|---|
| trained | base T=0.4 | all problems | 50 | +0.130 [+0.060, +0.205] | 0.0 | +0.100 [+0.000, +0.200] | 0.066 | -0.139 [-0.314, +0.044] | even (held out) |
| trained | base T=0.4 | all solved | 18 | +0.000 [-0.021, +0.021] | 1.0 | +0.000 [+0.000, +0.000] | 1.0 | +0.000 [+0.000, +0.000] | even (held out) |
| trained | base T=0.4 | some solved | 27 | +0.185 [+0.092, +0.296] | 0.0 | +0.074 [-0.074, +0.222] | 0.437 | -0.072 [-0.296, +0.176] | even (held out) |
| trained | base T=0.4 | none solved | 5 | +0.300 [-0.025, +0.650] | 0.082 | +0.600 [+0.200, +1.000] | 0.017 | -0.275 [-0.474, -0.041] | even (held out) |
| trained | base T=0.4 | all problems | 100 | +0.138 [+0.091, +0.190] | 0.0 | +0.090 [+0.030, +0.160] | 0.008 | -0.079 [-0.198, +0.045] | all (in sample) |
| trained | base T=0.4 | all solved | 33 | +0.004 [-0.008, +0.019] | 0.798 | +0.000 [+0.000, +0.000] | 1.0 | +0.000 [+0.000, +0.000] | all (in sample) |
| trained | base T=0.4 | some solved | 56 | +0.212 [+0.138, +0.290] | 0.0 | +0.089 [+0.000, +0.179] | 0.058 | -0.064 [-0.213, +0.083] | all (in sample) |
| trained | base T=0.4 | none solved | 11 | +0.159 [+0.000, +0.352] | 0.073 | +0.364 [+0.000, +0.727] | 0.107 | -0.137 [-0.309, +0.063] | all (in sample) |

## Per-problem transfer: newly solved, lost, dominant wrong answer, dispersion

| measure | value |
|---|---|
| problems with a dominant wrong answer in batch A | 67 |
| mass on it, base B | 0.127 |
| mass on it, trained | 0.007 |
| Δ mass [95% CI] | -0.119 [-0.164, -0.078] |
| p | 0.0 |
| distinct wrong answers per problem, base B | 2.75 |
| distinct wrong answers per problem, trained | 0.19 |
| no-answer share, base B | 0.03 |
| no-answer share, trained | 0.239 |
| problems | 100 |
| apparently newly solved (audit these) | 5 |
| newly solved ids | test/geometry/561.json,test/intermediate_algebra/1197.json,test/intermediate_algebra/90.json,test/precalculus/1146.json,test/precalculus/1201.json |
| noise baseline: solved by base B, never by base A | 1 |
| lost (base solved, trained never) | 2 |
| lost ids | test/counting_and_probability/525.json,test/number_theory/820.json |
| noise baseline: solved by base A, never by base B | 3 |

## Coverage audit

| problem | extra_base_samples | base_solved_in_extra | verdict |
|---|---|---|---|
| test/geometry/561.json | 64 | 0 | still never solved by the base: a candidate new solve, review the derivation by hand |
| test/intermediate_algebra/1197.json | 64 | 0 | still never solved by the base: a candidate new solve, review the derivation by hand |
| test/intermediate_algebra/90.json | 64 | 0 | still never solved by the base: a candidate new solve, review the derivation by hand |
| test/precalculus/1146.json | 64 | 1 | the base solves it too: not new |
| test/precalculus/1201.json | 64 | 2 | the base solves it too: not new |

