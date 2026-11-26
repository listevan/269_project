# Project handoff: negative-only training (OPSA) study, CS 269, as of 2026-10-16

## Owner, deliverable, constraints
- Owner: Steven (UCLA MSCS, Fall 2026). Class project for CS 269, graded on novelty. Solo ("it is I, not team").
- Proposal doc (Claude Doc, two tabs): https://claude.ai/code/artifact/6345c291-d6b0-4017-a4d2-8008e408107b
  - Tab "Plan": tab id 4f8e13b4-928d, body node f907aa9a-cddc (rev 117). Long working plan: motivation, RQs,
    experiments-at-a-glance table, detailed experiment sections, possible-results table, datasets, 14 measurement
    hygiene checks, compute, deferred experiment 6, related work and novelty, 11 sources. Uses OLD experiment
    numbers (see mapping below). May still hold stale rows from a dropped verification/tool study; check before reuse.
  - Tab "Abstract": tab id 85750be1-bd6c, body node d0400f8b-564b (rev 400). Submission text. Steven edits it
    himself; never overwrite his edits. First person singular, planned voice ("I will..."), about 450 words.
    Current H1: "Can Punishment Alone Teach a Model Something New, and What Does It Take Away?"
    (doc name still "Final Project: When Does Negative-Only Training Grow Rare Correct Behaviors, and What Does It Quietly Remove?").
- Code repo: ~/projects/269_project, GitHub listevan/269_project (PRIVATE), branch main, first commit 62be5cb.
- Drive run folder: My Drive/UCLA (Jun 2026 onward)/Classes/CS 269/269_project/runs/ (Drive for desktop mirror at
  ~/Library/CloudStorage/GoogleDrive-listevan101@gmail.com).
- Writing rules: no em dashes anywhere (chat, docs, commits). Plain wording; avoid AI-sounding phrasing ("By
  construction...", semicolon splices, short rhetorical sentences). Concise abstract-style prose.
- Working rules: models under 7B. Small gated pilots; never scale before a gate passes; a smoke run checks wiring only.
- Hardware: Colab (A100 40GB used in his other project); own Radeon RX 9070 XT 16 GB (Linux ROCm 7.2 works; vLLM has
  an RDNA4 build). OPSA's code is SGLang + Megatron (slime), so training arms must be reimplemented in HF/PEFT.

## Core idea
Punishment-only post-training methods (OPSA, negative sample reinforcement) improve reasoning, but a negative update
on a token moves its probability to tokens the model already favors, so they should mainly sharpen existing habits.
The behaviors post-training wants are usually rare ones: following an unfamiliar rule, dropping a confident wrong
answer, admitting a problem cannot be solved. Negative updates also shift probability on tokens never targeted
directly (softmax coupling, shared weights). The project asks whether these indirect effects can teach a new
behavior, and which rare good behaviors they erase.

- RQ1: Under what conditions can teacher-free, negative-only training increase initially rare correct responses,
  and which initially rare good behaviors does it suppress along the way? ("Initially rare", not "absent".)
- RQ2: What are the roles of token selection, teacher information, positive feedback, and exposure to useful actions?

## Methods compared (all from the Abstract)
| Method | What it pushes down | Information used |
|---|---|---|
| OPSA | the model's least likely tokens | none (no teacher, reward, or reference answer) |
| Random baseline | same number of tokens chosen at random | none; tests whether choosing unlikely tokens matters |
| NSR (negative sample reinforcement) | every token of failed answers | answer key |
| Teacher-guided negative | tokens the teacher finds unlikely | a teacher |
| On-policy distillation (OPD) | teacher both signs (rewards and penalties) | a teacher; shows what positive feedback adds |
| SFT on worked demonstrations | (positive only) | demonstrations; run first to show each task is learnable |
| Top-k / top-p decoding of base | (no training) | tests whether OPSA does more than sharpen sampling |
Plan tab also has: "tail-token suppression on failed responses" (answer-key arm, secondary), layer-2 matched
contrasts, an action-exposure diagnostic, and a suggested plain GRPO reference arm in study 1's near-zero operators.

## The three studies (Abstract numbering) and old Plan numbering
| Abstract study | Plan tab name | Needs training? |
|---|---|---|
| 1. Learning a hidden rule | Experiment 3 (main) | Yes, every arm is a training run. Only the base-rate and teacher checks need none |
| 2. Side effects on untargeted rare behaviors | Experiment 5 (main) | First pass no (released OPSA checkpoints + premise audit); second pass yes (study 1's trained models) |
| 3. Usefulness of penalized tokens (token replay) | Experiment 1 (explanatory) | No. Pilot is coded (below) |
| (framework) | Experiment 4: comparison arms, matched contrasts | defines the arms |
| (supporting) | Experiments 2 + 7: per-problem and pass@k analysis of released checkpoints | No for released ckpts |
| (deferred) | former Experiment 6: representation-direction / extrapolation study | not planned |

### Study 1: hidden rule (Plan exp 3)
- A novel operation the model has not seen, e.g. a # b = 2a + b. Definition left OUT of training prompts. Teacher =
  same model WITH the definition in its prompt (privileged-context, OPSD/SDFT/OPCD family). Each method updates the
  model on its own answers.
- About 12 operators spanning base correct rates 0 to 50% (rarity dial).
- Measure before/after: share correct, share dominant wrong answer (e.g. reading # as +), share other, plus the
  correct-answer token probability from logits.
- Transfer on new numbers without the definition (primary), also ranges, compositions, with-definition sanity check.
- OPSA has no signal about which answer is right here by design, so the Abstract compares NSR and the two
  teacher-guided methods (OPSA expected to only sharpen the dominant guess).
- Demonstration (SFT) control; teacher-signal decile audit before training and masking of length-capped responses
  (both from Cui et al.). Compute: LoRA on Qwen3-1.7B, 512-token outputs, fits free Colab or the 16 GB card.
- GRPO dead zone: operators near 0% give all-wrong groups, zero advantage; plain GRPO there is a reference arm.

### Study 2: side effects (Plan exp 5)
- Does punishment remove refusals on unanswerable math? Data: SUM (from "The Hallucination Tax of Reinforcement
  Finetuning", arXiv 2505.13988; RFT cut Qwen2.5-7B-Instruct refusal 0.30 to 0.08) and their solvable originals.
- Mechanism claim: pushing down unlikely tokens peaks every next-token distribution; doubt tokens ("probably",
  "this cannot be determined") are unlikely tokens competing with just stating a number.
- Premise audit first: do refusal tokens in the base model's responses fall inside OPSA's penalized set?
- Main measures: refusal rate and false-answer rate on unanswerable problems, accuracy on solvable ones. Secondary:
  doubt-when-wrong (word list), verbalized-confidence calibration (Lin 2022, Tian 2023, Xiong 2024, Damani 2025,
  cited from memory, re-check), refusal-token probability from logits. Also track length, truncation, repetition.
- Controls: frozen base re-evaluated with new seeds (noise), top-k/top-p decoding, temperature sweep, fixed budgets,
  thinking on/off, unlabeled unanswerable prompts in training.
- First pass: released OPSA checkpoints vs base. Second pass: study 1's trained arms.

### Study 3: token usefulness by replay (Plan exp 1). This is the coded pilot.
- On MATH-500 (and MBPP+ later), take the tokens each method penalizes, continue the response with and without each
  token, compare how often continuations reach a correct answer. Share of useful tokens in a method's penalized set
  = how much good behavior it pushes down. Plan also lists: decision-level replay, all methods' tokens, penalty
  strength, replay noise floor, low-success stratum, local update probes (one small update per rule on shared
  frozen rollouts, then evaluate) as the bridge from replay to training.

### Supporting (Plan exps 2 + 7)
- Released checkpoints on MATH-500 and MBPP+; three batches per problem (grouping, scoring = frozen control,
  trained); probability of correct and of the dominant wrong answer; full pass@k; Sharpening Tax as secondary;
  coverage audit (extra base samples + manual review); temperature sweep; fixed budgets. pass@32 up does not imply
  negative tax.

## OPSA facts
- Paper: Ding and Zhang, arXiv 2608.31046 v2. Code: github.com/DripNowhy/On-Policy-Self-Adaptation (repo title "Does
  On-Policy Distillation Really Distill? From Noisy Teacher to Self-Improvement"), files
  slime/backends/megatron_utils/opsa.py, slime/examples/opsa/run_opsa.sh, models/qwen3-1.7b.sh, docs/en/advanced/opsa.md.
- Selector (read in code 2026-10-16): DP-local. Each data-parallel rank concatenates the valid response tokens of its
  packed batch and takes K = max(1, floor(f N)) tokens by ascending recomputed actor log-prob (stable argsort).
  f = 0.2. No gathering across ranks. No EOS mask, no think mask, no entropy threshold, no random branch.
- Advantage: entropy min-max ranked among selected tokens, r = (H - Hmin)/(Hmax - Hmin), A = A_max + (A_min - A_max) r,
  A_min = -1.0, A_max = -0.5 (most uncertain gets -1.0); all equal entropies gives -1.0. Unselected tokens get zero
  advantage and are excluded from the loss mask and denominator. Reference-free: no reference forward, no KL, zero task
  reward. Presets: opsa, fixed-negative (-0.5), fixed-positive (+0.2); sweep 10/20/30/40%.
- Qwen3-1.7B preset: 700 rollout steps, 4 actor + 4 rollout GPUs, rollout batch 64 prompts, 1 sample per prompt,
  global batch 64 (so 16 responses per actor rank), rollout T = 1, max response 12,000 tokens, eval max 32,768,
  thinking disabled by default, eval T 0.7 / top-p 0.8 / top-k 20, 4 samples per eval prompt.
- Results: Qwen3-1.7B AIME24 Avg@32 13.44 to 48.85 (OPD 32.08, GRPO 33.96, NSR 32.08). Answers grow about 4,457 to
  23,205 tokens. Ablations: top-logp only, fixed sign, entropy sign, fraction 10 to 40%, masking "wait/but" forks,
  NSR, token budget, min-p/temperature grid (figure only). No replay, per-problem, or random-mask analysis.
- Released checkpoints: Tuwhy/Qwen3-1.7B-OPSA, Tuwhy/Qwen3-4B-OPSA (public, Apache-2.0; HF card tags a -Base model
  though the paper says hybrid Qwen3). JustRL-Qwen3-1.7B/4B RL checkpoints exist for comparison.

## Related work and novelty (checked 2026-10-01, updated through 2026-10-13)
- Replay method and training-signal audits exist: Forking Paths (2412.07961), Cliff Tokens (2606.25524, released
  per-token rollouts for 7 models on MATH-500 and AIME25, possible reuse), Cancellation Hypothesis (2605.08666, GRPO
  tokens, value gaps in thousandths), Unmasking On-Policy Distillation (2605.10889), Extremely Sparse Supervision
  (2609.04565: tokens a teacher punishes hardest are mostly correct tokens it dislikes). Replay noise: Re-feeding Is
  Not Replaying (2606.15621), identical runs disagree on 9 to 23% of estimates. Where Hindsight Credit Can Reside
  (2604.11056) splits reward updates by sign and entropy.
- Per-problem sharpening shown for TTRL (2605.19444), NSR (2512.21625), general post-training (Sharpening Tax,
  Oh et al., 2610.01509: WebShop gemma-4-31B solved-with-retries 87.6 to 30.0%, never-solved 12.4 to 44.0%; tax at 8
  predicts 32, rho 0.85; names on-policy distillation as untested) and teacher OPD (Cui et al., 2610.03185: teacher
  = implicit reward; OPD pass@1 gains vanish by k = 256; no coverage expansion after a 1,024-sample audit, 73 to
  65/63 solvable; collapse = reward hacking, repetition 4.2 to 38%, truncation 4.1 to 99.4%; fixes: mask capped
  responses, SFT warm-up; teacher-advantage deciles show misalignment before training).
- All-negative GRPO groups get negative pushes: RL-ZVP 2509.21880, NGRPO 2509.18851, LENS 2510.08696, EGPO 2602.22751,
  AVSPO 2605.21125, teacher OPD on failed groups: Distill Where You Fail 2608.00782; DAPO drops such groups. All use
  the answer key and report aggregate accuracy (abstracts only read). Anchors the answer-key arm.
- RIDE (2609.36484): RL-teacher hidden-state residual extrapolation, never mentions OPSA (relevant only to deferred exp 6).
- Open slot claimed: replay usefulness plus per-problem analysis for punishment-only training with a random-selection
  control and the four-rule family; hidden-rule rarity dial; side effects of punishment without reward.
- Risk: unpublished fork github.com/duterscmy/On-Policy-Self-Adaptation (last push 25 Sep 2026): asks whether
  distillation favors low-confidence tokens because of the loss; correlational top-K vs wrong-answer check on AIME24
  and MATH-500 with Qwen3-1.7B/4B ("only moderate" correlations); no replay, random arm, answer-key arm, rule tasks,
  or results. Re-check novelty before submitting.
- Rejected ideas (do not revive): harness distillation, spec-decoding drafters, "surface entropy", test-call /
  verification tool environment ("just a benchmark in the end"), calculator hidden-tool design.

## Code: token replay pilot (study 3), ~/projects/269_project
Stages run as subprocesses from notebooks/token_replay_pilot.ipynb (built by tools/build_notebook.py; first cell has
the design tables with a reason per choice; the VM clones the repo from GitHub with no credentials, as 239_project does,
so the repo must be public; Steven wants no PAT and no code copies on Drive).

| Stage | File | What it does | Output |
|---|---|---|---|
| 1 GPU vLLM | src/rollout.py | Qwen3-1.7B, thinking off, 60 MATH-500 problems (levels 3-5, seed 0) x 8 answers, T 1.0, top-p 1, no top-k, max 4,096 tokens, generation_config="vllm", stop_token_ids = <|im_end|>, <|endoftext|> (end token appended if missing), graded | problems.jsonl (prompt ids), rollouts.jsonl (response ids, text, vLLM sampled logprobs, correct), rollout.meta.json |
| 2 GPU HF | src/score.py | teacher-forced fp32 log-softmax per response token for student and teacher Qwen3-4B (vocab equality asserted), student entropy; rules on pools of 16; 64 positions per rule; 8 replacement tokens per position drawn from softmax(logits with kept token at -inf), seeded | scores.jsonl, selected.jsonl, positions.jsonl, score.meta.json |
| 3 GPU vLLM | src/replay.py | per position: 8 kept continuations of prompt + y[:t] + y_t; 1 continuation per replacement token; second kept pass on 25% (dup, noise floor); token-id prefixes; same total 4,096 budget; chosen end token graded as is; chunks of 32, resumable via replays.jsonl.partial | replays.jsonl, replay.meta.json |
| analysis | src/analyze.py | tables + gates, laptop or Colab | tables/*.csv |

- Rules (src/select_rules.py): opsa = lowest 20% student logprob per pool (OPSA's code); random = same K uniform;
  teacher_neg = lowest 20% of (log p_teacher - log p_student), negative only (my definition: negative half of OPD
  at OPSA's budget); nsr = every token of a failed response. Pools = 16 shuffled responses (mimics 64 prompts x 1
  sample over 4 GPUs; pilot responses are 8 per problem, so a pool can hold several answers to one problem).
  Positions sampled uniformly without replacement per rule; a position drawn by two rules is replayed once.
- Measures: Q_kept, Q_replaced; usefulness U = Q_kept - Q_replaced (> 0: token helps, pushing it down hurts);
  damage D = p(1 - p)U = first-order loss of success per unit negative step on the token's logit (from
  dJ/dz_y = p_y (Q_y - V) and Q_y - V = (1 - p_y)(Q_y - Q_not_y)); share U > 0, share U < 0; strata by response
  outcome and by problem (all / some / none of 8 correct); token categories of whole selected sets (end token,
  space/newline, digit, doubt/fork word, math/LaTeX, word, other); overlap between rules; context examples.
  95% intervals: bootstrap resampling problems; contrasts vs random resample problems jointly.
- Gates: (0) at least 20 mixed problems (some but not all answers correct), checked right after stage 1 in its
  own notebook cell and printed by rollout.py; replay positions are drawn from mixed problems only
  (score.py --positions-from mixed, default). First real stage 1 (2026-10-21): 30 of 60 mixed, level 3 8/13 all
  correct, level 4 10/26 all correct, pass@1 .64, 3% truncated; (1) mean |HF - vLLM| logprob < 0.02 and 99th pct < 0.25; (2) corr(Q_kept at random positions, problem's
  rollout success) > 0.5; (3) mean |Q_kept - Q_dup| <= 1.25 x binomial expectation + 0.01; (4) OPSA minus random
  usefulness, bootstrap p < 0.05 = detected.
- Size: about 256 positions, about 4,600 continuations; detects only large gaps. SMOKE config: 4 problems x 4
  answers, 1,024 tokens, 3 positions per rule, 2 kept, 2 replacements, dup 0.5 (wiring only).
- Grading (src/grade.py): math-verify on the last complete \boxed{} only, spawned worker pool with 30 s per-item
  timeout (string-match fallback only without math-verify).
- Verified: 23 CPU tests pass (uv run pytest): selector matches OPSA's definition, advantages, sets, position
  sampling, replay requests (prefixes, budgets, end token, dup, seeds), grading, analysis end to end on a synthetic
  run, notebook up to date and its flags exist in the scripts. HF path checked on CPU with Qwen3-0.6B: token scores
  match a full forward to 5e-6, replacement never equals the kept token, Qwen3 0.6B/1.7B/4B share vocabulary and
  prompt ids, chat template has thinking disabled ("<think>\n\n</think>").
- NOT verified: vLLM stages never ran (no GPU on laptop). Risks: generation_config="vllm" arg, logprobs=0 extraction,
  whether the stop token is included in token_ids, runtime (unmeasured), T = 1 answer quality at 1.7B.
- Laptop: uv sync; uv run src/analyze.py --run "<run folder>"; uv run tools/build_notebook.py after cell edits.
- Not in the pilot yet: MBPP+ (needs sandboxed tests), Tuwhy/Qwen3-1.7B-OPSA as the replayed policy, study 1's models.

## Next steps
1. Steven runs the smoke run on Colab, then the pilot; read gates 1 to 3 before gate 4.
2. If gates pass: more positions, MBPP+, same replay on the released OPSA checkpoint (did training remove useful tokens?).
3. Training-free parts still uncoded: study 2 first pass (released OPSA ckpts on SUM, refusal-token premise audit),
   study 1 base rates per operator.
4. Then HF/PEFT reimplementation of the arms for study 1 (LoRA, Qwen3-1.7B).

## Lessons carried from his sibling project (~/projects/239_project, tool atrophy)
- vLLM strips special tokens before stop-string matching: pass end tokens as stop_token_ids.
- Some tokenizers decode wrong under transformers v5 (NuminaMath/DeepSeekMath LlamaTokenizer issue); Qwen3 is fine.
- A regex in grading once hung a Colab run: grade in spawned workers with timeouts.
- He wants every stage gated; he was unhappy when a run scaled up without a gate, and when a model was swapped
  without checking it passed the positive control.

## Status 2026-10-22 (after the first real pilot run, report in reports/)
- Pilot ran locally on the RX 9070 XT (ROCm vLLM). Gates 0 to 3 pass. Signal gates redefined and both NOT DETECTED:
  gate 4 = OPSA's mean damage D interval excludes 0 (D = -0.0053 [-0.0132, +0.0007]); gate 5 = OPSA's useful share
  outside the noise band (0.25 vs null 0.24 [0.14, 0.34]). The old gate 4 (OPSA vs random usefulness) fires for
  structural reasons (random tokens sit at p near 1) and is reported, not gated.
- analyze.py gained share_null (null share of U > 0 and sd of true U), bands (by p of the kept token), and
  `--positions-csv` (rebuild replay tables from the committed CSV). Tables regenerated byte-identical.
- Next, in the Plan's order: study 2 premise audit on SUM (HF `lime-nlp/Synthetic_Unanswerable_Math`), then 32 + 32
  continuations at the same positions if the per-token share is wanted, then the released OPSA checkpoint, then MBPP+.
