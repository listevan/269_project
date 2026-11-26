# 269_project

Code for the CS 269 project "Can Punishment Alone Teach a Model Something New, and What Does It Take Away?"
([proposal doc](https://claude.ai/code/artifact/6345c291-d6b0-4017-a4d2-8008e408107b)).

The first piece is the **token replay pilot** (study 3 in the proposal, no training): take the tokens each
negative-only rule would push down in Qwen3-1.7B's own MATH-500 answers, continue each answer with the token
kept and with it replaced, and compare how often the two continuations end correct. The notebook's first cell
describes the design, the models, the data and the reason for each choice.

## Run it locally (Linux, AMD RX 9070 XT)

The three GPU stages run on the local card through vLLM's own ROCm wheels. PyTorch and the ROCm runtime come
from pip with them, so nothing is installed system-wide and no sudo is needed: only the kernel's amdgpu driver
and read access to `/dev/kfd` and `/dev/dri/renderD128` (a desktop login has it; otherwise add yourself to the
`render` and `video` groups).

```
uv sync                       # Python 3.12 (.python-version) + the dev and rocm groups into .venv (3 GB download, 14 GB on disk)
uv run run_pilot.py --smoke   # wiring check, a few minutes; fetches Qwen3-1.7B and Qwen3-4B (about 12 GB) into ~/.cache/huggingface
uv run run_pilot.py           # the pilot; stops after stage 1 if gate 0 fails
```

Each run lands in `runs/<date>_qwen3-1.7b/` (gitignored): the JSONL files, `config.json`, `pilot.log`, and the
tables in `tables/`. `--run-id` names or resumes a run (finished stages are skipped, replay continues where it
stopped), `--stop-after rollout` lets you read gate 0 before paying for stages 2 and 3, and
`--set n_problems=100` changes the config (use a new `--run-id`). `uv run run_pilot.py -h` lists the rest.

The notebook works locally too: pick `.venv` as the kernel in VS Code. Cells 2 and 4 then install nothing and
run this checkout instead of cloning.

### Experiment 3: learning a hidden rule (the proposal's main controlled study)

```
uv run run_rules.py --smoke     # 3 operators, 3 arms x 2 training steps: about 20 minutes
uv run run_rules.py             # about 12 operators, 7 arms x 60 LoRA steps, 16-sample evaluations: overnight
```

The task (`src/rules_data.py`): "Compute 7 # 4." with the definition (a # b = 2a + b) left out, so the correct answer
starts rare and the dominant wrong answer is identifiable. The driver first measures the base model's correct rate on
26 candidate symbol/definition pairs and keeps about twelve spanning near 0% to about 50%. `src/neg_train.py` trains
one arm at a time (LoRA on Qwen3-1.7B, rollouts with `generate`, pools of 16 responses, token-mean loss): OPSA, random
selection, teacher-negative, teacher with both signs, teacher-selected under OPSA's penalty, the answer-key arm and the
SFT demonstration control; the teacher is the same model with the definition in its prompt. `src/rules_eval.py`
samples 16 answers per item on four splits (new numbers, new range, compositions, with the definition) with vLLM
serving the adapter; `src/rules_probe.py` reads p(correct) and p(dominant wrong) at the box from the logits;
`src/teacher_audit.py` is the decile audit of the teacher's signal before training; `src/rules_analysis.py` writes
the tables. Training checkpoints every 5 steps and resumes; `tools/exp3_watchdog.sh` relaunches the driver from cron
if a session restart kills it. Results: `reports/2026-11-24_hidden_rule.md` (no on-policy arm grew a rare correct
answer; SFT did; lr 1e-4 collapsed OPSA into capped responses, so every arm ran at 2e-5).

### Experiments 2 and 7: pass@k and per-problem analysis of the released OPSA checkpoint

```
uv run run_passk.py --smoke     # 4 problems, a few minutes
uv run run_passk.py             # 100 MATH-500 problems: base (grouping and scoring batches, a temperature sweep), the OPSA checkpoint at a 16k cap, the coverage audit; about 6 hours
```

`src/passk.py` turns the rollouts into pass@k curves, the Sharpening Tax, per-group accuracies (by what the base's
grouping batch did), the dominant-wrong-answer mass, side effects, and the list of problems the trained model solved
that the base never did; those get 64 extra base samples each. All arms share one problem subset
(`rollout.py --problem-seed`). Results: `reports/2026-11-24_passk_opsa_vs_base.md`.

### Study 2: the premise audit on SUM (the proposal's experiment 5, measured first)

```
uv run run_premise_audit.py --smoke     # 4 problem pairs, a few minutes
uv run run_premise_audit.py             # base model (seeds 0 and 1) and the released OPSA checkpoint on all of SUM, about 6 hours
uv run run_premise_audit.py --arms opsa:0 --set max_tokens=16384 --set samples=1 --set gpu_mem=0.92 --rates-only --tag <tag>
                                        # the checkpoint writes long answers: rates only at a 16k cap, about 3.5 hours
```

For each arm, stage 1 samples 8 answers to every one of SUM's 284 unanswerable problems and to their solvable
originals (`src/rollout.py --dataset sum`), then `src/premise_audit.py` scores every token with the model itself,
applies OPSA's rule on pools of 16, and measures: refusal, false-answer and no-answer rates on the unanswerable
items with accuracy on the originals; whether the refusal tokens of refusing responses (the boxed refusal, its first
token, the hedge phrases in the reasoning) fall inside OPSA's punished set; the probability of starting a refusal at
the decision point, read from the logits after `\boxed`; and the side effects (length, cap hits, repetition). The
base model's second seed is the frozen-model control. Runs land in `runs/<date>_sum_<model>_s<seed>/`, with a
comparison across arms in `runs/<date>_sum_summary.{log,json}`.

How the environment is built (`pyproject.toml`, group `rocm`): vLLM's ROCm 10.0 wheel index is searched before
PyPI, `torch[device-gfx1201]` brings the kernels for the RX 9070 XT only, and `exclude-dependencies` drops the
other architectures' kernels that the vLLM wheel would otherwise pull in. The wheels are Linux x86_64 and
Python 3.12 only, so on another machine the group reduces to its pure-Python packages. Triton (vLLM's
torch.compile and its kernels) needs a C compiler at run time; on a machine without gcc the vLLM stages point it
at `tools/rocm_cc`, a wrapper around the clang that the pip ROCm SDK ships.

### Poster

`docs/poster/plan.md` is the poster plan (story, panels, numbers, figures). `tools/poster_figures.py` draws the
figures as SVG from the committed tables (`docs/poster/*.svg`; `--scale 1.3` gives poster type), and
`tools/poster_build.js` lays out the 48 x 36 in PowerPoint file `docs/poster/poster_48x36.pptx` with pptxgenjs.
`tools/poster_make.sh` runs all three (needs node with pptxgenjs on `NODE_PATH`, and rsvg-convert).

## Run it (Colab)

1. Open `notebooks/token_replay_pilot.ipynb` in Colab (File > Open notebook > GitHub, repo
   `listevan/269_project`), on a GPU runtime: an A100 if available; an L4 or T4 works, slower.
2. Run top to bottom with `SMOKE = True` (a few minutes, wiring only), then again with `SMOKE = False`.
   Results land in `My Drive/.../CS 269/269_project/runs/<date>_qwen3-1.7b/`.

Cell 4 clones this repo onto the VM with no credentials, the same way 239_project does, so the repo has to be
public. The VM runs what is on GitHub: push edits before running.

## Layout

| File | Stage | What it does |
|---|---|---|
| `src/rollout.py` | 1, GPU (vLLM) | 8 answers per problem at T = 1.0, graded; `--dataset sum` for study 2; `--temperature`, `--problem-seed`, `--problem-ids` for experiments 2 and 7 |
| `src/score.py` | 2, GPU (transformers) | student and teacher log-probability of every answer token; the four rules; replay positions and their replacement tokens |
| `src/replay.py` | 3, GPU (vLLM) | kept and replaced continuations, graded; resumable |
| `run_pilot.py` | local GPU | the whole pipeline from the command line: config, the three stages as subprocesses, gate 0, the tables |
| `run_premise_audit.py` | local GPU | study 2: SUM rollouts and the premise audit for the base model (two seeds) and the OPSA checkpoint, with a comparison table |
| `run_rules.py`, `src/neg_train.py` | local GPU | experiment 3: operator choice, base batches, the teacher-signal audit, seven LoRA training arms, their evaluations and probes, the tables |
| `src/rules_data.py`, `src/rules_eval.py`, `src/rules_probe.py`, `src/teacher_audit.py`, `src/rules_analysis.py` | GPU / CPU | the hidden-rule task, vLLM evaluation of a (merged) checkpoint, the logit read, the decile audit, the analysis |
| `run_passk.py`, `src/passk.py` | local GPU, then CPU | experiments 2 and 7: base batches, temperature sweep, OPSA checkpoint; pass@k curves, Sharpening Tax, per-group and per-problem tables, coverage audit |
| `src/sum_data.py` | | SUM problems (unanswerable + solvable original), the dataset's prompt, refusal / false-answer grading |
| `src/premise_audit.py` | 2, GPU (transformers) | token scores, OPSA's rule, refusal and hedge tokens against the punished set, the decision-point distribution, rates and side effects |
| `src/analyze.py` | laptop or Colab | usefulness tables, bootstrap intervals over problems, noise floor, gates |
| `src/select_rules.py` | | OPSA, random, teacher-negative and NSR selection (OPSA's selector follows its released code) |
| `src/common.py`, `src/grade.py` | | prompt, data, files, boxed-answer grading |
| `tools/build_notebook.py` | | writes the Colab notebook; edit cells there, rebuild, commit |
| `tools/rocm_cc` | local GPU | C compiler for Triton on a machine without gcc: the pip ROCm SDK's clang |

## Laptop

```
uv sync                                         # on a machine without the GPU wheels: the pure-Python packages only
uv run pytest                                   # CPU tests, no GPU or model downloads
uv run src/analyze.py --run "<run folder>"      # rebuild the tables from a synced run
uv run tools/build_notebook.py                  # after editing the notebook cells
```
