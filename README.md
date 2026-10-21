# 269_project

Code for the CS 269 project "Can Punishment Alone Teach a Model Something New, and What Does It Take Away?"
([proposal doc](https://claude.ai/code/artifact/6345c291-d6b0-4017-a4d2-8008e408107b)).

The first piece is the **token replay pilot** (study 3 in the proposal, no training): take the tokens each
negative-only rule would push down in Qwen3-1.7B's own MATH-500 answers, continue each answer with the token
kept and with it replaced, and compare how often the two continuations end correct. The notebook's first cell
describes the design, the models, the data and the reason for each choice.

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
| `src/rollout.py` | 1, GPU (vLLM) | 8 answers per problem at T = 1.0, graded |
| `src/score.py` | 2, GPU (transformers) | student and teacher log-probability of every answer token; the four rules; replay positions and their replacement tokens |
| `src/replay.py` | 3, GPU (vLLM) | kept and replaced continuations, graded; resumable |
| `src/analyze.py` | laptop or Colab | usefulness tables, bootstrap intervals over problems, noise floor, gates |
| `src/select_rules.py` | | OPSA, random, teacher-negative and NSR selection (OPSA's selector follows its released code) |
| `src/common.py`, `src/grade.py` | | prompt, data, files, boxed-answer grading |
| `tools/build_notebook.py` | | writes the notebook; edit cells there, rebuild, commit |

## Laptop

```
uv sync
uv run pytest                                   # CPU tests, no GPU or model downloads
uv run src/analyze.py --run "<run folder>"      # rebuild the tables from a synced run
uv run tools/build_notebook.py                  # after editing the notebook cells
```
