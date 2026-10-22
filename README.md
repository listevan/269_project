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

How the environment is built (`pyproject.toml`, group `rocm`): vLLM's ROCm 10.0 wheel index is searched before
PyPI, `torch[device-gfx1201]` brings the kernels for the RX 9070 XT only, and `exclude-dependencies` drops the
other architectures' kernels that the vLLM wheel would otherwise pull in. The wheels are Linux x86_64 and
Python 3.12 only, so on another machine the group reduces to its pure-Python packages. Triton (vLLM's
torch.compile and its kernels) needs a C compiler at run time; on a machine without gcc the vLLM stages point it
at `tools/rocm_cc`, a wrapper around the clang that the pip ROCm SDK ships.

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
| `run_pilot.py` | local GPU | the whole pipeline from the command line: config, the three stages as subprocesses, gate 0, the tables |
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
