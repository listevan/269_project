"""Write notebooks/token_replay_pilot.ipynb from the cell sources below (uv run tools/build_notebook.py).

The notebook holds no code of its own beyond calling src/, which the VM clones from GitHub. Edit the cells
here, rebuild, commit, push.
"""
from pathlib import Path

import nbformat as nbf

INTRO = r"""# Pilot: how useful are the tokens negative-only training pushes down?

This is the pilot for the third study in the proposal. Negative-only methods push down some tokens in a model's
own answers. OPSA pushes down the least likely ones, so it cannot tell a mistake from a rare good step. For each
method I take the tokens it would push down, continue the answer from that point once with the token kept and
once with it replaced, and compare how often the two continuations reach the correct answer.

Nothing is trained. Each rule is applied to the base model's own answers, and the replays only sample from it.

### What I compare

| Rule | Tokens it pushes down | Uses | Why it is in the comparison |
|---|---|---|---|
| **OPSA** | lowest 20% of student log-probabilities in each pool of 16 responses | nothing | the method under study; selector reimplemented from its released code |
| **Random 20%** | 20% of tokens drawn uniformly from the same pools | nothing | tells whether picking the *unlikely* tokens matters at all |
| **Teacher-negative** | lowest 20% of log p_teacher − log p_student, negative values only | a stronger model | the negative half of on-policy distillation, cut to OPSA's budget |
| **NSR** | every token of a failed response | the answer key | the punishment-only method that knows which answers are wrong |

### What I have

| Item | What it is | Why this one |
|---|---|---|
| Student | `Qwen/Qwen3-1.7B`, thinking off | OPSA's main model and setting; a released OPSA checkpoint of it exists for the follow-up; under 7B and fits a single Colab GPU |
| Teacher | `Qwen/Qwen3-4B`, thinking off | same tokenizer (checked: identical vocabulary and prompt ids), so its score for each student token lines up one to one; stronger at math; under 7B |
| Dataset | MATH-500, 60 problems from levels 3 to 5 (fixed seed) | the abstract's math set, with exact answers to grade. At levels 3 to 5 the 1.7B model is often right but not always, which is where kept and replaced tokens can lead to different outcomes |
| Responses | 8 per problem, T = 1.0, no top-p or top-k, at most 4,096 tokens | OPSA's training rollouts use T = 1 with no truncation of the distribution, so these are the tokens it would actually train on. 4,096 tokens covers non-thinking answers (truncation is reported) |
| Selection pools | 16 responses, fraction f = 0.2 | OPSA selects across one GPU's share of the batch: 64 prompts × 1 sample over 4 GPUs = 16 responses, K = max(1, ⌊0.2 N⌋) |
| Replay sample | 64 positions per rule, drawn within problems the model sometimes solves (gate 0), 8 kept + 8 replaced continuations each, a second kept pass on 25% of positions | about 4,600 continuations, enough to see large gaps only; the second pass measures replay noise. Where the model always or never succeeds the kept side is pinned near 1 or 0, so a token's usefulness can barely show |
| Grading | math-verify on the last `\boxed{}` only | the prompt asks for a boxed answer; stray unboxed numbers get no credit |
| Prefixes | token ids, never re-tokenized text | re-feeding decoded text can change the state being replayed |
| Not used yet | released OPSA checkpoint `Tuwhy/Qwen3-1.7B-OPSA`; MBPP+; the trained models of study 1 | each waits until this pilot passes its gates (MBPP+ also needs sandboxed test runs) |

### What I measure

For a token y at position t of a response to prompt x, with probability p under the student:

| Measure | Definition | How to read it |
|---|---|---|
| Q_kept | share of 8 continuations of x, y<sub>&lt;t</sub>, y that end correct | success if the token stays |
| Q_replaced | share of 8 continuations of x, y<sub>&lt;t</sub>, a that end correct, each a drawn from p with y removed | success if the token's probability goes elsewhere |
| usefulness U | Q_kept − Q_replaced | U > 0: the token helps, so pushing it down hurts |
| damage D | p (1 − p) U | expected loss in success from one unit of push on y's logit |
| share useful | share of a rule's tokens (in the replayed problems) with U > 0 | the abstract's "share of useful tokens in a method's penalized set" |

**Why replace by drawing from p with y removed.** A negative update on y lowers its logit, and the softmax hands
its probability to every other token in proportion to that token's probability. Drawing from p with y removed is
exactly that redistribution.

**Why damage.** Let J be the chance of success from this point, J = Σ<sub>a</sub> p<sub>a</sub> Q<sub>a</sub>.
Lowering y's logit by ε changes it by

$$\Delta J = -\,\varepsilon\; p\,(1-p)\,\bigl(Q_\text{kept} - Q_\text{replaced}\bigr) = -\,\varepsilon\, D .$$

U alone treats a token the model was 99.99% sure of the same as an uncertain one. D accounts for the fact that a
push barely moves a near-certain token, so D is the fairer number for comparing rules that pick tokens of very
different probability. This is a local, one-position view. It ignores how an update at one token changes
behavior at other tokens through the shared weights.

### Gates (0 right after stage 1, the rest at the end)

0. **Enough mixed problems.** At least 20 problems have some but not all of their 8 answers correct. Replay positions are drawn from these problems only, and the bootstrap resamples them, so there have to be enough of them. If this fails, raise `n_problems` (levels 4 and 5 alone hold 262) or `samples`, change `RUN_ID`, and run stage 1 again.
1. **Scoring matches sampling.** Teacher-forced log-probabilities equal vLLM's for the same tokens (mean gap under 0.02).
2. **Replay reproduces the state.** Q_kept at random positions tracks the problem's rollout success (correlation above 0.5).
3. **Replay noise is only sampling noise.** The second kept pass differs from the first by no more than binomial noise predicts.
4. **Signal.** OPSA's mean usefulness differs from random's, with a 95% interval that resamples problems.

If 0 to 3 pass, the next step is a larger run (more positions, MBPP+) and the same replay on the released OPSA checkpoint.

**Run order.** Set `SMOKE = True` first: about 5 minutes, checks wiring only. Then `SMOKE = False` for the pilot.
"""

GPU = r"""# 1. GPU
import shutil, subprocess
if shutil.which("nvidia-smi"):
    print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv"],
                         capture_output=True, text=True).stdout)
else:
    print("NO GPU in this runtime. Colab: Runtime > Change runtime type > GPU, then run again from this cell.")
"""

INSTALL = r"""# 2. Install (vllm pins its own torch and transformers; every stage runs in a subprocess, so no restart)
import subprocess, sys
PKGS = "vllm datasets math-verify pandas"
if subprocess.run(f"pip install -q uv && uv pip install --system -q {PKGS}", shell=True).returncode != 0:
    subprocess.run(f"pip install -q {PKGS}", shell=True, check=True)
print(subprocess.run([sys.executable, "-c", "import vllm, transformers; print('vllm', vllm.__version__, "
                      "'transformers', transformers.__version__)"], capture_output=True, text=True).stdout
      or "import check failed, see above")
"""

DRIVE = r"""# 3. Drive: reuse a 269_project folder anywhere up to 5 levels under MyDrive, else make one in "CS 269"
import glob, os
try:
    from google.colab import drive
    drive.mount("/content/drive")
    my = "/content/drive/MyDrive"
    hits = [p for d in range(6) for p in glob.glob(f"{my}/" + "*/" * d + "269_project") if os.path.isdir(p)]
    if hits:
        ROOT = sorted(hits, key=lambda p: p.count("/"))[0]
    else:
        cs = [p for d in range(4) for p in glob.glob(f"{my}/" + "*/" * d + "CS 269") if os.path.isdir(p)]
        ROOT = f"{cs[0]}/269_project" if cs else f"{my}/269_project"
except ImportError:
    ROOT = os.path.abspath("269_project_local")   # outside Colab: a local folder
RUNS = f"{ROOT}/runs"
os.makedirs(RUNS, exist_ok=True)
print("runs ->", RUNS)
"""

CLONE = r"""# 4. Code: clone the repo onto the VM, or pull it to the latest commit. The VM runs what is on GitHub.
# No credentials, as in 239_project: the repo must be public. A private repo stops here with a clear message.
import os, subprocess, sys
REPO, BRANCH = "https://github.com/listevan/269_project.git", "main"
WORK = "/content/269_project" if os.path.isdir("/content") else os.path.abspath("269_project_work")
quiet = dict(capture_output=True, text=True)
if os.path.isdir(f"{WORK}/.git"):
    r = subprocess.run(["git", "-C", WORK, "fetch", "-q", "origin", BRANCH], **quiet)
    if r.returncode == 0:
        subprocess.run(["git", "-C", WORK, "reset", "-q", "--hard", f"origin/{BRANCH}"], **quiet)
else:
    r = subprocess.run(["git", "clone", "-q", "--depth", "1", "-b", BRANCH, REPO, WORK], **quiet)
if r.returncode != 0:
    raise SystemExit("clone failed. Is the repo public? git said: " + r.stderr.strip()[-300:])
os.chdir(WORK)
if f"{WORK}/src" not in sys.path:
    sys.path.insert(0, f"{WORK}/src")
print(subprocess.run(["git", "log", "-1", "--format=%h  %s  (%cr)"], **quiet).stdout.strip())
"""

CONFIG = r"""# 5. Configure. SMOKE = True checks the wiring in a few minutes; its numbers mean nothing.
import datetime, json, os
SMOKE = True

CFG = dict(student="Qwen/Qwen3-1.7B", teacher="Qwen/Qwen3-4B",
           levels="3,4,5", n_problems=60, samples=8, max_tokens=4096,   # stage 1
           pool_size=16, fraction=0.2, positions=64, redraws=8,          # stage 2
           kept=8, dup_fraction=0.25,                                    # stage 3
           seed=0)
if SMOKE:
    CFG.update(n_problems=4, samples=4, max_tokens=1024, positions=3, redraws=2, kept=2, dup_fraction=0.5)

# One folder per run. Set RUN_ID to an existing folder to resume it; finished stages are skipped.
RUN_ID = f"{datetime.date.today():%Y-%m-%d}_qwen3-1.7b" + ("-smoke" if SMOKE else "")
OUT = f"{RUNS}/{RUN_ID}"
os.makedirs(OUT, exist_ok=True)
cfg_path = f"{OUT}/config.json"
if os.path.exists(cfg_path) and json.load(open(cfg_path)) != CFG:
    raise SystemExit(f"{cfg_path} holds a different config; change RUN_ID or delete the folder")
json.dump(CFG, open(cfg_path, "w"), indent=1)
print(OUT)
print(json.dumps(CFG, indent=1))
"""

HELPERS = r"""# 6. Run helper: each stage is a subprocess, so the GPU is free again when it exits
import subprocess, sys, time

def run(script, *args):
    cmd = [sys.executable, "-u", f"src/{script}", "--run", OUT, *map(str, args)]
    print("$", " ".join(cmd[1:]), flush=True)
    t0 = time.time()
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in p.stdout:
        if "it/s]" in line or "s/it]" in line:   # progress bars: keep the log short
            continue
        print(line, end="")
    if p.wait() != 0:
        raise RuntimeError(f"{script} failed (exit {p.returncode}); the log above has the error")
    print(f"-- {script} done in {(time.time() - t0) / 60:.1f} min")
"""

STAGE1_MD = r"""### Stage 1: sample the student's answers

60 problems × 8 answers at T = 1.0, graded. Writes `problems.jsonl`, `rollouts.jsonl`."""
STAGE1 = r"""run("rollout.py", "--model", CFG["student"], "--levels", CFG["levels"], "--n-problems", CFG["n_problems"],
    "--samples", CFG["samples"], "--max-tokens", CFG["max_tokens"], "--seed", CFG["seed"])
"""

GATE0_MD = r"""### Gate 0: enough mixed problems

The replay only has signal on problems the model sometimes solves, so stage 2 draws its positions from those.
Read this before stage 2. If the verdict is CHECK, raise `n_problems` in the config cell, change `RUN_ID`, and
run stage 1 again."""
GATE0 = r"""import importlib
import analyze
importlib.reload(analyze)
from IPython.display import display

tables, _ = analyze.analyze(OUT, show=None, write=False)
display(tables["rollouts"])
display(tables["gates"])
"""

STAGE2_MD = r"""### Stage 2: score every token, apply the four rules, draw the replacement tokens

Student and teacher log-probabilities for every response token (transformers, teacher-forced), the four rules
on pools of 16 responses, 64 positions per rule from the mixed problems, and 8 replacement tokens per position. Writes `scores.jsonl`,
`selected.jsonl`, `positions.jsonl`."""
STAGE2 = r"""run("score.py", "--teacher", CFG["teacher"], "--pool-size", CFG["pool_size"], "--fraction", CFG["fraction"],
    "--positions", CFG["positions"], "--redraws", CFG["redraws"], "--seed", CFG["seed"])
"""

STAGE3_MD = r"""### Stage 3: replay

Kept and replaced continuations for every position, graded. Resumable: if the runtime disconnects, run cells 1
to 6 again and this cell continues where it stopped. Writes `replays.jsonl`."""
STAGE3 = r"""run("replay.py", "--kept", CFG["kept"], "--dup-fraction", CFG["dup_fraction"], "--seed", CFG["seed"])
"""

ANALYSIS_MD = r"""### Results

Tables are also saved under `<run>/tables/`. Read the gates first."""
ANALYSIS = r"""import importlib
import analyze, common
importlib.reload(analyze)
from IPython.display import display, Markdown

tables, pf = analyze.analyze(OUT, show=None)
titles = dict(gates="Gates", rollouts="Stage 1 answers by level", usefulness="Usefulness of each rule's tokens",
              contrasts="Each rule minus random", strata="By response outcome and problem difficulty",
              noise="Replay noise (second kept pass)", selection="What each rule selects (whole sets)",
              categories="Token types in each rule's set (shares)", overlap="Overlap: share of the row rule's tokens also in the column rule's set")
for name in ("gates", "rollouts", "usefulness", "contrasts", "strata", "noise", "selection", "categories", "overlap"):
    if name in tables:
        display(Markdown(f"**{titles[name]}**"))
        display(tables[name])
"""

EXAMPLES = r"""# The most useful and most harmful tokens OPSA would push down, in context ([kept token])
scores = common.read_jsonl(f"{OUT}/scores.jsonl")
for rule in ("opsa", "teacher_neg"):
    display(Markdown(f"**{analyze.RULES[rule]}**"))
    display(analyze.examples(pf, scores, rule, k=5))
"""

LAPTOP = r"""### On the laptop

Google Drive for desktop syncs the run folder, so the same tables can be rebuilt without a GPU:

```
cd ~/projects/269_project
uv run src/analyze.py --run "<Drive>/CS 269/269_project/runs/<run id>"
```
"""


def main():
    nb = nbf.v4.new_notebook()
    cells = [nbf.v4.new_markdown_cell(INTRO)]
    for src in (GPU, INSTALL, DRIVE, CLONE, CONFIG, HELPERS):
        cells.append(nbf.v4.new_code_cell(src.strip()))
    for md, src in ((STAGE1_MD, STAGE1), (GATE0_MD, GATE0), (STAGE2_MD, STAGE2), (STAGE3_MD, STAGE3),
                    (ANALYSIS_MD, ANALYSIS)):
        cells += [nbf.v4.new_markdown_cell(md.strip()), nbf.v4.new_code_cell(src.strip())]
    cells += [nbf.v4.new_code_cell(EXAMPLES.strip()), nbf.v4.new_markdown_cell(LAPTOP.strip())]
    for i, c in enumerate(cells):   # fixed ids, so a rebuild with no edits changes nothing
        c["id"] = f"cell{i:02d}"
    nb.cells = cells
    nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}, "accelerator": "GPU", "colab": {"gpuType": "A100"}}
    out = Path(__file__).resolve().parents[1] / "notebooks" / "token_replay_pilot.ipynb"
    nbf.validate(nb)
    nbf.write(nb, out)
    print(f"wrote {out} ({len(cells)} cells)")


if __name__ == "__main__":
    main()
