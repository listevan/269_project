#!/usr/bin/env python
"""Run the token replay pilot on this machine, start to finish: the local counterpart of the Colab notebook.

    uv run run_pilot.py --smoke                         # wiring check, a few minutes; its numbers mean nothing
    uv run run_pilot.py                                 # the pilot (stops after stage 1 if gate 0 fails)
    uv run run_pilot.py --set n_problems=100 --run-id 2026-10-22_qwen3-1.7b-n100
    uv run run_pilot.py --stop-after rollout            # look at gate 0 before paying for stages 2 and 3

The stages are the scripts in src/ (rollout, score, replay), each run as a subprocess so the GPU is free again
when it exits; a finished stage is skipped on a rerun, and replay resumes where it stopped. Everything a run
produces lands in runs/<run id>/ (gitignored): the JSONL files, config.json, pilot.log, and tables/.
"""
import argparse
import datetime
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

CFG = dict(student="Qwen/Qwen3-1.7B", teacher="Qwen/Qwen3-4B",
           levels="3,4,5", n_problems=60, samples=8, max_tokens=4096,   # stage 1
           pool_size=16, fraction=0.2, positions=64, redraws=8,          # stage 2
           kept=8, dup_fraction=0.25,                                    # stage 3
           seed=0)
SMOKE = dict(n_problems=4, samples=4, max_tokens=1024, positions=3, redraws=2, kept=2, dup_fraction=0.5)
STAGES = ("rollout", "score", "replay", "analyze")
TITLES = dict(gates="Gates", rollouts="Stage 1 answers by level", usefulness="Usefulness of each rule's tokens",
              contrasts="Each rule minus random", strata="By response outcome and problem difficulty",
              noise="Replay noise (second kept pass)", selection="What each rule selects (whole sets)",
              categories="Token types in each rule's set (shares)",
              overlap="Overlap: share of the row rule's tokens also in the column rule's set")

GPU_CHECK = """
import torch
if not torch.cuda.is_available():
    raise SystemExit("torch sees no GPU. Is /dev/kfd readable (ls -l /dev/kfd), and was the env made with "
                     "`uv sync --group rocm`?")
p = torch.cuda.get_device_properties(0)
hip = getattr(torch.version, "hip", None)
print(f"GPU: {p.name} ({p.gcnArchName if hasattr(p, 'gcnArchName') else 'cuda'}), {p.total_memory / 2**30:.1f} GiB, "
      f"torch {torch.__version__}" + (f", ROCm/HIP {hip}" if hip else ""))
"""


def parse_sets(items, cfg):
    """--set key=value, coerced to the type of the default (ints, floats, strings)."""
    out = {}
    for item in items:
        key, _, val = item.partition("=")
        if key not in cfg or not _:
            sys.exit(f"--set {item}: expected one of {', '.join(cfg)}=value")
        kind = type(cfg[key])
        out[key] = kind(val) if kind is not str else val
    return out


class Log:
    """Print and append to <run>/pilot.log, so an unattended run leaves a record."""

    def __init__(self, path):
        self.f = open(path, "a")
        self.f.write(f"\n===== {datetime.datetime.now():%Y-%m-%d %H:%M:%S} {' '.join(sys.argv)}\n")

    def __call__(self, *parts, end="\n"):
        text = " ".join(str(p) for p in parts) + end
        print(text, end="", flush=True)
        self.f.write(text)
        self.f.flush()


def run(log, script, out, *args):
    cmd = [sys.executable, "-u", str(ROOT / "src" / script), "--run", str(out), *map(str, args)]
    log("$", " ".join(cmd[1:]))
    t0 = time.time()
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=ROOT)
    for line in p.stdout:
        if "it/s]" in line or "s/it]" in line:   # progress bars: keep the log short
            continue
        log(line, end="")
    if p.wait() != 0:
        raise SystemExit(f"{script} failed (exit {p.returncode}); the log above has the error")
    log(f"-- {script} done in {(time.time() - t0) / 60:.1f} min")


def show(log, tables, names):
    import pandas as pd

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.max_colwidth", 90)
    for name in names:
        if name in tables:
            log(f"\n** {TITLES[name]} **")
            log(tables[name].to_string(index=name in ("categories", "overlap")))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true", help="tiny settings: checks the wiring in a few minutes")
    ap.add_argument("--run-id", help="run folder name; an existing one is resumed (default: <date>_qwen3-1.7b)")
    ap.add_argument("--runs", default=str(ROOT / "runs"), help="where run folders live (default: runs/ in the repo)")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override a config value, e.g. --set n_problems=100 (repeatable)")
    ap.add_argument("--stop-after", choices=STAGES, help="stop after this stage")
    ap.add_argument("--ignore-gate0", action="store_true", help="go on to stage 2 even if gate 0 says CHECK")
    ap.add_argument("--skip-gpu-check", action="store_true")
    args = ap.parse_args(argv)

    cfg = dict(CFG)
    if args.smoke:
        cfg.update(SMOKE)
    cfg.update(parse_sets(args.set, cfg))
    run_id = args.run_id or f"{datetime.date.today():%Y-%m-%d}_qwen3-1.7b" + ("-smoke" if args.smoke else "")
    out = Path(args.runs) / run_id
    out.mkdir(parents=True, exist_ok=True)
    cfg_path = out / "config.json"
    if cfg_path.exists() and json.loads(cfg_path.read_text()) != cfg:
        sys.exit(f"{cfg_path} holds a different config; pick another --run-id or delete the folder")
    cfg_path.write_text(json.dumps(cfg, indent=1))
    log = Log(out / "pilot.log")
    log("run:", out)
    log(json.dumps(cfg, indent=1))

    if not args.skip_gpu_check:
        r = subprocess.run([sys.executable, "-c", GPU_CHECK], capture_output=True, text=True)
        log((r.stdout + r.stderr).strip())
        if r.returncode != 0:
            return 1

    import analyze
    import common

    # Stage 1: sample the student's answers, then gate 0 (enough problems with some but not all answers correct)
    run(log, "rollout.py", out, "--model", cfg["student"], "--levels", cfg["levels"], "--n-problems", cfg["n_problems"],
        "--samples", cfg["samples"], "--max-tokens", cfg["max_tokens"], "--seed", cfg["seed"])
    tables, _ = analyze.analyze(out, show=None, write=False)
    show(log, tables, ("rollouts", "gates"))
    gate0 = tables["gates"].iloc[0]["verdict"]
    if gate0 != "PASS" and not args.smoke and not args.ignore_gate0:
        log("\ngate 0 is CHECK: too few mixed problems for the replay to have signal. Raise n_problems "
            "(e.g. --set n_problems=100) under a new --run-id and run again, or pass --ignore-gate0.")
        return 2
    if args.stop_after == "rollout":
        return 0

    # Stage 2: score every token, apply the four rules, draw the replacement tokens
    run(log, "score.py", out, "--teacher", cfg["teacher"], "--pool-size", cfg["pool_size"], "--fraction", cfg["fraction"],
        "--positions", cfg["positions"], "--redraws", cfg["redraws"], "--seed", cfg["seed"])
    if args.stop_after == "score":
        return 0

    # Stage 3: kept and replaced continuations, graded (resumable)
    run(log, "replay.py", out, "--kept", cfg["kept"], "--dup-fraction", cfg["dup_fraction"], "--seed", cfg["seed"])
    if args.stop_after == "replay":
        return 0

    # Results: every table also goes to <run>/tables/. Read the gates first.
    tables, pf = analyze.analyze(out, show=None)
    show(log, tables, ("gates", "rollouts", "usefulness", "contrasts", "strata", "noise", "selection", "categories", "overlap"))
    scores = common.read_jsonl(out / "scores.jsonl")
    for rule in ("opsa", "teacher_neg"):
        log(f"\n** {analyze.RULES[rule]}: most useful and most harmful replayed tokens, [kept token] in context **")
        log(analyze.examples(pf, scores, rule, k=5).to_string(index=False))
    log(f"\ntables written to {out / 'tables'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
