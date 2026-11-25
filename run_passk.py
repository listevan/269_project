#!/usr/bin/env python
"""Experiments 2 and 7 on this machine: per-problem and pass@k analysis of the released OPSA checkpoint against its base.

    uv run run_passk.py --smoke          # 4 problems, a few minutes
    uv run run_passk.py                  # 100 MATH-500 problems (levels 3-5): about 6 hours, most of it the OPSA arm

Arms, each a stage-1 rollout (src/rollout.py) on the same problems:
  base A   Qwen3-1.7B, T = 1.0, 16 samples   grouping batch (all / some / no sampled attempts solved)
  base B   Qwen3-1.7B, T = 1.0,  8 samples   scoring batch, the frozen-model control
  sweep    Qwen3-1.7B at T = 0.8, 0.6, 0.4, 8 samples each   the sampling control (how much a sharper sampler alone does)
  trained  Tuwhy/Qwen3-1.7B-OPSA, T = 1.0, 8 samples, 16,384-token cap (it writes long answers)
Then src/passk.py: pass@k curves, Sharpening Tax, per-group accuracies, dominant-wrong-answer mass, side effects, and the
list of problems the trained model solved that neither base batch did. Those get the coverage audit: 64 extra base
samples each; a problem counts as newly solvable only if the base still never solves it.
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

BASE, OPSA = "Qwen/Qwen3-1.7B", "Tuwhy/Qwen3-1.7B-OPSA"
CFG = dict(n_problems=100, levels="3,4,5", problem_seed=0, group_samples=16, samples=8, base_max_tokens=4096, trained_max_tokens=16384,
           sweep="0.8,0.6,0.4", audit_samples=64, k=8, gpu_mem=0.92)   # problem_seed 0: the pilot's 60 problems are the first 60
SMOKE = dict(n_problems=4, group_samples=4, samples=2, base_max_tokens=512, trained_max_tokens=1024, sweep="0.6", audit_samples=4, k=2)


class Log:
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
        if "it/s" in line or "s/it" in line:
            continue
        log(line, end="")
    if p.wait() != 0:
        raise SystemExit(f"{script} failed (exit {p.returncode}); the log above has the error")
    log(f"-- {script} done in {(time.time() - t0) / 60:.1f} min")


def rollout(log, out, cfg, model, samples, max_tokens, temperature, seed, problem_ids=None):
    out.mkdir(parents=True, exist_ok=True)
    this = dict(model=model, samples=samples, max_tokens=max_tokens, temperature=temperature, seed=seed,
                n_problems=cfg["n_problems"], levels=cfg["levels"], problem_seed=cfg["problem_seed"], problem_ids=problem_ids)
    cfg_path = out / "config.json"
    if cfg_path.exists() and json.loads(cfg_path.read_text()) != this:
        sys.exit(f"{cfg_path} holds a different config; use another --tag or delete the folder")
    cfg_path.write_text(json.dumps(this, indent=1))
    args = ["--model", model, "--levels", cfg["levels"], "--n-problems", cfg["n_problems"], "--samples", samples,
            "--max-tokens", max_tokens, "--temperature", temperature, "--seed", seed, "--problem-seed", cfg["problem_seed"],
            "--gpu-mem", cfg["gpu_mem"]]
    if problem_ids:
        args += ["--problem-ids", ",".join(problem_ids)]
    run(log, "rollout.py", out, *args)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--runs", default=str(ROOT / "runs"))
    ap.add_argument("--tag", default=None)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--skip-audit", action="store_true")
    ap.add_argument("--skip-gpu-check", action="store_true")
    args = ap.parse_args(argv)

    cfg = dict(CFG)
    if args.smoke:
        cfg.update(SMOKE)
    for item in args.set:
        k, _, v = item.partition("=")
        if k not in cfg:
            sys.exit(f"--set {item}: expected one of {', '.join(cfg)}=value")
        cfg[k] = type(cfg[k])(v)
    if cfg["k"] > min(cfg["samples"], cfg["group_samples"]):
        sys.exit(f"k={cfg['k']} exceeds the batch size (samples={cfg['samples']}, group_samples={cfg['group_samples']})")
    tag = (args.tag or f"{datetime.date.today():%Y-%m-%d}") + ("-smoke" if args.smoke else "")
    runs = Path(args.runs)
    runs.mkdir(parents=True, exist_ok=True)
    log = Log(runs / f"{tag}_passk.log")
    log("config:", json.dumps(cfg))

    if not args.skip_gpu_check:
        r = subprocess.run([sys.executable, "-c", "import torch; assert torch.cuda.is_available(), 'no GPU'; "
                            "p = torch.cuda.get_device_properties(0); print('GPU:', p.name, f'{p.total_memory / 2**30:.1f} GiB')"],
                           capture_output=True, text=True)
        log((r.stdout + r.stderr).strip())
        if r.returncode != 0:
            return 1

    pre = runs / f"{tag}_passk"
    sweeps = {}
    log("\n=== base A (grouping)")
    group = rollout(log, pre / "baseA_T1.0", cfg, BASE, cfg["group_samples"], cfg["base_max_tokens"], 1.0, 10)
    log("\n=== base B (scoring, frozen-model control)")
    base = rollout(log, pre / "baseB_T1.0", cfg, BASE, cfg["samples"], cfg["base_max_tokens"], 1.0, 11)
    for i, t in enumerate(x.strip() for x in cfg["sweep"].split(",") if x.strip()):
        log(f"\n=== base sweep T={t}")
        sweeps[t] = rollout(log, pre / f"base_T{t}", cfg, BASE, cfg["samples"], cfg["base_max_tokens"], float(t), 12 + i)
    log("\n=== trained: OPSA checkpoint")
    trained = rollout(log, pre / "opsa_T1.0", cfg, OPSA, cfg["samples"], cfg["trained_max_tokens"], 1.0, 15)

    import passk

    tables = passk.analyze(group, base, trained, sweeps, cfg["k"], pre / "tables", show=log)
    new_ids = [q for q in str(tables["transfer"].iloc[0]["newly solved ids"]).split(",") if q]
    if new_ids and not args.skip_audit:
        log(f"\n=== coverage audit: {len(new_ids)} problem(s) the trained model solved and neither base batch did; "
            f"{cfg['audit_samples']} extra base samples each")
        audit = rollout(log, pre / "audit_base", cfg, BASE, cfg["audit_samples"], cfg["base_max_tokens"], 1.0, 20, problem_ids=new_ids)
        import common
        pp = passk.per_problem(common.read_jsonl(audit / "rollouts.jsonl"))
        missing = [q for q in new_ids if q not in pp]
        if missing:
            log("audit: no rollouts for", ", ".join(missing))
        rows = [dict(problem=q, extra_base_samples=pp[q]["n"], base_solved_in_extra=pp[q]["c"],
                     verdict="still never solved by the base: a candidate new solve, review the derivation by hand" if pp[q]["c"] == 0
                     else "the base solves it too: not new") for q in new_ids if q in pp]
        import pandas as pd
        pd.DataFrame(rows).to_csv(pre / "tables" / "audit.csv", index=False)
        log(pd.DataFrame(rows).to_string(index=False))
    log(f"\ntables written to {pre / 'tables'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
