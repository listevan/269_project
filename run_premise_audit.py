#!/usr/bin/env python
"""Study 2 (the proposal's experiment 5), the part it says to measure first, on this machine:

    uv run run_premise_audit.py --smoke        # wiring check on 4 problem pairs, a few minutes
    uv run run_premise_audit.py                # base model (two seeds) and the released OPSA checkpoint on all of SUM

For every (model, seed): stage 1 samples the model's answers to SUM's 284 unanswerable problems and their solvable
originals (src/rollout.py --dataset sum), then src/premise_audit.py scores the tokens, applies OPSA's rule and
measures (a) refusal, false-answer and no-answer rates on the unanswerable items and accuracy on the originals,
(b) whether the refusal tokens in refusing responses fall inside OPSA's punished set, (c) the probability of starting
a refusal at the decision point, read from the logits, and (d) side effects: length, cap hits, repetition.
The base model's second seed is the frozen-model control: the gap between two base batches is the noise floor a
trained model's change must exceed. Runs land in runs/<date>_sum_<model>_s<seed>/; a comparison table across runs
is printed and written to runs/<date>_sum_summary.{json,md}.
"""
import argparse
import datetime
import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

CFG = dict(n_problems=284, samples=8, max_tokens=4096, pool_size=16, fraction=0.2, gpu_mem=0.85)   # the seed is the arm's
SMOKE = dict(n_problems=4, samples=2, max_tokens=768)
BASE, OPSA = "Qwen/Qwen3-1.7B", "Tuwhy/Qwen3-1.7B-OPSA"
ARMS = [(BASE, 0), (BASE, 1), (OPSA, 0)]   # (model, rollout seed); seed 1 of the base is the frozen-model control


def short(model):
    return model.split("/")[-1].lower()


def run(log, script, out, *args):
    cmd = [sys.executable, "-u", str(ROOT / "src" / script), "--run", str(out), *map(str, args)]
    log("$", " ".join(cmd[1:]))
    t0 = time.time()
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=ROOT)
    for line in p.stdout:
        if "it/s]" in line or "s/it]" in line:
            continue
        log(line, end="")
    if p.wait() != 0:
        raise SystemExit(f"{script} failed (exit {p.returncode}); the log above has the error")
    log(f"-- {script} done in {(time.time() - t0) / 60:.1f} min")


class Log:
    def __init__(self, path):
        self.f = open(path, "a")
        self.f.write(f"\n===== {datetime.datetime.now():%Y-%m-%d %H:%M:%S} {' '.join(sys.argv)}\n")

    def __call__(self, *parts, end="\n"):
        text = " ".join(str(p) for p in parts) + end
        print(text, end="", flush=True)
        self.f.write(text)
        self.f.flush()


def compare(run_dirs, log):
    """Rates and premise numbers side by side, plus the frozen-model noise floor (base seed 0 vs seed 1)."""
    import common

    rates, premise, decision = {}, {}, {}
    for name, d in run_dirs.items():
        a = common.read_json(d / "audit.json")
        rates[name] = {(r["variant"], k): v for r in a["rates"] for k, v in r.items() if k not in ("variant", "problems", "responses")}
        premise[name] = {(r["tokens"], k): v for r in a["premise"] for k, v in r.items() if k != "tokens"}
        decision[name] = {(r["responses"], k): v for r in a["decision"] for k, v in r.items() if k != "responses"}
    point = lambda s: float(str(s).split(" ")[0]) if isinstance(s, str) and s[:1].isdigit() else s  # noqa: E731
    out = {}
    for title, table in (("rates", rates), ("premise", premise), ("decision", decision)):
        df = pd.DataFrame(table)
        if df.empty:
            log(f"\n** {title} **: empty"); continue
        df.index = pd.MultiIndex.from_tuples(df.index)
        if title == "rates" and "base s0" in df and "base s1" in df:
            df["noise floor |s0 - s1|"] = [abs(point(a) - point(b)) if isinstance(point(a), float) and isinstance(point(b), float) else ""
                                          for a, b in zip(df["base s0"], df["base s1"])]
            df["noise floor |s0 - s1|"] = df["noise floor |s0 - s1|"].map(lambda v: round(v, 3) if v != "" else "")
        log(f"\n** {title} **")
        log(df.to_string())
        out[title] = df.reset_index().rename(columns={"level_0": "group", "level_1": "measure"}).to_dict("records")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--arms", default=None, help="comma list of model:seed, default base:0,base:1,opsa:0")
    ap.add_argument("--runs", default=str(ROOT / "runs"))
    ap.add_argument("--tag", default=None, help="run folder prefix (default: today's date)")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--skip-gpu-check", action="store_true")
    ap.add_argument("--rates-only", action="store_true", help="stage 1 and the rates table only; no token scoring or audit")
    args = ap.parse_args(argv)

    cfg = dict(CFG)
    if args.smoke:
        cfg.update(SMOKE)
    for item in args.set:
        k, _, v = item.partition("=")
        if k not in cfg:
            sys.exit(f"--set {item}: expected one of {', '.join(cfg)}=value")
        cfg[k] = type(cfg[k])(v)
    arms = ARMS
    if args.arms:
        names = {"base": BASE, "opsa": OPSA}
        arms = [(names.get(m, m), int(s)) for m, s in (a.split(":") for a in args.arms.split(","))]
    tag = args.tag or f"{datetime.date.today():%Y-%m-%d}"
    tag += "-smoke" if args.smoke else ""
    runs = Path(args.runs)
    runs.mkdir(parents=True, exist_ok=True)
    log = Log(runs / f"{tag}_sum_summary.log")
    log("config:", json.dumps(cfg), "| arms:", arms)

    if not args.skip_gpu_check:
        r = subprocess.run([sys.executable, "-c", "import torch; assert torch.cuda.is_available(), 'no GPU'; "
                            "p = torch.cuda.get_device_properties(0); print('GPU:', p.name, f'{p.total_memory / 2**30:.1f} GiB')"],
                           capture_output=True, text=True)
        log((r.stdout + r.stderr).strip())
        if r.returncode != 0:
            return 1

    run_dirs = {}
    for model, seed in arms:
        name = ("base" if model == BASE else "opsa" if model == OPSA else short(model)) + f" s{seed}"
        out = runs / f"{tag}_sum_{short(model)}_s{seed}"
        out.mkdir(parents=True, exist_ok=True)
        cfg_path = out / "config.json"
        this = dict(cfg, model=model, seed=seed)
        if cfg_path.exists() and json.loads(cfg_path.read_text()) != this:
            sys.exit(f"{cfg_path} holds a different config; use another --tag or delete the folder")
        cfg_path.write_text(json.dumps(this, indent=1))
        log(f"\n=== {name}: {out}")
        run(log, "rollout.py", out, "--model", model, "--dataset", "sum", "--n-problems", cfg["n_problems"],
            "--samples", cfg["samples"], "--max-tokens", cfg["max_tokens"], "--seed", seed, "--gpu-mem", cfg["gpu_mem"])
        if args.rates_only:
            import common
            import premise_audit
            rates = premise_audit.rates_table(common.read_jsonl(out / "rollouts.jsonl"))
            (out / "tables").mkdir(exist_ok=True)
            rates.to_csv(out / "tables" / "rates.csv", index=False)
            pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
            log("\n** Rates (95% CI over problem pairs) **"); log(rates.to_string(index=False))
            continue
        run(log, "premise_audit.py", out, "--pool-size", cfg["pool_size"], "--fraction", cfg["fraction"], "--seed", seed)
        run_dirs[name] = out

    if not run_dirs:
        log("\nrates only; no cross-arm audit summary")
        return 0
    summary = compare(run_dirs, log)
    (runs / f"{tag}_sum_summary.json").write_text(json.dumps(dict(config=cfg, arms=arms, runs={k: str(v) for k, v in run_dirs.items()},
                                                                 tables=summary), indent=1, default=str))
    log(f"\nsummary written to {runs / f'{tag}_sum_summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
