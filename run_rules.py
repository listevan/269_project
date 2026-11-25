#!/usr/bin/env python
"""Experiment 3 on this machine: learning a hidden rule under negative-only training (the proposal's main controlled study).

    uv run run_rules.py --smoke          # 3 operators, 2 training steps, 3 arms: about 15 minutes
    uv run run_rules.py                  # the study: about 12 operators, 7 arms x 60 steps, 16-sample evaluations

Stages (each skips when its output exists, so a stopped run resumes):
  candidates  the base model's correct rate on every candidate symbol/definition (rules_data.CANDIDATES, 8 samples on
              the new_numbers items); pick_operators keeps about 12 spanning near 0% to about 50% -> operators.json
  items       train (96 pairs) and the four evaluation splits (24 items each) per operator -> items.jsonl
  base        base A (16 samples, every split; the reference batch), base B (a second batch: the frozen-model control),
              and base rollouts on the train prompts for the teacher-signal decile audit (teacher_audit.py)
  probe       p(correct answer) and p(dominant wrong answer) at the box, read from the logits (rules_probe.py)
  train       each arm of neg_train.py (opsa, random, teacher_neg, teacher_both, sft, teacher_sel, answer_key) as LoRA,
              then its evaluation (rules_eval.py with the adapter merged) and probe
  analysis    rules_analysis.py tables -> runs/<tag>_rules/tables/
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

# lr: 1e-4 drove the OPSA arm into capped, answerless responses by step 24 of the 2026-11-24 run (its log is kept as
# train_opsa_lr1e-4_collapsed); 2e-5 x 60 steps moves the weights about as far as 12 steps did at 1e-4, where the
# effects were already large (correct share 0.20 -> 0.07) but responses still finished.
CFG = dict(cand_items=24, cand_samples=8, n_ops=12, max_rate=0.55, operators="", n_train=96, n_eval=24, eval_samples=16,
           audit_samples=2, max_tokens=512, steps=60, prompts_per_step=128, lr=2e-5, lora_r=16, lora_alpha=32, pool_size=16,
           fraction=0.2, adv_clip=5.0, save_steps="30", arms="opsa,random,teacher_neg,teacher_both,sft,teacher_sel,answer_key",
           seeds="0", gen_batch=64, score_batch=8, train_batch=4, adapter_mode="lora", gpu_mem=0.85)
SMOKE = dict(cand_items=4, cand_samples=2, n_ops=3, n_train=8, n_eval=4, eval_samples=2, audit_samples=1, max_tokens=256, steps=2,
             prompts_per_step=32, save_steps="1", arms="opsa,teacher_both,sft", gen_batch=32)


class Log:
    def __init__(self, path):
        self.f = open(path, "a")
        self.f.write(f"\n===== {datetime.datetime.now():%Y-%m-%d %H:%M:%S} {' '.join(sys.argv)}\n")

    def __call__(self, *parts, end="\n"):
        text = " ".join(str(p) for p in parts) + end
        print(text, end="", flush=True)
        self.f.write(text)
        self.f.flush()


def run(log, script, *args):
    cmd = [sys.executable, "-u", str(ROOT / "src" / script), *map(str, args)]
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


def write_items(path, items):
    import common
    common.write_jsonl_atomic(path, items)


def evaluate(log, cfg, out, items, seed, splits, samples, adapter=None, keep_ids=False):
    args = ["--run", out, "--items", items, "--splits", splits, "--samples", samples, "--max-tokens", cfg["max_tokens"],
            "--seed", seed, "--gpu-mem", cfg["gpu_mem"]]
    if adapter:
        args += ["--adapter", adapter, "--adapter-mode", cfg["adapter_mode"]]
    if keep_ids:
        args += ["--keep-ids"]
    run(log, "rules_eval.py", *args)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--runs", default=str(ROOT / "runs"))
    ap.add_argument("--tag", default=None)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--stop-after", choices=("candidates", "base", "audit", "train"), default=None)
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
    tag = (args.tag or f"{datetime.date.today():%Y-%m-%d}") + ("-smoke" if args.smoke else "")
    runs = Path(args.runs)
    runs.mkdir(parents=True, exist_ok=True)
    log = Log(runs / f"{tag}_rules.log")
    log("config:", json.dumps(cfg))
    if not args.skip_gpu_check:
        r = subprocess.run([sys.executable, "-c", "import torch; assert torch.cuda.is_available(), 'no GPU'; "
                            "p = torch.cuda.get_device_properties(0); print('GPU:', p.name, f'{p.total_memory / 2**30:.1f} GiB')"],
                           capture_output=True, text=True)
        log((r.stdout + r.stderr).strip())
        if r.returncode != 0:
            return 1

    import common
    import rules_analysis as ra
    import rules_data as rd

    pre = runs / f"{tag}_rules"
    pre.mkdir(parents=True, exist_ok=True)

    # --- candidates -> operators.json
    ops_path = pre / "operators.json"
    if ops_path.exists():
        ops = common.read_json(ops_path)
        log(f"operators (existing): {''.join(ops['syms'])}")
    else:
        if cfg["operators"]:
            syms = [s for s in cfg["operators"].split(",") if s]
            ops = dict(syms=syms, rates=None, source="--set operators")
        else:
            log("\n=== candidates: base correct rate per symbol/definition")
            cand_items = [it for s in rd.BY_SYM for it in rd.make_items(s, 0, cfg["cand_items"])["new_numbers"]]
            write_items(pre / "cand_items.jsonl", cand_items)
            out = evaluate(log, cfg, pre / "candidates", pre / "cand_items.jsonl", 90, "new_numbers", cfg["cand_samples"])
            rows = common.read_jsonl(out / "rollouts.jsonl")
            per = {}
            for r in rows:
                per.setdefault(r["sym"], []).append(r)
            rates = {s: round(sum(r["correct"] for r in v) / len(v), 4) for s, v in per.items()}
            detail = {}
            for s, v in per.items():
                dom = ra.dominant_map(v)
                detail[s] = dict(correct=rates[s], habit=round(sum(r["got"] is not None and r["got"] == r["habit"] for r in v) / len(v), 4),
                                 no_answer=round(sum(r["got"] is None for r in v) / len(v), 4),
                                 truncated=round(sum(r["truncated"] for r in v) / len(v), 4),
                                 mean_tokens=round(sum(r["n_tokens"] for r in v) / len(v), 1),
                                 dominant_wrong_share=round(sum(r["got"] is not None and r["got"] == dom.get(r["item_id"]) for r in v) / len(v), 4))
            syms = rd.pick_operators(rates, cfg["n_ops"], cfg["max_rate"])
            ops = dict(syms=syms, rates=rates, detail=detail, source="candidates")
            for s in sorted(rates, key=rates.get):
                d = detail[s]
                log(f"  {s:3s} {rd.BY_SYM[s]['expr']:12s} correct {d['correct']:.3f}  habit {d['habit']:.3f}  dominant {d['dominant_wrong_share']:.3f}  "
                    f"no answer {d['no_answer']:.3f}  truncated {d['truncated']:.3f}  tokens {d['mean_tokens']:.0f}{'  <- kept' if s in syms else ''}")
        common.write_json(ops_path, ops)
        log(f"operators: {''.join(ops['syms'])}")
    syms = ops["syms"]
    if len(syms) < 2:
        sys.exit("fewer than 2 operators at or below max_rate; lower the bar or add candidates")
    if args.stop_after == "candidates":
        return 0

    # --- items
    items_path = pre / "items.jsonl"
    if not items_path.exists():
        write_items(items_path, rd.all_items(syms, cfg["n_train"], cfg["n_eval"]))
    splits = ",".join(rd.SPLITS)

    # --- base batches, audit, probe
    log("\n=== base A (reference batch)")
    base_a = evaluate(log, cfg, pre / "base_A", items_path, 100, splits, cfg["eval_samples"])
    log("\n=== base B (frozen-model control)")
    base_b = evaluate(log, cfg, pre / "base_B", items_path, 101, splits, cfg["eval_samples"])
    dom_path, dom_b_path = pre / "dominant.json", pre / "dominant_B.json"
    if not dom_path.exists():
        common.write_json(dom_path, ra.dominant_map(common.read_jsonl(base_a / "rollouts.jsonl")))
    if not dom_b_path.exists():   # base A's own probe uses base B's map (a batch's own most common wrong answer is inflated)
        common.write_json(dom_b_path, ra.dominant_map(common.read_jsonl(base_b / "rollouts.jsonl")))
    if args.stop_after == "base":
        return 0
    log("\n=== base rollouts on the train prompts, teacher-signal audit")
    base_train = evaluate(log, cfg, pre / "base_train", items_path, 102, "train", cfg["audit_samples"], keep_ids=True)
    run(log, "teacher_audit.py", "--run", base_train, "--out", pre / "audit", "--pool-size", cfg["pool_size"], "--fraction", cfg["fraction"])
    log("\n=== probe: base A")
    run(log, "rules_probe.py", "--run", base_a, "--dominant", dom_b_path)
    log("\n=== probe: base B")
    run(log, "rules_probe.py", "--run", base_b, "--dominant", dom_path)
    if args.stop_after == "audit":
        return 0

    # --- arms
    runs_by_arm, train_dirs = {"base_A": base_a, "base_B": base_b}, {}
    arms = [a for a in cfg["arms"].split(",") if a]
    seeds = [int(s) for s in cfg["seeds"].split(",") if s]
    for seed in seeds:
        for i, arm in enumerate(arms):
            name = arm if len(seeds) == 1 else f"{arm}_s{seed}"
            log(f"\n=== train {name}")
            tdir = pre / f"train_{name}"
            run(log, "neg_train.py", "--run", tdir, "--arm", arm, "--items", items_path, "--steps", cfg["steps"],
                "--prompts-per-step", cfg["prompts_per_step"], "--max-tokens", cfg["max_tokens"], "--pool-size", cfg["pool_size"],
                "--fraction", cfg["fraction"], "--adv-clip", cfg["adv_clip"], "--lr", cfg["lr"], "--lora-r", cfg["lora_r"],
                "--lora-alpha", cfg["lora_alpha"], "--gen-batch", cfg["gen_batch"], "--score-batch", cfg["score_batch"],
                "--train-batch", cfg["train_batch"], "--save-steps", cfg["save_steps"], "--seed", seed)
            log(f"\n=== evaluate {name}")
            edir = evaluate(log, cfg, pre / f"eval_{name}", items_path, 200 + 10 * seed + i, splits, cfg["eval_samples"],
                            adapter=tdir / "adapter_final")
            run(log, "rules_probe.py", "--run", edir, "--dominant", dom_path, "--adapter", tdir / "adapter_final")
            runs_by_arm[name], train_dirs[name] = edir, tdir
    if args.stop_after == "train":
        return 0

    log("\n=== analysis")
    ra.analyze(runs_by_arm, items_path, pre / "tables", show=log, train_dirs=train_dirs)
    log(f"\ndone: {pre}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
