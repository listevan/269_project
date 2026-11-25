#!/usr/bin/env python
"""Stage 1 (GPU, vLLM): sample the student's responses at OPSA's rollout settings and grade them.

    python src/rollout.py --run <run dir> --model Qwen/Qwen3-1.7B --levels 3,4,5 --n-problems 60 --samples 8
    python src/rollout.py --run <run dir> --dataset sum --n-problems 284        # study 2: SUM, both variants

T = 1.0 with no top-p or top-k (OPSA's training rollouts, run_opsa.sh), thinking disabled, at most --max-tokens
response tokens. Writes problems.jsonl (with the prompt token ids), rollouts.jsonl (response token ids, text,
vLLM's log-probability of each sampled token, correct) and rollout.meta.json. Skips if the meta file exists.
With --dataset sum each problem comes as an unanswerable version and its solvable original (sum_data.py), the
prompt carries SUM's own instruction, and every row also records the variant and the outcome (refusal,
false_answer, no_answer; correct, wrong); `correct` then means the wanted behaviour: a refusal on the
unanswerable version, the right answer on the original.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import sum_data  # noqa: E402
from grade import Grader, grader_name  # noqa: E402


def sampled_logprobs(ids, logprobs):
    """vLLM's log-probability of each sampled token, None where it is missing (only a cross-check)."""
    out = []
    for i, d in zip(ids, logprobs or []):
        try:
            out.append(round(float(d[i].logprob), 5))
        except Exception:  # noqa: BLE001
            out.append(None)
    return out + [None] * (len(ids) - len(out))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--model", default=common.STUDENT)
    ap.add_argument("--dataset", choices=("math500", "sum"), default="math500")
    ap.add_argument("--levels", default="3,4,5", help="MATH-500 levels (ignored for sum)")
    ap.add_argument("--n-problems", type=int, default=60)
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--temperature", type=float, default=1.0, help="OPSA's rollouts use 1.0; lower values are the sampling control")
    ap.add_argument("--problem-ids", default=None, help="comma list: keep only these problem ids (the coverage audit)")
    ap.add_argument("--seed", type=int, default=0, help="sampling seed")
    ap.add_argument("--problem-seed", type=int, default=None,
                    help="seed of the problem subset (default: --seed); arms that must share problems share this")
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    args = ap.parse_args(argv)

    run = Path(args.run)
    run.mkdir(parents=True, exist_ok=True)
    if (run / "rollout.meta.json").exists():
        print(f"rollouts already done: {run / 'rollout.meta.json'}")
        return

    common.ensure_c_compiler()
    import torch
    import vllm
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model)
    end_ids = common.end_token_ids(tok)
    problem_seed = args.seed if args.problem_seed is None else args.problem_seed
    if args.dataset == "sum":
        problems = sum_data.load_sum(args.n_problems, problem_seed)
        instruction, sep = sum_data.SUM_INSTRUCTION, " "
    else:
        problems = common.load_math500([int(x) for x in args.levels.split(",")], args.n_problems, problem_seed)
        instruction, sep = common.INSTRUCTION, "\n"
    if args.problem_ids:
        keep = set(args.problem_ids.split(","))
        problems = [p for p in problems if p["problem_id"] in keep]
        if not problems:
            sys.exit("--problem-ids matched no problem")
    for p in problems:
        p["prompt_ids"] = common.prompt_ids(tok, p["question"], instruction, sep)
    max_prompt = max(len(p["prompt_ids"]) for p in problems)
    dtype = "bfloat16" if torch.cuda.get_device_capability()[0] >= 8 else "float16"   # T4: no native bf16

    # generation_config="vllm": ignore the model's suggested sampling defaults (Qwen3 ships T 0.6, top-p 0.95,
    # top-k 20); every setting below is explicit.
    llm = LLM(model=args.model, dtype=dtype, seed=args.seed, max_model_len=max_prompt + args.max_tokens + 8,
              gpu_memory_utilization=args.gpu_mem, enable_prefix_caching=True, generation_config="vllm")
    sps = [SamplingParams(n=args.samples, temperature=args.temperature, top_p=1.0, max_tokens=args.max_tokens,
                          stop_token_ids=end_ids, logprobs=0, seed=common.stable_seed("rollout", args.seed, p["problem_id"]))
           for p in problems]
    outs = llm.generate([{"prompt_token_ids": p["prompt_ids"]} for p in problems], sps)
    gen_s = time.time() - t0

    rows, n_end_added = [], 0
    for p, out in zip(problems, outs):
        for j, c in enumerate(out.outputs):
            ids = list(c.token_ids)
            lps = sampled_logprobs(ids, c.logprobs)
            if c.finish_reason == "stop" and (not ids or ids[-1] not in end_ids):
                ids.append(end_ids[0])   # keep the end token: OPSA trains on it like any other response token
                lps.append(None)
                n_end_added += 1
            row = dict(key=common.response_key(p["problem_id"], j), problem_id=p["problem_id"], sample_idx=j,
                       level=p["level"], gold=p["gold"], response_ids=ids, n_tokens=len(ids),
                       finish_reason=c.finish_reason, truncated=c.finish_reason == "length",
                       text=tok.decode(ids, skip_special_tokens=True), vllm_lp=lps)
            if args.dataset == "sum":
                row.update(pair=p["pair"], variant=p["variant"])
            rows.append(row)

    graded = [r for r in rows if r.get("variant") != "unanswerable"]   # a refusal gold is not a math answer
    with Grader() as g:
        verdict = dict(zip((r["key"] for r in graded), g.grade_many([(r["text"], r["gold"]) for r in graded])))
        n_timeout = g.n_timeout
    for r in rows:
        if args.dataset == "sum":
            r["outcome"] = sum_data.outcome(r["text"], r["variant"], verdict.get(r["key"], False))
            r["correct"] = sum_data.wanted(r["outcome"], r["variant"])
        else:
            r["correct"] = verdict[r["key"]]

    common.write_jsonl_atomic(run / "problems.jsonl", problems)
    common.write_jsonl_atomic(run / "rollouts.jsonl", rows)
    pass1 = sum(r["correct"] for r in rows) / len(rows)
    per = {}
    for r in rows:
        per.setdefault(r["problem_id"], []).append(r["correct"])
    n_mixed = sum(0 < sum(v) < len(v) for v in per.values())
    n_all = sum(all(v) for v in per.values())
    meta = dict(stage="rollout", model=args.model, dataset=args.dataset, instruction=instruction,
                levels=args.levels if args.dataset == "math500" else None, n_problems=len(problems), samples=args.samples,
                max_tokens=args.max_tokens, temperature=args.temperature, top_p=1.0, top_k=None, thinking=False, seed=args.seed,
                problem_seed=problem_seed,
                dtype=dtype, vllm=vllm.__version__, grader=grader_name(), grade_timeouts=n_timeout,
                end_ids=end_ids, end_added=n_end_added, n_responses=len(rows), pass1=pass1,
                problems_mixed=n_mixed, problems_all_correct=n_all, problems_none_correct=len(per) - n_mixed - n_all,
                truncated=sum(r["truncated"] for r in rows) / len(rows),
                mean_tokens=sum(r["n_tokens"] for r in rows) / len(rows),
                gen_seconds=round(gen_s), seconds=round(time.time() - t0),
                example_prompt=common.prompt_text(tok, problems[0]["question"], instruction, sep))
    common.write_json(run / "rollout.meta.json", meta)
    print(f"{len(rows)} responses, pass@1 {pass1:.3f}, truncated {meta['truncated']:.1%}, "
          f"mean {meta['mean_tokens']:.0f} tokens, {meta['seconds']}s")
    print(f"gate 0: {n_mixed} of {len(per)} problems mixed (some but not all answers correct), "
          f"{n_all} all correct, {len(per) - n_mixed - n_all} none correct")
    if args.dataset == "sum":
        for variant in ("unanswerable", "answerable"):
            sub = [r for r in rows if r["variant"] == variant]
            counts = {o: sum(r["outcome"] == o for r in sub) for o in sorted({r["outcome"] for r in sub})}
            print(f"{variant}: " + ", ".join(f"{o} {n / len(sub):.3f}" for o, n in counts.items()))


if __name__ == "__main__":
    main()
