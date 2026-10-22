#!/usr/bin/env python
"""Stage 1 (GPU, vLLM): sample the student's responses at OPSA's rollout settings and grade them.

    python src/rollout.py --run <run dir> --model Qwen/Qwen3-1.7B --levels 3,4,5 --n-problems 60 --samples 8

T = 1.0 with no top-p or top-k (OPSA's training rollouts, run_opsa.sh), thinking disabled, at most --max-tokens
response tokens. Writes problems.jsonl (with the prompt token ids), rollouts.jsonl (response token ids, text,
vLLM's log-probability of each sampled token, correct) and rollout.meta.json. Skips if the meta file exists.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
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
    ap.add_argument("--levels", default="3,4,5")
    ap.add_argument("--n-problems", type=int, default=60)
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
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
    problems = common.load_math500([int(x) for x in args.levels.split(",")], args.n_problems, args.seed)
    for p in problems:
        p["prompt_ids"] = common.prompt_ids(tok, p["question"])
    max_prompt = max(len(p["prompt_ids"]) for p in problems)
    dtype = "bfloat16" if torch.cuda.get_device_capability()[0] >= 8 else "float16"   # T4: no native bf16

    # generation_config="vllm": ignore the model's suggested sampling defaults (Qwen3 ships T 0.6, top-p 0.95,
    # top-k 20); every setting below is explicit.
    llm = LLM(model=args.model, dtype=dtype, seed=args.seed, max_model_len=max_prompt + args.max_tokens + 8,
              gpu_memory_utilization=args.gpu_mem, enable_prefix_caching=True, generation_config="vllm")
    sps = [SamplingParams(n=args.samples, temperature=1.0, top_p=1.0, max_tokens=args.max_tokens,
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
            rows.append(dict(key=common.response_key(p["problem_id"], j), problem_id=p["problem_id"], sample_idx=j,
                             level=p["level"], gold=p["gold"], response_ids=ids, n_tokens=len(ids),
                             finish_reason=c.finish_reason, truncated=c.finish_reason == "length",
                             text=tok.decode(ids, skip_special_tokens=True), vllm_lp=lps))

    with Grader() as g:
        for r, ok in zip(rows, g.grade_many([(r["text"], r["gold"]) for r in rows])):
            r["correct"] = ok
        n_timeout = g.n_timeout

    common.write_jsonl_atomic(run / "problems.jsonl", problems)
    common.write_jsonl_atomic(run / "rollouts.jsonl", rows)
    pass1 = sum(r["correct"] for r in rows) / len(rows)
    per = {}
    for r in rows:
        per.setdefault(r["problem_id"], []).append(r["correct"])
    n_mixed = sum(0 < sum(v) < len(v) for v in per.values())
    n_all = sum(all(v) for v in per.values())
    meta = dict(stage="rollout", model=args.model, levels=args.levels, n_problems=len(problems), samples=args.samples,
                max_tokens=args.max_tokens, temperature=1.0, top_p=1.0, top_k=None, thinking=False, seed=args.seed,
                dtype=dtype, vllm=vllm.__version__, grader=grader_name(), grade_timeouts=n_timeout,
                end_ids=end_ids, end_added=n_end_added, n_responses=len(rows), pass1=pass1,
                problems_mixed=n_mixed, problems_all_correct=n_all, problems_none_correct=len(per) - n_mixed - n_all,
                truncated=sum(r["truncated"] for r in rows) / len(rows),
                mean_tokens=sum(r["n_tokens"] for r in rows) / len(rows),
                gen_seconds=round(gen_s), seconds=round(time.time() - t0),
                example_prompt=common.prompt_text(tok, problems[0]["question"]))
    common.write_json(run / "rollout.meta.json", meta)
    print(f"{len(rows)} responses, pass@1 {pass1:.3f}, truncated {meta['truncated']:.1%}, "
          f"mean {meta['mean_tokens']:.0f} tokens, {meta['seconds']}s")
    print(f"gate 0: {n_mixed} of {len(per)} problems mixed (some but not all answers correct), "
          f"{n_all} all correct, {len(per) - n_mixed - n_all} none correct")


if __name__ == "__main__":
    main()
