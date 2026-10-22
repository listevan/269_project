#!/usr/bin/env python
"""Stage 3 (GPU, vLLM): replay each sampled position with its token kept and with it redrawn, and grade.

    python src/replay.py --run <run dir> [--kept 8] [--dup-fraction 0.25]

For a position t of response y to prompt x:
  kept      --kept continuations of x + y[:t] + [y_t]
  redrawn   one continuation of x + y[:t] + [a_j] per redrawn token a_j (score.py drew them from the student's
            distribution at t with y_t removed, which is where a negative update on y_t moves its probability)
  dup       a second, independent kept pass on a --dup-fraction subset of positions: the replay noise floor
Continuations reuse the rollout settings (T = 1, no top-p or top-k) and the rollout's total token budget, so a
continuation is cut where the original response would have been. Prefixes are token ids, never re-tokenized
text. A chosen end token ends the response at t; it is graded as is without sampling.
Resumable: positions are appended to replays.jsonl.partial a chunk at a time and skipped on a rerun.
"""
import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
from grade import Grader, grader_name  # noqa: E402


def is_dup(pos_id, fraction):
    return common.stable_seed("dup", pos_id) % 10000 < round(fraction * 10000)


def build_requests(pos, resp_ids, end_ids, kept, max_tokens, dup_fraction, seed):
    """One request per continuation group: {pos_id, kind, j, resp_prefix, n, max_tokens, seed, fixed}.
    resp_prefix is the response part up to and including the token at t (kept or redrawn); fixed means
    nothing is sampled (the token ends the response, or the budget is used up) and the prefix is graded as is."""
    t = pos["t"]
    base = resp_ids[:t]
    groups = [("kept", 0, resp_ids[t], kept, common.stable_seed("kept", seed, pos["pos_id"]))]
    if is_dup(pos["pos_id"], dup_fraction):
        groups.append(("dup", 0, resp_ids[t], kept, common.stable_seed("dup", seed, pos["pos_id"])))
    for j, a in enumerate(pos["alt_ids"]):
        groups.append(("redrawn", j, a, 1, common.stable_seed("redrawn", seed, pos["pos_id"], j)))
    reqs = []
    for kind, j, tok_id, n, s in groups:
        prefix = base + [tok_id]
        budget = max_tokens - len(prefix)
        reqs.append(dict(pos_id=pos["pos_id"], key=pos["key"], problem_id=pos["problem_id"], kind=kind, j=j,
                         token_id=tok_id, resp_prefix=prefix, n=n,
                         max_tokens=budget, seed=s, fixed=tok_id in end_ids or budget <= 0))
    return reqs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--kept", type=int, default=8, help="kept continuations per position")
    ap.add_argument("--dup-fraction", type=float, default=0.25)
    ap.add_argument("--chunk", type=int, default=32, help="positions per vLLM call (and per checkpoint)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    args = ap.parse_args(argv)

    run = Path(args.run)
    final, partial = run / "replays.jsonl", run / "replays.jsonl.partial"
    if final.exists():
        print(f"replays already done: {final}")
        return

    common.ensure_c_compiler()
    import torch
    import vllm
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    rmeta = common.read_json(run / "rollout.meta.json")
    model, max_tokens = rmeta["model"], rmeta["max_tokens"]
    problems = {p["problem_id"]: p for p in common.read_jsonl(run / "problems.jsonl")}
    rollouts = {r["key"]: r for r in common.read_jsonl(run / "rollouts.jsonl")}
    positions = common.read_jsonl(run / "positions.jsonl")
    done = {r["pos_id"] for r in common.read_jsonl(partial)} if partial.exists() else set()
    todo = [p for p in positions if p["pos_id"] not in done]
    print(f"{len(positions)} positions, {len(done)} already replayed, {len(todo)} to go")

    tok = AutoTokenizer.from_pretrained(model)
    end_ids = common.end_token_ids(tok)
    t0 = time.time()
    if todo:
        max_prompt = max(len(p["prompt_ids"]) for p in problems.values())
        dtype = "bfloat16" if torch.cuda.get_device_capability()[0] >= 8 else "float16"   # T4: no native bf16
        llm = LLM(model=model, dtype=dtype, seed=args.seed, max_model_len=max_prompt + max_tokens + 8,
                  gpu_memory_utilization=args.gpu_mem, enable_prefix_caching=True, generation_config="vllm")
        with Grader() as grader:
            for c in range(0, len(todo), args.chunk):
                chunk = todo[c:c + args.chunk]
                reqs = [q for p in chunk for q in build_requests(
                    p, rollouts[p["key"]]["response_ids"], end_ids, args.kept, max_tokens, args.dup_fraction, args.seed)]
                live = [q for q in reqs if not q["fixed"]]
                outs = llm.generate(
                    [{"prompt_token_ids": problems[q["problem_id"]]["prompt_ids"] + q["resp_prefix"]} for q in live],
                    [SamplingParams(n=q["n"], temperature=1.0, top_p=1.0, max_tokens=q["max_tokens"], stop_token_ids=end_ids,
                                    seed=q["seed"]) for q in live], use_tqdm=False)
                conts = {id(q): [(list(o.token_ids), o.finish_reason) for o in out.outputs] for q, out in zip(live, outs)}
                rows, pairs = [], []
                for q in reqs:
                    gold = rollouts[q["key"]]["gold"]
                    for k, (ids, fin) in enumerate(conts.get(id(q), [([], "fixed")] * q["n"])):
                        rows.append(dict(pos_id=q["pos_id"], kind=q["kind"], j=q["j"] if q["kind"] == "redrawn" else k,
                                         token_id=q["token_id"], n_cont=len(ids), finish_reason=fin,
                                         truncated=fin == "length" or (q["fixed"] and q["max_tokens"] <= 0),
                                         cont_text=tok.decode(ids, skip_special_tokens=True)))
                        pairs.append((tok.decode(q["resp_prefix"] + ids, skip_special_tokens=True), gold))
                for r, ok in zip(rows, grader.grade_many(pairs)):
                    r["correct"] = ok
                common.append_jsonl(partial, rows)
                print(f"   {min(c + args.chunk, len(todo))}/{len(todo)} positions, {len(rows)} continuations, "
                      f"{time.time() - t0:.0f}s", flush=True)
            n_timeout = grader.n_timeout
    else:
        n_timeout = 0

    if partial.exists():
        os.replace(partial, final)
    else:
        final.write_text("")
    rows = common.read_jsonl(final)
    common.write_json(run / "replay.meta.json", dict(
        stage="replay", model=model, kept=args.kept, dup_fraction=args.dup_fraction, seed=args.seed,
        max_tokens=max_tokens, n_positions=len(positions), n_continuations=len(rows), grader=grader_name(),
        grade_timeouts=n_timeout, vllm=vllm.__version__, seconds=round(time.time() - t0)))
    print(f"done: {len(rows)} continuations for {len(positions)} positions, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
