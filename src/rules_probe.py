#!/usr/bin/env python
"""Experiment 3's logit read (GPU, transformers): the probability of the correct and of the dominant wrong answer
at the decision point of each sampled response.

    python src/rules_probe.py --run <eval dir> --dominant <dominant.json> [--adapter <dir>] [--split new_numbers]

Two reads of p(answer = N) for the item's correct answer and its dominant wrong answer (the base model's most
common wrong answer to the item, dominant.json), under the checkpoint:
  own context    for every response of the split that reaches a \\boxed{: the prompt plus the response up to and
                 including the last "\\boxed" (the model's own reasoning; mostly re-reads the sampled answer)
  fixed context  once per item: the prompt plus "\\boxed" as the first assistant tokens (no reasoning; the model's
                 belief about the answer before any deliberation, a read that can move below one sample in 16)
In both, p(N) = p("{" + N's digit tokens | context) x p(the next token is not a digit | context, "{", N). Qwen
tokenizes digits one per token but merges "{" with a following "-" ("{-3"), so the candidate carries the brace
and the context stops before it; "}" alone is a rare token (the model writes "}\n"), so nothing after the digits
is scored. Writes probe.jsonl (one row per response and per item) and probe.json (means per operator and
overall for both reads). Skips if probe.json exists.
"""
import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import rules_data as rd  # noqa: E402
from neg_train import pad_right  # noqa: E402


def candidates(tok, r, dominant):
    cands = {"correct": number_ids(tok, r["answer"])}
    dom = dominant.get(r["item_id"])
    if dom is not None:
        cands["dominant"] = number_ids(tok, dom)
    return cands


def contexts(rows, tok, dominant):
    """[(row index, context ids, {name: candidate ids})] for rows whose response reached a box: the own-context read."""
    out = []
    for i, r in enumerate(rows):
        k = r["text"].rfind("\\boxed{")
        if k < 0:
            continue
        ctx = tok(rd.prompt_text_for(tok, r, with_def=r["split"] == "with_def") + r["text"][:k + len("\\boxed")],
                  add_special_tokens=False)["input_ids"]
        out.append((i, ctx, candidates(tok, r, dominant)))
    return out


def fixed_contexts(rows, tok, dominant):
    """[(item_id, context ids, {name: candidate ids})], one per item: the prompt and "\\boxed" as the first tokens."""
    seen, out = set(), []
    for r in rows:
        if r["item_id"] in seen:
            continue
        seen.add(r["item_id"])
        ctx = tok(rd.prompt_text_for(tok, r, with_def=r["split"] == "with_def") + "\\boxed", add_special_tokens=False)["input_ids"]
        out.append((r["item_id"], ctx, candidates(tok, r, dominant)))
    return out


def number_ids(tok, n):
    """Tokens of "{N": the brace (merged with a minus sign for negative N), then one token per digit."""
    ids = tok("{" + str(n), add_special_tokens=False)["input_ids"]
    text = [tok.decode([t]) for t in ids]
    assert "".join(text) == "{" + str(n) and all(len(t) == 1 for t in text[1:]), f"unexpected tokens for {n}: {text}"
    return ids


def digit_ids(tok):
    ids = [tok(str(d), add_special_tokens=False)["input_ids"] for d in range(10)]
    assert all(len(x) == 1 for x in ids), "digits are not single tokens"
    return [x[0] for x in ids]


def candidate_logprobs(model, jobs, pad_id, digits, batch=16):
    """jobs: [(context ids, number ids)] -> log p(number | context) each: the number's tokens, then no digit."""
    import torch

    out = []
    dig = torch.tensor(digits)
    for i in range(0, len(jobs), batch):
        chunk = jobs[i:i + batch]
        ids, mask = pad_right([c + d for c, d in chunk], pad_id, model.device)
        with torch.no_grad():
            logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits
            for j, (c, d) in enumerate(chunk):
                logp = torch.log_softmax(logits[j, len(c) - 1:len(c) + len(d)].float(), dim=-1)   # len(d) + 1 positions
                lp_digits = logp[:-1].gather(1, torch.tensor(d, device=logp.device)[:, None]).sum()
                p_more = logp[-1, dig.to(logp.device)].exp().sum().clamp(max=1 - 1e-6)
                out.append(float(lp_digits + torch.log1p(-p_more)))
        del logits
    return out


def summarize(probes):
    """Means of p(correct) and p(dominant) per operator and overall. probes: rows with sym, lp_correct, lp_dominant."""
    per = {}
    for p in probes:
        per.setdefault(p["sym"], []).append(p)
    def stats(ps):
        pc = [math.exp(p["lp_correct"]) for p in ps]
        pd_ = [math.exp(p["lp_dominant"]) for p in ps if p.get("lp_dominant") is not None]
        return dict(n=len(ps), p_correct=round(float(np.mean(pc)), 4), p_correct_median=round(float(np.median(pc)), 4),
                    p_dominant=round(float(np.mean(pd_)), 4) if pd_ else None, n_dominant=len(pd_))
    return dict(overall=stats([p for ps in per.values() for p in ps]), per_sym={s: stats(ps) for s, ps in sorted(per.items())})


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="a rules_eval run; probe files are written beside its rollouts")
    ap.add_argument("--dominant", required=True, help="dominant.json: {item_id: dominant wrong answer}")
    ap.add_argument("--model", default=common.STUDENT)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--split", default="new_numbers")
    ap.add_argument("--batch", type=int, default=8, help="sequences per forward; 16 made the allocator thrash at 512-token responses")
    args = ap.parse_args(argv)

    run = Path(args.run)
    if (run / "probe.json").exists():
        print(f"probe already done: {run / 'probe.json'}")
        return
    common.ensure_c_compiler()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model)
    rows = [r for r in common.read_jsonl(run / "rollouts.jsonl") if r["split"] == args.split]
    dominant = {k: v for k, v in common.read_json(args.dominant).items() if v is not None}
    jobs = contexts(rows, tok, dominant)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter).eval()
    fixed = fixed_contexts(rows, tok, dominant)
    digits = digit_ids(tok)

    def score(job_list, tag):
        flat = [(ctx, d) for _, ctx, cands in job_list for d in cands.values()]
        lps = candidate_logprobs(model, flat, tok.pad_token_id, digits, args.batch)
        out, k = [], 0
        for who, ctx, cands in job_list:
            r = rows[who] if tag == "own" else next(x for x in rows if x["item_id"] == who)
            rec = dict(read=tag, item_id=r["item_id"], sym=r["sym"], sample_idx=r["sample_idx"] if tag == "own" else None,
                       lp_correct=None, lp_dominant=None)
            for name in cands:
                rec["lp_" + name] = round(lps[k], 4)
                k += 1
            out.append(rec)
        return out

    own, fix = score(jobs, "own"), score(fixed, "fixed")
    common.write_jsonl_atomic(run / "probe.jsonl", own + fix)
    summary = dict(own=summarize(own), fixed=summarize(fix), split=args.split, adapter=args.adapter, n_rows=len(rows),
                   n_probed=len(jobs), n_items=len(fix), seconds=round(time.time() - t0))
    summary["overall"] = summary["own"]["overall"]      # kept for older readers
    summary["per_sym"] = summary["own"]["per_sym"]
    common.write_json(run / "probe.json", summary)
    print(f"probe: own context {len(jobs)} of {len(rows)} responses reached a box, p(correct) {summary['own']['overall']['p_correct']:.3f}, "
          f"p(dominant) {summary['own']['overall']['p_dominant']}; fixed context {len(fix)} items, p(correct) "
          f"{summary['fixed']['overall']['p_correct']:.3f}, p(dominant) {summary['fixed']['overall']['p_dominant']}; {summary['seconds']}s")


if __name__ == "__main__":
    main()
