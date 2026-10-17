#!/usr/bin/env python
"""Stage 2 (GPU, transformers): score every response token, apply the four rules, sample the replay positions,
and draw the redrawn tokens.

    python src/score.py --run <run dir> [--teacher Qwen/Qwen3-4B] [--positions 64] [--redraws 8]

1. Student and teacher log-probabilities of every response token (teacher-forced, fp32 log-softmax at T = 1),
   plus the student's entropy (OPSA scales its penalty by it). -> scores.jsonl
2. The rules of select_rules.py on pools of --pool-size responses. -> selected.jsonl (nsr is implicit: every
   token of a failed response)
3. --positions positions drawn uniformly from each rule's set; for each, --redraws tokens drawn from the
   student's distribution at that position with the kept token removed. -> positions.jsonl
Each output is skipped if it already exists, so a disconnect only repeats the unfinished step.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import select_rules as sr  # noqa: E402


def device_dtype():
    """GPU in bf16 where it is native (compute capability 8+), fp16 on older GPUs, fp32 on CPU (laptop checks)."""
    import torch

    if torch.cuda.is_available():
        return "cuda", torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    return "cpu", torch.float32


def load_model(name):
    from transformers import AutoModelForCausalLM

    dev, dtype = device_dtype()
    try:
        model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype, attn_implementation="sdpa")
    except TypeError:   # transformers < 4.56 calls it torch_dtype
        model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype, attn_implementation="sdpa")
    model = model.to(dev).eval()
    head = model.get_output_embeddings()
    assert getattr(head, "bias", None) is None, "output layer has a bias; token_scores assumes none"
    return model, head.weight.float()


def hidden(model, ids):
    import torch

    with torch.no_grad():
        return model.base_model(input_ids=torch.tensor([ids], device=model.device)).last_hidden_state[0]


def token_scores(model, head_w, prompt_ids, resp_ids, entropy=False, chunk=1024):
    """log p(y_t | x, y_<t) for every response token (and the entropy of each next-token distribution)."""
    import torch

    n = len(prompt_ids)
    h = hidden(model, prompt_ids + resp_ids)[n - 1:n - 1 + len(resp_ids)]
    tgt = torch.tensor(resp_ids, device=h.device)
    lps, ents = [], []
    with torch.no_grad():
        for i in range(0, h.shape[0], chunk):
            logp = torch.log_softmax(h[i:i + chunk].float() @ head_w.T, dim=-1)
            lps.append(logp.gather(1, tgt[i:i + chunk, None])[:, 0])
            if entropy:
                ents.append(-(logp.exp() * logp).sum(-1))
    lp = [round(v, 5) for v in torch.cat(lps).tolist()]
    return (lp, [round(v, 5) for v in torch.cat(ents).tolist()]) if entropy else (lp, None)


def score_all(name, problems, rollouts, entropy):
    import torch

    model, head_w = load_model(name)
    out, t0 = {}, time.time()
    for i, r in enumerate(rollouts, 1):
        out[r["key"]] = token_scores(model, head_w, problems[r["problem_id"]]["prompt_ids"], r["response_ids"], entropy)
        if i % 100 == 0:
            print(f"   {name}: {i}/{len(rollouts)} responses scored, {time.time() - t0:.0f}s", flush=True)
    del model, head_w
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def redraw(model, head_w, prompt_ids, resp_ids, ts, k, seeds, tok, top=5):
    """At each position t: the kept token's probability and the entropy (T = 1), the top alternatives, and k
    tokens drawn from the distribution with the kept token removed (softmax of the logits with that logit set
    to -inf, so a kept probability of 1.0 in fp32 is still fine)."""
    import torch

    n = len(prompt_ids)
    h = hidden(model, prompt_ids + resp_ids)
    out = []
    with torch.no_grad():
        for t, seed in zip(ts, seeds):
            logits = h[n - 1 + t].float() @ head_w.T
            logp = torch.log_softmax(logits, -1)
            y = resp_ids[t]
            masked = logits.clone()
            masked[y] = float("-inf")
            q = torch.softmax(masked, -1)
            g = torch.Generator(device="cpu").manual_seed(seed)
            alts = torch.multinomial(q.cpu(), k, replacement=True, generator=g).tolist()
            top_p, top_i = torch.topk(q, top)
            out.append(dict(p_kept=float(logp[y].exp()), lp_kept=float(logp[y]), entropy=float(-(logp.exp() * logp).sum()),
                            alt_ids=alts, alt_strs=[tok.decode([a]) for a in alts],
                            top_alts=[(int(i), tok.decode([int(i)]), round(float(p) * (1 - float(logp[y].exp())), 5))
                                      for p, i in zip(top_p, top_i)]))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--teacher", default=common.TEACHER)
    ap.add_argument("--pool-size", type=int, default=16, help="responses per selection pool (OPSA: 16 per rank)")
    ap.add_argument("--fraction", type=float, default=0.2, help="OPSA's token fraction f")
    ap.add_argument("--positions", type=int, default=64, help="replay positions drawn per rule")
    ap.add_argument("--redraws", type=int, default=8, help="redrawn tokens per position")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    from transformers import AutoTokenizer

    run = Path(args.run)
    rmeta = common.read_json(run / "rollout.meta.json")
    student = rmeta["model"]
    problems = {p["problem_id"]: p for p in common.read_jsonl(run / "problems.jsonl")}
    rollouts = common.read_jsonl(run / "rollouts.jsonl")
    tok = AutoTokenizer.from_pretrained(student)
    t_tok = AutoTokenizer.from_pretrained(args.teacher)
    if tok.get_vocab() != t_tok.get_vocab():
        sys.exit(f"{args.teacher} does not share {student}'s vocabulary; its scores would not line up token by token")
    end_ids = common.end_token_ids(tok)
    t0 = time.time()

    if not (run / "scores.jsonl").exists():
        s = score_all(student, problems, rollouts, entropy=True)
        t = score_all(args.teacher, problems, rollouts, entropy=False)
        rows = [dict(key=r["key"], problem_id=r["problem_id"], correct=r["correct"], lp_s=s[r["key"]][0],
                     ent_s=s[r["key"]][1], lp_t=t[r["key"]][0], tokens=[tok.decode([i]) for i in r["response_ids"]])
                for r in rollouts]
        common.write_jsonl_atomic(run / "scores.jsonl", rows)
        print(f"scored {len(rows)} responses with {student} and {args.teacher}, {time.time() - t0:.0f}s")
    scores = common.read_jsonl(run / "scores.jsonl")

    sets, adv, pools = sr.selected_sets(scores, args.pool_size, args.fraction, args.seed)
    common.write_jsonl_atomic(run / "selected.jsonl", [
        dict(key=r["key"], problem_id=r["problem_id"], correct=r["correct"], n=len(r["lp_s"]),
             opsa=sets["opsa"][r["key"]], opsa_adv=[adv[r["key"]][t] for t in sets["opsa"][r["key"]]],
             random=sets["random"][r["key"]], teacher_neg=sets["teacher_neg"][r["key"]]) for r in scores])

    if not (run / "positions.jsonl").exists():
        picked = sr.sample_positions(sets, args.positions, args.seed)
        by_key = {r["key"]: r for r in rollouts}
        sc = {r["key"]: r for r in scores}
        model, head_w = load_model(student)
        out = []
        for key in sorted({p["key"] for p in picked}):
            ps = [p for p in picked if p["key"] == key]
            r = by_key[key]
            red = redraw(model, head_w, problems[r["problem_id"]]["prompt_ids"], r["response_ids"], [p["t"] for p in ps],
                         args.redraws, [common.stable_seed("redraw", args.seed, p["pos_id"]) for p in ps], tok)
            for p, d in zip(ps, red):
                t = p["t"]
                out.append(dict(p, problem_id=r["problem_id"], token_id=r["response_ids"][t], token_str=sc[key]["tokens"][t],
                                is_end=r["response_ids"][t] in end_ids, n_tokens=r["n_tokens"], response_correct=r["correct"],
                                lp_s=sc[key]["lp_s"][t], lp_t=sc[key]["lp_t"][t], ent_s=sc[key]["ent_s"][t],
                                opsa_adv=adv.get(key, {}).get(t), in_sets=sr.in_sets(sets, key, t), **d))
        common.write_jsonl_atomic(run / "positions.jsonl", out)
        print(f"{len(out)} replay positions ({args.positions} per rule, shared ones replayed once), {args.redraws} redraws each")

    common.write_json(run / "score.meta.json", dict(
        stage="score", student=student, teacher=args.teacher, pool_size=args.pool_size, fraction=args.fraction,
        positions_per_rule=args.positions, redraws=args.redraws, seed=args.seed, n_pools=len(pools),
        selected_tokens={m: sum(len(v) for v in sets[m].values()) for m in sets},
        total_tokens=sum(len(r["lp_s"]) for r in scores), seconds=round(time.time() - t0)))


if __name__ == "__main__":
    main()
