#!/usr/bin/env python
"""Experiment 3 training (GPU, transformers + PEFT): one arm of experiment 4's framework on the hidden-rule prompts.

    python src/neg_train.py --run <dir> --arm opsa --items <items.jsonl> --steps 60 --prompts-per-step 128

Every on-policy arm samples one response per prompt from the current policy (T = 1, no top-p/top-k, thinking off,
at most --max-tokens tokens), drops the responses that hit the cap (masked from every arm, check 14), scores the
kept ones token by token, turns one rule into per-token advantages A_t, and takes one AdamW step on the LoRA
weights with the token-mean loss  L = -sum_t A_t log p(y_t) / N  over the kept responses' N tokens. Pools of 16
responses (to different prompts, as OPSA's selector pools one rank's batch):
  opsa          the lowest-20% student log-prob tokens of the pool, entropy-ranked advantages -0.5..-1.0 (select_rules)
  random        20% of the pool's tokens drawn at random, the same advantage formula (selection is the only change)
  teacher_neg   every token the informed teacher likes less than the student: A = clip(log p_T - log p_S, -c, 0)
  teacher_both  the same with the positive terms kept: A = clip(log p_T - log p_S, -c, c) (self-distillation with the hint)
  teacher_sel   the lowest-20% by log p_T - log p_S (negative only), OPSA's advantages (teacher-selected, OPSA's penalty)
  answer_key    the lowest-20% student log-prob tokens among the pool's wrong responses, OPSA's advantages
  sft           cross-entropy on demonstrations (definition, computation, boxed answer); no rollouts
The teacher is the same model with the adapter disabled and the definition in its prompt (rules_data.prompt_body
with_def=True). Writes train_log.jsonl (one row per step: rollout shares, loss, gradient and update norms, the
number of punished tokens and their mean advantage), adapter_step<N>/ at --save-steps and adapter_final/.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import rules_data as rd  # noqa: E402
import select_rules as sr  # noqa: E402

ARMS = ("opsa", "random", "teacher_neg", "teacher_both", "teacher_sel", "answer_key", "sft")
NEEDS_TEACHER = ("teacher_neg", "teacher_both", "teacher_sel")
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


# --------------------------------------------------------------------------- advantages (pure numpy, tested)
def advantages(arm, rows, pool_size, fraction, clip, seed):
    """rows: dicts with key, correct, lp_s, ent_s (and lp_t for the teacher arms), per-token lists.
    Returns {key: np.array of per-token advantages} (zeros where a token is not updated)."""
    by_key = {r["key"]: r for r in rows}
    adv = {k: np.zeros(len(r["lp_s"])) for k, r in by_key.items()}
    if arm == "sft":
        for k in adv:
            adv[k][:] = 1.0
        return adv
    if arm in ("teacher_neg", "teacher_both"):
        for k, r in by_key.items():
            gap = np.asarray(r["lp_t"], dtype=np.float64) - np.asarray(r["lp_s"], dtype=np.float64)
            adv[k] = np.clip(gap, -clip, 0.0 if arm == "teacher_neg" else clip)
        return adv
    lp_s = {k: r["lp_s"] for k, r in by_key.items()}
    lengths = {k: len(r["lp_s"]) for k, r in by_key.items()}
    for p_idx, keys in enumerate(sr.make_pools(by_key, pool_size, seed)):
        if arm == "opsa":
            sel = sr.lowest_fraction(lp_s, keys, fraction)
        elif arm == "random":
            sel = sr.random_fraction(lengths, keys, fraction, common.stable_seed("random", seed, p_idx))
        elif arm == "teacher_sel":
            gap = {k: (np.asarray(by_key[k]["lp_t"]) - np.asarray(by_key[k]["lp_s"])).tolist() for k in keys}
            sel = sr.lowest_fraction(gap, keys, fraction, only_negative=True)
        elif arm == "answer_key":
            wrong = [k for k in keys if not by_key[k]["correct"]]
            k_budget = sr.budget(sum(lengths[k] for k in keys), fraction)
            sel = sr.lowest_fraction(lp_s, wrong, fraction, k=k_budget) if wrong else {}
        else:
            raise ValueError(arm)
        picked = [(k, t) for k in keys for t in sel.get(k, [])]
        a = sr.opsa_advantages([by_key[k]["ent_s"][t] for k, t in picked])
        for (k, t), v in zip(picked, a):
            adv[k][t] = v
    return adv


def pool_summary(adv):
    n_sel = sum(int((a != 0).sum()) for a in adv.values())
    n_tok = sum(len(a) for a in adv.values())
    mean_abs = float(sum(np.abs(a).sum() for a in adv.values()) / n_sel) if n_sel else 0.0
    return dict(n_tokens=n_tok, n_updated=n_sel, mean_abs_adv=round(mean_abs, 4))


# --------------------------------------------------------------------------- model
def load_student(name, lora_r, lora_alpha, seed):
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    torch.manual_seed(seed)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda")
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.config.use_cache = True
    cfg = LoraConfig(r=lora_r, lora_alpha=lora_alpha, lora_dropout=0.0, target_modules=LORA_TARGETS, task_type="CAUSAL_LM")
    model = get_peft_model(model, cfg)
    return model


def pad_right(seqs, pad_id, device):
    import torch

    L = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), L), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), L), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, :len(s)] = torch.tensor(s)
        mask[i, :len(s)] = 1
    return ids.to(device), mask.to(device)


def response_logprobs(model, pairs, pad_id, batch, entropy=False, grad=False, adv=None, n_norm=None):
    """pairs: [(prompt_ids, response_ids)]. Without grad: per-response lists of log p(y_t) (and the entropy of
    each next-token distribution). With grad: accumulates the gradient of -sum A_t log p(y_t) / n_norm and
    returns the loss value."""
    import torch

    lps, ents, total = [], [], 0.0
    for i in range(0, len(pairs), batch):
        chunk = pairs[i:i + batch]
        ids, mask = pad_right([p + r for p, r in chunk], pad_id, model.device)
        with torch.set_grad_enabled(grad):
            logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits
            loss = 0.0
            for j, (p, r) in enumerate(chunk):
                z = logits[j, len(p) - 1:len(p) - 1 + len(r)].float()
                logp = torch.log_softmax(z, dim=-1)
                tgt = torch.tensor(r, device=logp.device)
                lp = logp.gather(1, tgt[:, None])[:, 0]
                if grad:
                    a = torch.tensor(adv[i + j], dtype=torch.float32, device=lp.device)
                    loss = loss - (a * lp).sum() / n_norm
                else:
                    lps.append([round(v, 5) for v in lp.tolist()])
                    if entropy:
                        ents.append([round(v, 5) for v in (-(logp.exp() * logp).sum(-1)).tolist()])
            if grad:
                loss.backward()
                total += float(loss.detach())
        del logits
    return total if grad else (lps, ents if entropy else None)


def generate(model, tok, prompts, end_ids, max_tokens, batch, seed):
    """prompts: [prompt_ids]. Returns [(response_ids, finish_reason)], the end token kept when one was sampled."""
    import torch

    tok.padding_side = "left"
    out = []
    for i in range(0, len(prompts), batch):
        chunk = prompts[i:i + batch]
        enc = tok.pad({"input_ids": chunk}, padding=True, return_tensors="pt").to(model.device)
        torch.manual_seed(common.stable_seed("gen", seed, i))
        with torch.no_grad():
            gen = model.generate(**enc, do_sample=True, temperature=1.0, top_p=1.0, top_k=0, max_new_tokens=max_tokens,
                                 eos_token_id=list(end_ids), pad_token_id=tok.pad_token_id)
        L = enc["input_ids"].shape[1]
        for row in gen[:, L:].tolist():
            cut = next((k for k, t in enumerate(row) if t in end_ids), None)
            if cut is None:
                out.append((row[:max_tokens], "length"))
            else:
                out.append((row[:cut + 1], "stop"))
        del gen
    return out


def lora_params(model):
    return [p for p in model.parameters() if p.requires_grad]


def checkpoint(model, opt, ckpt, step, order, rng):
    """Adapter weights, optimizer state and the prompt order after `step`, written atomically enough: state.pt last."""
    import torch

    ckpt = Path(ckpt)
    ckpt.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ckpt / "adapter")
    torch.save(dict(step=step, opt=opt.state_dict(), order=order, rng=rng), ckpt / "state.pt.tmp")
    (ckpt / "state.pt.tmp").replace(ckpt / "state.pt")


def resume(model, opt, ckpt, log_path):
    """Load the checkpoint into the live PEFT model and optimizer; trim the log to the checkpointed step. Returns the next step."""
    import torch
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file

    ckpt = Path(ckpt)
    state = torch.load(ckpt / "state.pt", weights_only=False)
    weights = load_file(ckpt / "adapter" / "adapter_model.safetensors")
    result = set_peft_model_state_dict(model, weights)
    if getattr(result, "unexpected_keys", None):
        raise RuntimeError(f"adapter checkpoint has unexpected keys: {list(result.unexpected_keys)[:3]}")
    opt.load_state_dict(state["opt"])
    log_path = Path(log_path)
    if log_path.exists():
        rows = [r for r in common.read_jsonl(log_path) if r["step"] <= state["step"]]
        common.write_jsonl_atomic(log_path, rows)
    return state["step"] + 1


# --------------------------------------------------------------------------- the loop
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--arm", choices=ARMS, required=True)
    ap.add_argument("--items", required=True, help="items.jsonl from run_rules.py; the train split is used")
    ap.add_argument("--model", default=common.STUDENT)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--prompts-per-step", type=int, default=128)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--pool-size", type=int, default=16)
    ap.add_argument("--fraction", type=float, default=0.2)
    ap.add_argument("--adv-clip", type=float, default=5.0, help="|A| cap for the teacher arms")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--gen-batch", type=int, default=64)
    ap.add_argument("--score-batch", type=int, default=8)
    ap.add_argument("--train-batch", type=int, default=4)
    ap.add_argument("--save-steps", default="", help="comma list of steps after which the adapter is saved")
    ap.add_argument("--ckpt-every", type=int, default=5, help="write a resumable checkpoint (adapter, optimizer, order) every N steps")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    run = Path(args.run)
    run.mkdir(parents=True, exist_ok=True)
    if (run / "train.meta.json").exists():
        print(f"training already done: {run / 'train.meta.json'}")
        return
    common.ensure_c_compiler()
    import torch
    from transformers import AutoTokenizer

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model)
    end_ids = common.end_token_ids(tok)
    train_items = [it for it in common.read_jsonl(args.items) if it["split"] == "train"]
    if not train_items:
        sys.exit("no train items in " + args.items)
    for it in train_items:
        op = rd.BY_SYM[it["sym"]]
        it["prompt_ids"] = rd.prompt_ids_for(tok, it)
        it["teacher_ids"] = rd.prompt_ids_for(tok, it, with_def=True)
        if args.arm == "sft":
            it["demo_ids"] = tok(rd.demonstration(op, it), add_special_tokens=False)["input_ids"] + [end_ids[0]]
    model = load_student(args.model, args.lora_r, args.lora_alpha, args.seed)
    params = lora_params(model)
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    n_params = sum(p.numel() for p in params)
    print(f"{args.arm}: {len(train_items)} train prompts, {n_params / 1e6:.1f}M LoRA parameters, lr {args.lr}", flush=True)
    save_steps = {int(s) for s in args.save_steps.split(",") if s.strip()}
    order, rng = [], np.random.default_rng(common.stable_seed("order", args.seed))
    start = 1
    ckpt = run / "ckpt"
    if (ckpt / "state.pt").exists():
        start = resume(model, opt, ckpt, run / "train_log.jsonl")
        state = torch.load(ckpt / "state.pt", weights_only=False)
        order, rng = state["order"], state["rng"]
        print(f"resumed from step {start - 1}", flush=True)
    log_f = open(run / "train_log.jsonl", "a" if start > 1 else "w")
    for step in range(start, args.steps + 1):
        t_step = time.time()
        while len(order) < args.prompts_per_step:
            order += rng.permutation(len(train_items)).tolist()
        batch_items = [train_items[i] for i in order[:args.prompts_per_step]]
        order = order[args.prompts_per_step:]
        rec = dict(step=step)

        if args.arm == "sft":
            pairs = [(it["prompt_ids"], it["demo_ids"]) for it in batch_items]
            rows = [dict(key=f"{it['item_id']}#{step}#{i}", correct=True, lp_s=[0.0] * len(d), ent_s=[0.0] * len(d))
                    for i, (it, (_, d)) in enumerate(zip(batch_items, pairs))]
            t_gen = t_score = 0.0
        else:
            model.eval()
            gens = generate(model, tok, [it["prompt_ids"] for it in batch_items], end_ids, args.max_tokens, args.gen_batch,
                            common.stable_seed(args.seed, step))
            t_gen = time.time() - t_step
            torch.cuda.empty_cache()   # the KV cache's blocks would otherwise stay cached in generation-sized pieces
            texts = [tok.decode(r, skip_special_tokens=True) for r, _ in gens]
            answers = [rd.answer_of(t) for t in texts]
            kept = [(it, r, ans) for it, (r, fin), ans in zip(batch_items, gens, answers) if fin == "stop"]
            rec.update(n_rollouts=len(gens), truncated=round(1 - len(kept) / len(gens), 4),
                       mean_tokens=round(float(np.mean([len(r) for r, _ in gens])), 1),
                       correct=round(float(np.mean([a == it["answer"] for it, a in zip(batch_items, answers)])), 4),
                       habit=round(float(np.mean([a is not None and a == it["habit"] for it, a in zip(batch_items, answers)])), 4),
                       no_answer=round(float(np.mean([a is None for a in answers])), 4))
            if step in (1, args.steps):
                common.write_jsonl_atomic(run / f"rollouts_step{step}.jsonl",
                                          [dict(item_id=it["item_id"], answer=it["answer"], text=t, got=a, finish=fin)
                                           for it, (r, fin), t, a in zip(batch_items, gens, texts, answers)])
            if len(kept) < 2:
                rec.update(skipped="fewer than 2 kept responses", seconds=round(time.time() - t_step, 1))
                print(json.dumps(rec), flush=True)
                log_f.write(json.dumps(rec) + "\n")
                log_f.flush()
                continue
            t1 = time.time()
            pairs = [(it["prompt_ids"], r) for it, r, _ in kept]
            lp_s, ent_s = response_logprobs(model, pairs, tok.pad_token_id, args.score_batch, entropy=True)
            rows = [dict(key=f"{it['item_id']}#{step}#{i}", correct=ans == it["answer"], lp_s=lp, ent_s=en)
                    for i, ((it, r, ans), lp, en) in enumerate(zip(kept, lp_s, ent_s))]   # an item can recur in a step
            if args.arm in NEEDS_TEACHER:
                with model.disable_adapter():
                    lp_t, _ = response_logprobs(model, [(it["teacher_ids"], r) for it, r, _ in kept], tok.pad_token_id, args.score_batch)
                for row, lp in zip(rows, lp_t):
                    row["lp_t"] = lp
                    row["gap"] = round(float(np.mean(np.asarray(lp) - np.asarray(row["lp_s"]))), 4)
                rec["mean_gap"] = round(float(np.mean([r["gap"] for r in rows])), 4)
            rec["mean_entropy"] = round(float(np.mean([v for r in rows for v in r["ent_s"]])), 4)
            t_score = time.time() - t1
            torch.cuda.empty_cache()

        adv = advantages(args.arm, rows, args.pool_size, args.fraction, args.adv_clip, common.stable_seed("pools", args.seed, step))
        assert len(adv) == len(rows) == len(pairs) and all(len(adv[r["key"]]) == len(pr[1]) for r, pr in zip(rows, pairs)), "advantage/response misalignment"
        rec.update(pool_summary(adv))
        t2 = time.time()
        model.train()
        opt.zero_grad(set_to_none=True)
        n_norm = sum(len(r["lp_s"]) for r in rows)
        loss = response_logprobs(model, pairs, tok.pad_token_id, args.train_batch, grad=True,
                                 adv=[adv[r["key"]] for r in rows], n_norm=n_norm)
        before = [p.detach().clone() for p in params]
        gnorm = float(torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm))
        if not np.isfinite(loss) or not np.isfinite(gnorm):
            sys.exit(f"non-finite loss or gradient at step {step}: loss {loss}, grad norm {gnorm}")
        opt.step()
        upd = float(torch.sqrt(sum(((p.detach() - b) ** 2).sum() for p, b in zip(params, before))))
        pnorm = float(torch.sqrt(sum((p.detach() ** 2).sum() for p in params)))
        rec.update(loss=round(loss, 5), grad_norm=round(gnorm, 4), update_norm=round(upd, 5), param_norm=round(pnorm, 3),
                   gen_s=round(t_gen, 1), score_s=round(t_score, 1), train_s=round(time.time() - t2, 1),
                   seconds=round(time.time() - t_step, 1))
        print(json.dumps(rec), flush=True)
        log_f.write(json.dumps(rec) + "\n")
        log_f.flush()
        if step in save_steps and step != args.steps:
            model.save_pretrained(run / f"adapter_step{step}")
        if args.ckpt_every and step % args.ckpt_every == 0 and step != args.steps:
            checkpoint(model, opt, ckpt, step, order, rng)
    model.save_pretrained(run / "adapter_final")
    import shutil
    shutil.rmtree(ckpt, ignore_errors=True)
    meta = dict(stage="train", arm=args.arm, model=args.model, steps=args.steps, prompts_per_step=args.prompts_per_step,
                max_tokens=args.max_tokens, pool_size=args.pool_size, fraction=args.fraction, adv_clip=args.adv_clip, lr=args.lr,
                lora_r=args.lora_r, lora_alpha=args.lora_alpha, max_grad_norm=args.max_grad_norm, seed=args.seed,
                n_train_items=len(train_items), lora_params=n_params, temperature=1.0, top_p=1.0, top_k=None, thinking=False,
                loss="-(sum_t A_t log p_t) / N over the kept responses' N tokens, one AdamW step per rollout batch",
                seconds=round(time.time() - t0))
    common.write_json(run / "train.meta.json", meta)
    print(f"done: {args.steps} steps in {meta['seconds'] / 60:.1f} min -> {run / 'adapter_final'}")


if __name__ == "__main__":
    main()
