#!/usr/bin/env python
"""Study 2 (the proposal's experiment 5), measured first: refusal rates on SUM and the premise audit.

    python src/premise_audit.py --run <run dir> [--pool-size 16] [--fraction 0.2] [--seed 0]

Needs a SUM run of stage 1 (rollout.py --dataset sum). Three steps, each skipped if its output exists:

1. Student log-probability and entropy of every response token (transformers, teacher-forced, as score.py does),
   and the decision point of every response with a complete box: the position right after the last "\\boxed",
   where the model writes the brace token and, with it, the kind of content that follows (Qwen's tokenizer merges
   "{" with a following "\\", "-", "(", a letter...). The distribution there, plus two counterfactual forwards, one
   after a bare "{" and one after "{\\", give the probability of the canonical refusal path "{" then "I" and of the
   worded path "{\\" then "text", the same way for every response.  -> scores.jsonl, decisions.jsonl
2. OPSA's rule, the lowest --fraction of student log-probabilities in pools of --pool-size responses with its
   entropy-weighted advantages, and the random rule on the same pools.  -> selected.jsonl
3. The audit. The refusal tokens of every refusing response: the first token of the boxed refusal (where the model
   commits), every token of it, and hedge phrases in the reasoning before the box ("cannot be determined", "not
   enough information", ...). For each kind: the share OPSA's rule selects against the --fraction it selects
   overall, the tokens' probabilities, and their advantages. The first token of a boxed false answer, and of a
   correct answer on the originals, for contrast. Where OPSA's punished tokens sit in refusing responses.
   -> audit.json, tables/*.csv

Rates on the unanswerable items (refusal, false answer, no answer) and on their originals (correct, wrong, refusal)
carry 95% bootstrap intervals that resample problem pairs. Side effects: response length, share at the generation
cap, share with severe repetition (a 20-token span repeated three or more times).
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import select_rules as sr  # noqa: E402
import sum_data  # noqa: E402
from analyze import _category  # noqa: E402

SEP = r"(?:\s|\*|_)+"            # words may be separated by spaces or markdown emphasis
APO = "[\u2019']"
_MODAL = (r"(?:cannot|can" + APO + r"?t|could" + SEP + r"not|couldn" + APO + r"?t|unable" + SEP + r"to|impossible" + SEP + r"to|not"
          + SEP + r"possible" + SEP + r"to|no" + SEP + r"way" + SEP + r"to)")
_ADV = r"(?:\w+ly" + SEP + r")?"
HEDGE = re.compile(
    r"\b(?:"
    + _MODAL + SEP + r"be" + SEP + _ADV + r"(?:determined|computed|calculated|answered|known|found|solved|evaluated|concluded|inferred|deduced)"
    r"|" + _MODAL + SEP + _ADV + r"(?:determine|know|answer|compute|calculate|conclude|evaluate|proceed|infer|deduce)"
    r"|not" + SEP + r"enough" + SEP + r"(?:information|data|details|info)"
    r"|insufficient" + SEP + r"(?:information|data|details)"
    r"|(?:information|data|value|values|details?|constant|parameter)" + SEP + r"(?:is|are)" + SEP + r"(?:missing|insufficient|not" + SEP
    + r"(?:given|provided|specified|stated))"
    r"|(?:is|are|was|were)" + SEP + r"not" + SEP + r"(?:given|provided|specified|stated|mentioned)"
    r"|(?:is|are)" + SEP + r"missing|missing" + SEP + r"(?:information|value|data|detail|definition)"
    r"|do(?:es)?" + SEP + r"not" + SEP + r"(?:have|specify|give|provide|tell|state|mention|know)"
    r"|do(?:es)?n" + APO + r"?t" + SEP + r"(?:have|specify|give|provide|tell|state|mention|know)"
    r"|don" + APO + r"?t" + SEP + r"know|dont" + SEP + r"know"
    r"|without" + SEP + r"(?:more|additional|further)" + SEP + r"information|without" + SEP + r"knowing"
    r"|lack" + SEP + r"of" + SEP + r"information|lacks?" + SEP + r"(?:the" + SEP + r")?(?:information|data)"
    r"|unspecified|underspecified|undetermined|indeterminate(?!" + SEP + r"form)|ambiguous|ambiguity|unclear"
    r"|unanswerable|ill-posed|not" + SEP + r"well" + r"[- ]" + r"defined"
    r")\b", re.I)
TOP_K = 10
# token groups whose probability mass is read at the decision point (exact, case-sensitive, after stripping spaces)
GROUPS = {"I": ("I",), "text": ("text",), "frac": ("frac", "dfrac", "tfrac"), "sqrt": ("sqrt",), "mathrm": ("mathrm",),
          "textbf": ("textbf",), "(": ("(",), "-": ("-",)}


# --------------------------------------------------------------------------- token spans
def token_spans(tok, ids):
    """(text, [(start, end) per token]) with character offsets into text = decode(ids). Per-token decoding is used
    when it concatenates to the same text; otherwise prefixes are decoded (bytes split across tokens)."""
    text = tok.decode(ids, skip_special_tokens=False)
    parts = [tok.decode([i], skip_special_tokens=False) for i in ids]
    if "".join(parts) != text:
        parts, prev = [], ""
        for k in range(1, len(ids) + 1):
            cur = tok.decode(ids[:k], skip_special_tokens=False)
            parts.append(cur[len(prev):])
            prev = cur
        text = prev
    spans, pos = [], 0
    for p in parts:
        spans.append((pos, pos + len(p)))
        pos += len(p)
    return text, spans


def token_at(spans, char):
    """Index of the token whose span holds character `char` (the first token starting at or after it)."""
    for t, (a, b) in enumerate(spans):
        if a <= char < b or a >= char:
            return t
    return None


def tokens_in(spans, start, end):
    return [t for t, (a, b) in enumerate(spans) if b > start and a < end]


def last_box(text):
    """(open_char, close_char) of the last complete \\boxed{...} or \\fbox{...}: content is text[open_char:close_char].
    Mirrors common.last_boxed, which decides the outcome."""
    best = None
    for macro in ("\\boxed", "\\fbox"):
        i = text.rfind(macro + "{")
        if i < 0:
            continue
        j, depth = i + len(macro) + 1, 1
        while j < len(text) and depth:
            depth += {"{": 1, "}": -1}.get(text[j], 0)
            j += 1
        if depth == 0 and (best is None or i > best[0]):
            best = (i + len(macro) + 1, j - 1)
    return best


def decision_token(text, spans, box):
    """(t, glue): the first token of the box's content that is not brace/backslash glue, and the glue text before
    it. Qwen's tokenizer merges "{" and "\\" into one token, so for \\boxed{\\text{I don't know.}} the first
    token holding a choice is "text" (against "frac", a digit, ...), with glue "\\"; for \\boxed{I don't know.} it is
    "I" with no glue. None when there is no complete box or the box is empty."""
    if box is None or box[1] <= box[0]:
        return None, ""
    t = token_at(spans, box[0])
    while t is not None and t < len(spans) and spans[t][0] < box[1] and not text[max(spans[t][0], box[0]):spans[t][1]].strip(" {\\"):
        t += 1
    if t is None or t >= len(spans) or spans[t][0] >= box[1]:
        return None, ""
    return t, text[min(spans[t][0], box[0]):max(spans[t][0], box[0])]   # "\\" after a bare "{", or the "{" the token carries


def hedge_spans(text, before=None):
    end = len(text) if before is None else before
    return [(m.start(), m.end(), m.group(0)) for m in HEDGE.finditer(text[:end])]


def severe_repetition(ids, n=20, times=3):
    if len(ids) < n * times:
        return False
    counts = {}
    for i in range(len(ids) - n + 1):
        g = tuple(ids[i:i + n])
        counts[g] = counts.get(g, 0) + 1
        if counts[g] >= times:
            return True
    return False


# --------------------------------------------------------------------------- step 1: scores and decisions
def token_group_ids(tok):
    """Vocabulary ids whose decoded form, stripped of spaces, is exactly one of each group's strings; 'digit' = one digit."""
    out = {g: [] for g in GROUPS}
    out["digit"] = []
    for s, i in tok.get_vocab().items():
        d = tok.convert_tokens_to_string([s]).strip()
        for g, forms in GROUPS.items():
            if d in forms:
                out[g].append(i)
        if len(d) == 1 and d.isdigit():
            out["digit"].append(i)
    return out


def single_id(tok, text):
    ids = tok(text, add_special_tokens=False)["input_ids"]
    return ids[0] if len(ids) == 1 else None


def masses(p, gids, tok):
    import torch

    out = {g: round(float(p[torch.tensor(v, device=p.device)].sum()), 6) for g, v in gids.items() if v}
    top_p, top_i = torch.topk(p, TOP_K)
    out["top"] = [(int(i), tok.decode([int(i)]), round(float(v), 6)) for v, i in zip(top_p, top_i)]
    out["entropy"] = round(float(-(p * torch.log(p.clamp_min(1e-30))).sum()), 5)
    return out


def score_tokens(model, head_w, prompt_ids, resp_ids, chunk=1024):
    """Student log p and entropy of every response token, and the hidden states h with h[t] predicting resp_ids[t]."""
    import torch
    from score import hidden

    n = len(prompt_ids)
    h = hidden(model, prompt_ids + resp_ids)[n - 1:n - 1 + len(resp_ids)]
    tgt = torch.tensor(resp_ids, device=h.device)
    lps, ents = [], []
    with torch.no_grad():
        for i in range(0, h.shape[0], chunk):
            logp = torch.log_softmax(h[i:i + chunk].float() @ head_w.T, dim=-1)
            lps.append(logp.gather(1, tgt[i:i + chunk, None])[:, 0])
            ents.append(-(logp.exp() * logp).sum(-1))
    return [round(v, 5) for v in torch.cat(lps).tolist()], [round(v, 5) for v in torch.cat(ents).tolist()], h


def next_dist(model, head_w, prefix_ids):
    import torch
    from score import hidden

    with torch.no_grad():
        return torch.softmax(hidden(model, prefix_ids)[-1].float() @ head_w.T, -1)


def decide(model, head_w, tok, prompt_ids, ids, h, text, spans, box, gids, brace_ids):
    """The decision point: the token holding the box's opening brace (right after \\boxed), its distribution, and the
    distributions after a bare "{" and after "{\\" (counterfactual where the model wrote something else)."""
    import torch

    t_brace = token_at(spans, box[0] - 1)
    t_first, glue = decision_token(text, spans, box)
    if t_brace is None or t_brace >= len(ids):
        return None
    with torch.no_grad():
        p = torch.softmax(h[t_brace].float() @ head_w.T, -1)
        at = masses(p, {}, tok)
        at.update(t=t_brace, kept_id=ids[t_brace], kept_str=tok.decode([ids[t_brace]]), p_kept=round(float(p[ids[t_brace]]), 6),
                  p_bare=round(float(p[brace_ids["bare"]]), 6) if brace_ids["bare"] is not None else None,
                  p_backslash=round(float(p[brace_ids["backslash"]]), 6) if brace_ids["backslash"] is not None else None)
        prefix = prompt_ids + ids[:t_brace]
        after_bare = masses(next_dist(model, head_w, prefix + [brace_ids["bare"]]), gids, tok) if brace_ids["bare"] is not None else None
        after_bs = masses(next_dist(model, head_w, prefix + [brace_ids["backslash"]]), gids, tok) if brace_ids["backslash"] is not None else None
    first = None
    if t_first is not None:
        first = dict(t=t_first, kept_id=ids[t_first], kept_str=tok.decode([ids[t_first]]), glue=glue)
    return dict(brace=at, after_bare=after_bare, after_backslash=after_bs, first=first, box_content=text[box[0]:box[1]][:80])


def step_scores(run, rollouts, problems, student):
    import torch
    from score import load_model
    from transformers import AutoTokenizer

    scores_path, dec_path = run / "scores.jsonl", run / "decisions.jsonl"
    if scores_path.exists() and dec_path.exists():
        return
    tok = AutoTokenizer.from_pretrained(student)
    gids = token_group_ids(tok)
    brace_ids = dict(bare=single_id(tok, "{"), backslash=single_id(tok, "{\\"))
    print(f"brace tokens: {brace_ids}; group sizes: { {g: len(v) for g, v in gids.items()} }")
    model, head_w = load_model(student)
    srows, drows, t0 = [], [], time.time()
    for i, r in enumerate(rollouts, 1):
        ids = r["response_ids"]
        prompt_ids = problems[r["problem_id"]]["prompt_ids"]
        text, spans = token_spans(tok, ids)
        box = last_box(text)
        lp, ent, h = score_tokens(model, head_w, prompt_ids, ids)
        dec = decide(model, head_w, tok, prompt_ids, ids, h, text, spans, box, gids, brace_ids) if box and box[1] > box[0] else None
        del h
        srows.append(dict(key=r["key"], problem_id=r["problem_id"], correct=r["correct"], lp_s=lp, ent_s=ent,
                          tokens=[text[a:b] for a, b in spans]))
        drows.append(dict(key=r["key"], problem_id=r["problem_id"], variant=r["variant"], outcome=r["outcome"],
                          has_box=box is not None, box_chars=list(box) if box else None,
                          hedges=[(a, b, s) for a, b, s in hedge_spans(text, box[0] - 1 if box else None)],
                          decision=dec))
        if i % 200 == 0:
            print(f"   {student}: {i}/{len(rollouts)} responses scored, {time.time() - t0:.0f}s", flush=True)
    del model, head_w
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    common.write_jsonl_atomic(scores_path, srows)
    common.write_jsonl_atomic(dec_path, drows)
    print(f"scored {len(srows)} responses with {student}, {time.time() - t0:.0f}s")


# --------------------------------------------------------------------------- step 2: the rules
def step_select(run, scores, pool_size, fraction, seed):
    path = run / "selected.jsonl"
    if path.exists():
        return
    by_key = {r["key"]: r for r in scores}
    pools = sr.make_pools(list(by_key), pool_size, seed)
    lp_s = {k: r["lp_s"] for k, r in by_key.items()}
    lengths = {k: len(r["lp_s"]) for k, r in by_key.items()}
    opsa, rnd, adv = {}, {}, {}
    for p_idx, keys in enumerate(pools):
        sel = sr.lowest_fraction(lp_s, keys, fraction)
        opsa.update(sel)
        flat = [(k, t) for k in keys for t in sel[k]]
        for (k, t), a in zip(flat, sr.opsa_advantages([by_key[k]["ent_s"][t] for k, t in flat])):
            adv.setdefault(k, {})[t] = float(a)
        rnd.update(sr.random_fraction(lengths, keys, fraction, common.stable_seed("random", seed, p_idx)))
    common.write_jsonl_atomic(path, [dict(key=k, problem_id=by_key[k]["problem_id"], n=lengths[k], opsa=opsa[k],
                                         opsa_adv=[adv[k][t] for t in opsa[k]], random=rnd[k]) for k in by_key])
    print(f"{len(pools)} pools of {pool_size}; OPSA selects {sum(map(len, opsa.values()))} of {sum(lengths.values())} tokens")


# --------------------------------------------------------------------------- step 3: rates and audit
def boot_share(flags, clusters, n_boot=2000, seed=0):
    """(share, lo, hi) of a 0/1 list, clusters resampled with replacement."""
    flags, clusters = np.asarray(flags, float), np.asarray(clusters)
    if flags.size == 0:
        return np.nan, np.nan, np.nan
    cl = np.unique(clusters)
    idx = {c: i for i, c in enumerate(cl)}
    s = np.zeros(len(cl)); c = np.zeros(len(cl))
    for f, k in zip(flags, clusters):
        s[idx[k]] += f; c[idx[k]] += 1
    w = np.random.default_rng(seed).multinomial(len(cl), np.full(len(cl), 1 / len(cl)), size=n_boot)
    with np.errstate(invalid="ignore", divide="ignore"):
        b = (w @ s) / (w @ c)
    lo, hi = np.nanpercentile(b, [2.5, 97.5])
    return float(flags.mean()), float(lo), float(hi)


def fmt(m, lo, hi, d=3):
    return "n/a" if pd.isna(m) else f"{m:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]"


def rates_table(rollouts):
    rows = []
    for variant in ("unanswerable", "answerable"):
        sub = [r for r in rollouts if r["variant"] == variant]
        pairs = [r["pair"] for r in sub]
        outs = ["refusal", "false_answer", "no_answer"] if variant == "unanswerable" else ["correct", "wrong", "refusal", "no_answer"]
        row = dict(variant=variant, problems=len({r["problem_id"] for r in sub}), responses=len(sub))
        for o in outs:
            row[o] = fmt(*boot_share([r["outcome"] == o for r in sub], pairs))
        idk = [r["outcome"] == "refusal" and sum_data.boxed_kind(r["text"]) == "idk" for r in sub]
        row["refusal: \\boxed{I don't know} form"] = fmt(*boot_share(idk, pairs))
        row["mean tokens"] = round(float(np.mean([r["n_tokens"] for r in sub])))
        row["at generation cap"] = fmt(*boot_share([r["truncated"] for r in sub], pairs))
        row["severe repetition"] = fmt(*boot_share([severe_repetition(r["response_ids"]) for r in sub], pairs))
        rows.append(row)
    # pair-level: refuses the unanswerable version and solves the original, per (pair, sample)
    by = {}
    for r in rollouts:
        by.setdefault((r["pair"], r["sample_idx"]), {})[r["variant"]] = r["outcome"]
    both = [(v.get("unanswerable") == "refusal" and v.get("answerable") == "correct") for v in by.values()]
    rows.append(dict(variant="pair: refuses the unanswerable and solves the original (same sample index)",
                     problems=len({p for p, _ in by}), responses=len(both),
                     correct=fmt(*boot_share(both, [p for p, _ in by]))))
    return pd.DataFrame(rows)


def _spans(tokens):
    spans, pos = [], 0
    for tk in tokens:
        spans.append((pos, pos + len(tk)))
        pos += len(tk)
    return spans


def audit_tables(rollouts, scores, selected, decisions, fraction):
    sc = {r["key"]: r for r in scores}
    sel = {r["key"]: r for r in selected}
    roll = {r["key"]: r for r in rollouts}
    names = ("refusal: first content token of the boxed refusal", "refusal: every token of the boxed refusal",
             "refusal: brace token that opens the box", "hedge phrase in the reasoning of refusing responses: first token",
             "hedge phrase in the reasoning of refusing responses: every token",
             "hedge phrase in the reasoning of false-answer responses: first token",
             "false answer: first content token of the box (unanswerable)", "correct answer: first content token of the box (originals)")
    kinds = {k: [] for k in names}
    no_hedge_refusals = 0
    for d in decisions:
        r, s = roll[d["key"]], sc[d["key"]]
        spans = _spans(s["tokens"])
        dec = d["decision"]
        refusing = r["variant"] == "unanswerable" and d["outcome"] == "refusal"
        false_ans = r["variant"] == "unanswerable" and d["outcome"] == "false_answer"
        if dec is not None and dec.get("first") is not None:
            a, b = d["box_chars"]
            first, t_brace = dec["first"]["t"], dec["brace"]["t"]
            if refusing:
                kinds[names[0]].append((d["key"], first, r["pair"]))
                kinds[names[1]] += [(d["key"], t, r["pair"]) for t in tokens_in(spans, a, b)]
                kinds[names[2]].append((d["key"], t_brace, r["pair"]))
            elif false_ans:
                kinds[names[6]].append((d["key"], first, r["pair"]))
            elif r["variant"] == "answerable" and d["outcome"] == "correct":
                kinds[names[7]].append((d["key"], first, r["pair"]))
        if refusing or false_ans:
            if refusing and not d["hedges"]:
                no_hedge_refusals += 1
            for a, b, _ in d["hedges"]:
                ts = tokens_in(spans, a, b)
                if not ts:
                    continue
                if refusing:
                    kinds[names[3]].append((d["key"], ts[0], r["pair"]))
                    kinds[names[4]] += [(d["key"], t, r["pair"]) for t in ts]
                else:
                    kinds[names[5]].append((d["key"], ts[0], r["pair"]))
    rows = []
    for name, toks in kinds.items():
        if not toks:
            rows.append(dict(tokens=name, n=0)); continue
        in_opsa = [t in set(sel[k]["opsa"]) for k, t, _ in toks]
        in_rnd = [t in set(sel[k]["random"]) for k, t, _ in toks]
        ps = np.exp([sc[k]["lp_s"][t] for k, t, _ in toks])
        advs = [sel[k]["opsa_adv"][sel[k]["opsa"].index(t)] for (k, t, _), f in zip(toks, in_opsa) if f]
        pairs = [pr for _, _, pr in toks]
        rows.append({"tokens": name, "n": len(toks), "responses": len({k for k, _, _ in toks}),
                     "share in OPSA's set [95% CI]": fmt(*boot_share(in_opsa, pairs)),
                     "expected (fraction)": fraction, "share in random set": round(float(np.mean(in_rnd)), 3),
                     "median p(token)": round(float(np.median(ps)), 3), "share p<0.5": round(float((ps < 0.5).mean()), 3),
                     "mean OPSA advantage when selected": round(float(np.mean(advs)), 3) if advs else np.nan})
    premise = pd.DataFrame(rows)

    # the decision point: the brace token right after \boxed, read the same way for every response
    drows = []
    for group, flt in (("unanswerable, refused", lambda d: d["variant"] == "unanswerable" and d["outcome"] == "refusal"),
                       ("unanswerable, false answer", lambda d: d["variant"] == "unanswerable" and d["outcome"] == "false_answer"),
                       ("original, correct", lambda d: d["variant"] == "answerable" and d["outcome"] == "correct"),
                       ("original, wrong", lambda d: d["variant"] == "answerable" and d["outcome"] == "wrong"),
                       ("original, refused", lambda d: d["variant"] == "answerable" and d["outcome"] == "refusal")):
        ds = [d["decision"] for d in decisions if flt(d) and d["decision"] is not None and d["decision"]["after_bare"] is not None]
        if not ds:
            continue
        p_bare = np.array([d["brace"]["p_bare"] or 0.0 for d in ds])
        p_bs = np.array([d["brace"]["p_backslash"] or 0.0 for d in ds])
        p_I_given_bare = np.array([d["after_bare"].get("I", 0.0) for d in ds])
        p_digit_given_bare = np.array([d["after_bare"].get("digit", 0.0) for d in ds])
        p_text_given_bs = np.array([d["after_backslash"].get("text", 0.0) if d["after_backslash"] else 0.0 for d in ds])
        p_refusal = p_bare * p_I_given_bare
        p_worded = p_bs * p_text_given_bs
        drows.append({"responses": group, "n": len(ds),
                      "P(brace token actually written)": round(float(np.mean([d["brace"]["p_kept"] for d in ds])), 3),
                      "entropy at the brace": round(float(np.mean([d["brace"]["entropy"] for d in ds])), 3),
                      "P('{')": round(float(p_bare.mean()), 3), "P('{\\\\')": round(float(p_bs.mean()), 3),
                      "P('I' | '{')": round(float(p_I_given_bare.mean()), 4), "P(digit | '{')": round(float(p_digit_given_bare.mean()), 3),
                      "P(refusal path '{' 'I'), mean": round(float(p_refusal.mean()), 4), "median": round(float(np.median(p_refusal)), 4),
                      "share > 0.1": round(float((p_refusal > 0.1).mean()), 3), "share > 0.5": round(float((p_refusal > 0.5).mean()), 3),
                      "P(worded path '{\\\\' 'text'), mean": round(float(p_worded.mean()), 4)})
    decision = pd.DataFrame(drows)
    firsts = {}
    for d in decisions:
        dec = d["decision"]
        if d["variant"] == "unanswerable" and d["outcome"] == "refusal" and dec and dec.get("first"):
            k = (dec["first"]["glue"] + "|" + dec["first"]["kept_str"]).replace("\n", "\\n")
            firsts[k] = firsts.get(k, 0) + 1
    first_tokens = pd.DataFrame(sorted(firsts.items(), key=lambda x: -x[1]), columns=["glue|first content token of boxed refusals", "count"])

    # worded false answers on unanswerable items: the box contents, for checking the classification afterwards
    worded = [(d["key"], d["decision"]["box_content"]) for d in decisions
              if d["variant"] == "unanswerable" and d["outcome"] == "false_answer" and d["decision"] and "text" in d["decision"]["box_content"]]
    worded_t = pd.DataFrame(worded, columns=["key", "boxed content of a worded false answer"])

    # where OPSA's punished tokens sit in refusing responses
    cats, n_sel, before, after, no_hedge = {}, 0, 0, 0, 0
    for d in decisions:
        if not (d["variant"] == "unanswerable" and d["outcome"] == "refusal"):
            continue
        s, se = sc[d["key"]], sel[d["key"]]
        first_hedge = None
        if d["hedges"]:
            ts = tokens_in(_spans(s["tokens"]), d["hedges"][0][0], d["hedges"][0][1])
            first_hedge = ts[0] if ts else None
        for t in se["opsa"]:
            c = _category(s["tokens"][t]); cats[c] = cats.get(c, 0) + 1; n_sel += 1
            if first_hedge is None:
                no_hedge += 1
            else:
                before += t < first_hedge; after += t >= first_hedge
    mass = pd.DataFrame([{"token type": k, "share of OPSA's tokens in refusing responses": round(v / max(n_sel, 1), 3)}
                         for k, v in sorted(cats.items(), key=lambda x: -x[1])])
    mass_pos = dict(opsa_tokens_in_refusing_responses=n_sel, before_first_hedge=before, at_or_after_first_hedge=after,
                    in_refusals_with_no_hedge_detected=no_hedge, refusing_responses_with_no_hedge_detected=no_hedge_refusals)
    return premise, decision, first_tokens, mass, mass_pos, worded_t


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--pool-size", type=int, default=16)
    ap.add_argument("--fraction", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    run = Path(args.run)
    rmeta = common.read_json(run / "rollout.meta.json")
    if rmeta.get("dataset") != "sum":
        sys.exit(f"{run} is not a SUM run (rollout.meta.json dataset = {rmeta.get('dataset')})")
    student = rmeta["model"]
    problems = {p["problem_id"]: p for p in common.read_jsonl(run / "problems.jsonl")}
    rollouts = common.read_jsonl(run / "rollouts.jsonl")
    t0 = time.time()

    common.ensure_c_compiler()
    step_scores(run, rollouts, problems, student)
    scores = common.read_jsonl(run / "scores.jsonl")
    decisions = common.read_jsonl(run / "decisions.jsonl")
    step_select(run, scores, args.pool_size, args.fraction, args.seed)
    selected = common.read_jsonl(run / "selected.jsonl")

    rates = rates_table(rollouts)
    premise, decision, first_tokens, mass, mass_pos, worded = audit_tables(rollouts, scores, selected, decisions, args.fraction)
    (run / "tables").mkdir(exist_ok=True)
    for name, t in (("rates", rates), ("premise", premise), ("decision", decision), ("first_tokens", first_tokens), ("punished_mass", mass),
                    ("worded_false_answers", worded)):
        t.to_csv(run / "tables" / f"{name}.csv", index=False)
    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30); pd.set_option("display.max_colwidth", 70)
    for title, t in (("Rates (95% CI over problem pairs)", rates), ("Premise audit: are the refusal tokens in OPSA's set?", premise),
                     ("Decision point: the brace token after the last \\boxed, and the paths after '{' and '{\\\\'", decision),
                     ("First content token of boxed refusals", first_tokens), ("Where OPSA's punished tokens sit in refusing responses", mass),
                     ("Worded false answers on unanswerable items (check the classification)", worded.head(40))):
        print(f"\n** {title} **"); print(t.to_string(index=False))
    print("OPSA tokens in refusing responses relative to the first hedge phrase:", mass_pos)
    common.write_json(run / "audit.json", dict(stage="premise_audit", student=student, pool_size=args.pool_size, fraction=args.fraction,
                                               seed=args.seed, n_responses=len(rollouts), punished_mass_position=mass_pos,
                                               rates=rates.to_dict("records"), premise=premise.to_dict("records"),
                                               decision=decision.to_dict("records"), seconds=round(time.time() - t0)))


if __name__ == "__main__":
    main()
