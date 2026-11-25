#!/usr/bin/env python
"""Teacher-signal audit before training (GPU, transformers; check 13 of the plan, after Gains and Collapse in OPD).

    python src/teacher_audit.py --run <base train-split eval dir> --out <dir>

Scores the base model's sampled responses to the training prompts: the student's per-token log-probability and
entropy, and the informed teacher's log-probability (the same model with the definition in its prompt). Ranks
the responses (a) by the teacher's mean advantage log p_T - log p_S (what the teacher arms push toward) and (b)
by OPSA's mean per-token advantage (pools of 16, lowest 20%, entropy-ranked -0.5..-1.0; what OPSA pushes down most)
and reports, per decile, correctness, length, truncation, severe repetition and the two signals. A good teacher's
advantage should rise with correctness and fall with length. Writes audit.json and audit_<ranking>.csv.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import rules_data as rd  # noqa: E402
from neg_train import advantages, response_logprobs  # noqa: E402


def decile_table(rows, score, name, n_bins=10):
    """rows ranked by `score` (ascending), cut into n_bins groups of equal size."""
    order = np.argsort([r[score] for r in rows], kind="stable")
    bins = np.array_split(order, n_bins)
    out = []
    for d, idx in enumerate(bins, 1):
        sub = [rows[i] for i in idx]
        if not sub:
            continue
        out.append(dict(ranking=name, decile=d, n=len(sub), score_mean=round(float(np.mean([r[score] for r in sub])), 4),
                        correct=round(float(np.mean([r["correct"] for r in sub])), 4),
                        habit=round(float(np.mean([r["is_habit"] for r in sub])), 4),
                        mean_tokens=round(float(np.mean([r["n_tokens"] for r in sub])), 1),
                        truncated=round(float(np.mean([r["truncated"] for r in sub])), 4),
                        severe_rep=round(float(np.mean([r["severe_rep"] for r in sub])), 4),
                        teacher_adv=round(float(np.mean([r["teacher_adv"] for r in sub])), 4),
                        opsa_adv=round(float(np.mean([r["opsa_adv"] for r in sub])), 4)))
    return pd.DataFrame(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="rules_eval run on the train split with --keep-ids")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=common.STUDENT)
    ap.add_argument("--pool-size", type=int, default=16)
    ap.add_argument("--fraction", type=float, default=0.2)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "audit.json").exists():
        print(f"audit already done: {out / 'audit.json'}")
        return
    common.ensure_c_compiler()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model)
    rows = [r for r in common.read_jsonl(Path(args.run) / "rollouts.jsonl") if r.get("response_ids")]
    if not rows:
        sys.exit("no rows with response_ids; run rules_eval with --keep-ids")
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
    student = [(rd.prompt_ids_for(tok, r), r["response_ids"]) for r in rows]
    teacher = [(rd.prompt_ids_for(tok, r, with_def=True), r["response_ids"]) for r in rows]
    lp_s, ent_s = response_logprobs(model, student, tok.pad_token_id, args.batch, entropy=True)
    lp_t, _ = response_logprobs(model, teacher, tok.pad_token_id, args.batch)
    scored = []
    for r, ls, es, lt in zip(rows, lp_s, ent_s, lp_t):
        scored.append(dict(key=f"{r['item_id']}#{r['sample_idx']}", correct=bool(r["correct"]), lp_s=ls, ent_s=es, lp_t=lt,
                           is_habit=r["got"] is not None and r["got"] == r["habit"], n_tokens=r["n_tokens"], truncated=r["truncated"],
                           severe_rep=r["severe_rep"]))
    kept = [s for s in scored if not s["truncated"]]   # OPSA's pools hold the kept responses, as in training
    adv = advantages("opsa", kept, args.pool_size, args.fraction, 5.0, common.stable_seed("audit", args.seed))
    for s in scored:
        s["teacher_adv"] = float(np.mean(np.asarray(s["lp_t"]) - np.asarray(s["lp_s"])))
        s["opsa_adv"] = float(adv[s["key"]].mean()) if s["key"] in adv else 0.0
        s["opsa_selected"] = float((adv[s["key"]] != 0).mean()) if s["key"] in adv else 0.0
    tables = {"teacher": decile_table(scored, "teacher_adv", "teacher advantage, ascending"),
              "opsa": decile_table(kept, "opsa_adv", "OPSA mean advantage, ascending (most punished first)")}
    for name, df in tables.items():
        df.to_csv(out / f"audit_{name}.csv", index=False)
        print(df.to_string(index=False))
    def corr(xs, ys):
        return None if np.std(xs) == 0 or np.std(ys) == 0 else round(float(np.corrcoef(xs, ys)[0, 1]), 4)
    corr_c = {name: corr([s[name] for s in scored], [float(s["correct"]) for s in scored]) for name in ("teacher_adv", "opsa_adv")}
    corr_len = {name: corr([s[name] for s in scored], [s["n_tokens"] for s in scored]) for name in ("teacher_adv", "opsa_adv")}
    summary = dict(n_responses=len(scored), n_kept=len(kept), correct=round(float(np.mean([s["correct"] for s in scored])), 4),
                   corr_with_correct=corr_c, corr_with_length=corr_len,
                   teacher_adv_correct=round(float(np.mean([s["teacher_adv"] for s in scored if s["correct"]] or [float("nan")])), 4),
                   teacher_adv_wrong=round(float(np.mean([s["teacher_adv"] for s in scored if not s["correct"]] or [float("nan")])), 4),
                   opsa_selected_correct=round(float(np.mean([s["opsa_selected"] for s in kept if s["correct"]] or [float("nan")])), 4),
                   opsa_selected_wrong=round(float(np.mean([s["opsa_selected"] for s in kept if not s["correct"]] or [float("nan")])), 4),
                   seconds=round(time.time() - t0))
    common.write_json(out / "audit.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
