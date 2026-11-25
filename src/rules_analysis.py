"""Experiment 3 analysis (CPU): the shares of the correct, dominant wrong and other answers per operator, arm and
split, before and after training, with the frozen-model noise floor, the logit probe and the training curves.

analyze(runs, items_path, out_dir, show) takes {arm name: rules_eval run dir}; "base_A" (the reference batch) and
"base_B" (the frozen-model control) are required. The dominant wrong answer of an item is the base A batch's most
common wrong answer to it (rules_data.dominant_wrong_of), fixed across the trained arms and base B; base A itself
is scored with base B's map, since a batch's own most common wrong answer is inflated by construction.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

import common
import rules_data as rd

BUCKETS = ((0.0, 0.05, "rare (<5%)"), (0.05, 0.2, "uncommon (5-20%)"), (0.2, 1.01, "common (>=20%)"))


def dominant_map(rows):
    """{item_id: dominant wrong answer or None} from one batch's rows."""
    by_item = {}
    for r in rows:
        by_item.setdefault(r["item_id"], []).append(r)
    return {k: rd.dominant_wrong_of([r["got"] for r in v], v[0]["answer"]) for k, v in by_item.items()}


def classify(rows, dominant):
    for r in rows:
        r["outcome"] = rd.outcome(r["got"], r, dominant.get(r["item_id"]))
    return rows


def item_shares(rows):
    """{(sym, split, item_id): {outcome share..., truncated, mean_tokens, severe_rep}}"""
    by = {}
    for r in rows:
        by.setdefault((r["sym"], r["split"], r["item_id"]), []).append(r)
    out = {}
    for k, v in by.items():
        d = {o: np.mean([r["outcome"] == o for r in v]) for o in rd.OUTCOMES}
        d.update(habit=np.mean([r["got"] is not None and r["got"] == r["habit"] for r in v]),
                 truncated=np.mean([r["truncated"] for r in v]), mean_tokens=np.mean([r["n_tokens"] for r in v]),
                 severe_rep=np.mean([r["severe_rep"] for r in v]), n=len(v))
        out[k] = d
    return out


def op_split_means(shares):
    """{(sym, split): {measure: mean over items}}"""
    by = {}
    for (sym, split, _), d in shares.items():
        by.setdefault((sym, split), []).append(d)
    return {k: {m: float(np.mean([d[m] for d in v])) for m in v[0] if m != "n"} | {"n_items": len(v)} for k, v in by.items()}


def matrix(shares, syms, split, measure, item_ids):
    """[n_ops, n_items] array of per-item values, items aligned across arms by item_ids[sym]."""
    return np.array([[shares[(sym, split, iid)][measure] for iid in item_ids[sym]] for sym in syms])


def boot_delta(a, b, n_boot=2000, seed=0):
    """Mean over operators of the mean over items of (a - b), with a percentile interval that resamples items within
    each operator. a, b: [n_ops, n_items] aligned."""
    d = a - b
    rng = np.random.default_rng(seed)
    n_ops, n_items = d.shape
    idx = rng.integers(0, n_items, size=(n_boot, n_ops, n_items))
    boots = np.take_along_axis(d[None], idx, axis=2).mean(axis=2).mean(axis=1)
    return float(d.mean()), float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def fmt(x, lo=None, hi=None, nd=3):
    s = f"{x:+.{nd}f}" if lo is not None else f"{x:.{nd}f}"
    return s if lo is None else f"{s} [{lo:+.{nd}f}, {hi:+.{nd}f}]"


def train_curves(run_dir):
    p = Path(run_dir) / "train_log.jsonl"
    if not p.exists():
        return None
    return pd.DataFrame(common.read_jsonl(p))


def analyze(runs, items_path, out_dir, show=print, train_dirs=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = {arm: common.read_jsonl(Path(d) / "rollouts.jsonl") for arm, d in runs.items()}
    dominant, dominant_b = dominant_map(rows["base_A"]), dominant_map(rows["base_B"])
    common.write_json(out_dir / "dominant.json", dominant)
    common.write_json(out_dir / "dominant_B.json", dominant_b)
    shares = {arm: item_shares(classify(v, dominant_b if arm == "base_A" else dominant)) for arm, v in rows.items()}
    syms = sorted({r["sym"] for r in rows["base_A"]})
    splits = [s for s in rd.SPLITS if any(k[1] == s for k in shares["base_A"])]
    item_ids = {}
    for split in splits:
        item_ids[split] = {sym: sorted(k[2] for k in shares["base_A"] if k[0] == sym and k[1] == split) for sym in syms}
    for arm, sh in shares.items():
        for split in splits:
            missing = [(sym, iid) for sym in syms for iid in item_ids[split][sym] if (sym, split, iid) not in sh]
            if missing:
                raise ValueError(f"{arm} lacks {len(missing)} items of base_A in {split}, e.g. {missing[0]}")
    means = {arm: op_split_means(sh) for arm, sh in shares.items()}
    arms = list(runs)
    trained = [a for a in arms if a not in ("base_A", "base_B")]
    tables = {}

    # 1. operators: base rates and the dominant wrong answer
    recs = []
    for sym in syms:
        op = rd.BY_SYM[sym]
        m = means["base_A"][(sym, "new_numbers")]
        dom_counts = {}
        for iid in item_ids["new_numbers"][sym]:
            dom_counts[dominant.get(iid)] = dom_counts.get(dominant.get(iid), 0) + 1
        recs.append({"sym": sym, "definition": op["expr"], "expected habit": op["habit"], "base correct": round(m["correct"], 3),
                     "base B correct": round(means["base_B"][(sym, "new_numbers")]["correct"], 3),
                     "base dominant wrong": round(m["dominant_wrong"], 3), "base habit answer": round(m["habit"], 3),
                     "base no answer": round(m["no_answer"], 3), "base mean tokens": round(m["mean_tokens"], 0),
                     "items with a dominant wrong": sum(v for k, v in dom_counts.items() if k is not None)})
    tables["operators"] = pd.DataFrame(recs).sort_values("base correct")

    # 2. the main table: correct share per operator and arm on each split, delta against base A
    for split in splits:
        recs = []
        for sym in sorted(syms, key=lambda s: means["base_A"][(s, "new_numbers")]["correct"]):
            rec = {"sym": sym, "base rate": round(means["base_A"][(sym, "new_numbers")]["correct"], 3)}
            for arm in arms:
                rec[arm] = round(means[arm][(sym, split)]["correct"], 3)
            recs.append(rec)
        rec = {"sym": "mean", "base rate": round(float(np.mean([means["base_A"][(s, "new_numbers")]["correct"] for s in syms])), 3)}
        for arm in arms:
            rec[arm] = round(float(np.mean([means[arm][(s, split)]["correct"] for s in syms])), 3)
        recs.append(rec)
        tables[f"correct_{split}"] = pd.DataFrame(recs)

    # 3. per arm and split: shares with intervals against base A (items resampled within operators)
    recs = []
    for split in splits:
        base = {m: matrix(shares["base_A"], syms, split, m, item_ids[split]) for m in ("correct", "dominant_wrong", "other_wrong", "no_answer")}
        for arm in arms:
            rec = {"split": split, "arm": arm}
            for m in ("correct", "dominant_wrong", "other_wrong", "no_answer"):
                a = matrix(shares[arm], syms, split, m, item_ids[split])
                rec[m] = round(float(a.mean()), 3)
                if arm != "base_A":
                    rec[f"d_{m}"] = fmt(*boot_delta(a, base[m]))
            rec.update({"truncated": round(float(np.mean([means[arm][(s, split)]["truncated"] for s in syms])), 3),
                        "mean tokens": round(float(np.mean([means[arm][(s, split)]["mean_tokens"] for s in syms])), 0),
                        "severe rep": round(float(np.mean([means[arm][(s, split)]["severe_rep"] for s in syms])), 3)})
            recs.append(rec)
    tables["by_split"] = pd.DataFrame(recs)

    # 4. the reading: change in the correct share by initial rarity (base rate buckets), new_numbers
    recs = []
    base_rate = {s: means["base_A"][(s, "new_numbers")]["correct"] for s in syms}
    for lo, hi, name in BUCKETS:
        group = [s for s in syms if lo <= base_rate[s] < hi]
        if not group:
            continue
        for arm in arms:
            if arm == "base_A":
                continue
            rec = {"rarity": name, "operators": len(group), "syms": "".join(group), "arm": arm}
            for m in ("correct", "dominant_wrong"):
                a = matrix(shares[arm], group, "new_numbers", m, item_ids["new_numbers"])
                b = matrix(shares["base_A"], group, "new_numbers", m, item_ids["new_numbers"])
                rec[f"d_{m}"] = fmt(*boot_delta(a, b))
            recs.append(rec)
    tables["reading"] = pd.DataFrame(recs)

    # 5. the noise floor: base B against base A per operator
    recs = []
    for split in splits:
        d = [means["base_B"][(s, split)]["correct"] - means["base_A"][(s, split)]["correct"] for s in syms]
        recs.append({"split": split, "mean |base B - base A| per operator": round(float(np.mean(np.abs(d))), 3),
                     "max |d|": round(float(np.max(np.abs(d))), 3), "mean d": round(float(np.mean(d)), 3)})
    tables["noise"] = pd.DataFrame(recs)

    # 6. probe (logit reads), where present
    recs = []
    for arm, d in runs.items():
        p = Path(d) / "probe.json"
        if not p.exists():
            continue
        s = common.read_json(p)
        for read in ("fixed", "own"):
            if read not in s:
                continue
            o = s[read]["overall"]
            rec = {"arm": arm, "read": read, "n": o["n"], "p(correct)": o["p_correct"], "p(correct) median": o["p_correct_median"],
                   "p(dominant)": o["p_dominant"]}
            for sym in syms:
                if sym in s[read]["per_sym"]:
                    rec[f"p_c {sym}"] = s[read]["per_sym"][sym]["p_correct"]
            recs.append(rec)
    if recs:
        tables["probe"] = pd.DataFrame(recs)

    # 7. training curves summary
    if train_dirs:
        recs, curves = [], []
        for arm, d in train_dirs.items():
            df = train_curves(d)
            if df is None or df.empty:
                continue
            df = df.drop_duplicates("step", keep="last")
            df.insert(0, "arm", arm)
            curves.append(df)
            ok = df[df["skipped"].isna()] if "skipped" in df else df
            if ok.empty:
                continue
            first, last = ok.iloc[0], ok.iloc[-1]
            rec = {"arm": arm, "steps": int(last["step"]), "minutes": round(float(ok["seconds"].sum()) / 60, 1),
                   "updated tokens/step": round(float(ok["n_updated"].mean()), 0), "share updated": round(float((ok["n_updated"] / ok["n_tokens"]).mean()), 3),
                   "mean |A|": round(float(ok["mean_abs_adv"].mean()), 3), "grad norm (pre-clip)": round(float(ok["grad_norm"].mean()), 3),
                   "total update norm": round(float(ok["update_norm"].sum()), 3), "final param norm": round(float(last["param_norm"]), 3)}
            if "correct" in ok:
                rec.update({"rollout correct first/last": f"{first['correct']:.3f} / {last['correct']:.3f}",
                            "rollout habit first/last": f"{first['habit']:.3f} / {last['habit']:.3f}",
                            "truncated first/last": f"{first['truncated']:.3f} / {last['truncated']:.3f}",
                            "mean tokens first/last": f"{first['mean_tokens']:.0f} / {last['mean_tokens']:.0f}"})
            recs.append(rec)
        if recs:
            tables["training"] = pd.DataFrame(recs)
            pd.concat(curves).to_csv(out_dir / "train_curves.csv", index=False)

    for name, df in tables.items():
        df.to_csv(out_dir / f"{name}.csv", index=False)
        show(f"\n--- {name}")
        show(df.to_string(index=False))
    return tables
