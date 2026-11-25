#!/usr/bin/env python
"""Experiments 2 and 7 (supporting evidence): per-problem and pass@k analysis of a trained model against its base.

    uv run src/passk.py --group <base run A> --base <base run B> --trained <trained run> [--sweep T=<run> ...] [--out <dir>]

Three batches per problem, as the plan says: a base batch used only for grouping (all / some / no sampled attempts
succeeded), a second base batch for scoring (the frozen-model control: its difference from the first is noise), and
the trained model's batch. Measures per arm and per group: pass@1, the full pass@k curve (unbiased estimator per
problem, averaged), success at the largest budget, the Sharpening Tax's scalability score

    S(K) = 1/(K-1) * sum_{k=1}^{K-1} (pass@K - pass@k) / (1 - pass@1),   Tax(K) = S_base(K) - S_trained(K)

(the share of a model's successes that come from retries; a positive tax means the trained model relies less on
retries), the probability of the base's dominant wrong answer, the dispersion of errors, and side effects (length,
cap hits, severe repetition, reflection markers). Problems the trained model solves that neither base batch ever
solved are listed for the coverage audit. Intervals are 95% bootstrap over problems. Pure CPU.
"""
import argparse
import json
import re
import sys
from collections import Counter
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

REFLECT = re.compile(r"\b(wait|but|however|hmm|actually|alternatively|let me (?:check|verify|re-?check|double-check)|"
                     r"double-check|re-?check|verify|re-?examine|reconsider)\b", re.I)
# the strict form: a sentence or line that starts with a reflection word, or an explicit self-check
REFLECT_STRICT = re.compile(r"(?:^|[.!?]\s+|\n\s*)(?:\*\*)?(wait|but|hmm|actually|alternatively|however)\b|"
                            r"\blet me (?:check|verify|re-?check|double-check)|\bdouble-check", re.I | re.M)
BUDGETS = (1024, 2048, 4096, 8192, 16384)
MIN_GROUP = 5   # fewer problems than this: intervals and p-values are not reported


# --------------------------------------------------------------------------- per-problem data
def norm_answer(text):
    """The boxed answer as a comparable string (None if no complete box): math-verify's parse when it parses (so
    \\frac{1}{2}, 1/2 and \\dfrac12 agree), else a normalized string."""
    b = common.last_boxed(text)
    if b is None:
        return None
    try:
        from math_verify import parse  # type: ignore

        parsed = parse("\\boxed{" + b + "}", parsing_timeout=2)
        if parsed:
            return "mv:" + re.sub(r"(\.\d*?[1-9])0+\b|\.0+\b", r"\1", str(parsed[0]))
    except Exception:  # noqa: BLE001
        pass
    b = re.sub(r"\\(left|right|!|,|;|quad|displaystyle|\$)", "", b)
    b = re.sub(r"\\(text|mathrm|textbf)\s*\{([^{}]*)\}", r"\2", b)
    b = b.replace("dfrac", "frac").replace("tfrac", "frac").replace("$", "").replace("{,}", "").replace(",", "").replace(" ", "")
    b = re.sub(r"^[a-zA-Z]\s*=", "", b)
    b = re.sub(r"(\.\d*?[1-9])0+$|\.0+$", r"\1", b)
    return b.strip(" .") or None


def severe_repetition(ids, n=20, times=3):
    """Some 20-token span occurs at least `times` times (the SUM report's definition)."""
    return repeated_share(ids, n) * len(ids) / n >= times if len(ids) >= n * times else False


def repeated_share(ids, n=20):
    """Share of the response's tokens covered by repeats of its most frequent n-gram (0 when none repeats)."""
    if len(ids) < 2 * n:
        return 0.0
    counts = {}
    for i in range(len(ids) - n + 1):
        g = tuple(ids[i:i + n])
        counts[g] = counts.get(g, 0) + 1
    top = max(counts.values())
    return 0.0 if top < 2 else min(1.0, top * n / len(ids))


def per_problem(rollouts):
    """{problem_id: dict(n, c, answers=[canonical or None], correct=[bool], tokens, truncated, repetition, heavy, reflect,
    reflect_strict, level)}"""
    out = {}
    for r in rollouts:
        d = out.setdefault(r["problem_id"], dict(n=0, c=0, answers=[], correct=[], tokens=[], truncated=[], repetition=[], heavy=[],
                                                 reflect=[], reflect_strict=[], level=r["level"]))
        d["n"] += 1
        d["c"] += int(r["correct"])
        d["answers"].append(norm_answer(r["text"]))
        d["correct"].append(bool(r["correct"]))
        d["tokens"].append(r["n_tokens"])
        d["truncated"].append(bool(r["truncated"]))
        share = repeated_share(r["response_ids"])
        d["repetition"].append(share * r["n_tokens"] / 20 >= 3)
        d["heavy"].append(share >= 0.2)
        d["reflect"].append(len(REFLECT.findall(r["text"])))
        d["reflect_strict"].append(len(REFLECT_STRICT.findall(r["text"])))
    return out


def dominant_wrong(d):
    """The most common wrong answer in a batch (None if every attempt was right or unboxed)."""
    wrong = Counter(a for a, ok in zip(d["answers"], d["correct"]) if not ok and a is not None)
    return wrong.most_common(1)[0][0] if wrong else None


def pass_at_k(n, c, k):
    """Unbiased pass@k for one problem with c correct of n samples."""
    if k > n:
        return np.nan
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


# --------------------------------------------------------------------------- statistics
def boot_diff(a, b, n_boot=2000, seed=0):
    """Paired over problems: mean(a - b) with a 95% interval and a two-sided bootstrap p."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = ~(np.isnan(a) | np.isnan(b))
    d = a[ok] - b[ok]
    if d.size == 0:
        return np.nan, np.nan, np.nan, np.nan
    idx = np.random.default_rng(seed).integers(0, len(d), (n_boot, len(d)))
    bs = d[idx].mean(1)
    p = min(1.0, 2 * min((bs <= 0).mean(), (bs >= 0).mean()))
    return float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), float(p)


def fmt(m, lo, hi, d=3):
    return "n/a" if pd.isna(m) else f"{m:+.{d}f} [{lo:+.{d}f}, {hi:+.{d}f}]"


def tax_boot(pp_a, pp_b, qs, K, n_boot=1000, seed=0):
    """Tax(K) = S_b(K) - S_a(K) on problems qs, with a bootstrap interval over problems (both curves recomputed per draw)."""
    qs = list(qs)
    if len(qs) < MIN_GROUP:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    pa = np.array([[pass_at_k(pp_a[q]["n"], pp_a[q]["c"], k) for k in range(1, K + 1)] for q in qs])
    pb = np.array([[pass_at_k(pp_b[q]["n"], pp_b[q]["c"], k) for k in range(1, K + 1)] for q in qs])

    def tax(idx):
        ca, cb = pa[idx].mean(0), pb[idx].mean(0)
        sa, sb = scalability(dict(enumerate(ca, 1)), K), scalability(dict(enumerate(cb, 1)), K)
        return sb - sa
    point = tax(np.arange(len(qs)))
    draws = np.array([tax(rng.integers(0, len(qs), len(qs))) for _ in range(n_boot)])
    draws = draws[~np.isnan(draws)]
    if np.isnan(point) or draws.size == 0:
        return np.nan, np.nan, np.nan
    return float(point), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def curve(pp, ks, problems):
    """Dataset pass@k for each k: mean over problems of the per-problem estimator."""
    return {k: float(np.nanmean([pass_at_k(pp[q]["n"], pp[q]["c"], k) for q in problems])) for k in ks}


def scalability(cv, K):
    """S(K) from a pass@k curve dict; nan if pass@1 is 1 (nothing left to gain from retries)."""
    p1 = cv[1]
    if K < 2 or np.isnan(cv[K]) or 1 - p1 <= 1e-12:
        return np.nan
    return float(np.mean([(cv[K] - cv[k]) / (1 - p1) for k in range(1, K)]))


# --------------------------------------------------------------------------- tables
def groups_of(pp_group):
    g = {}
    for q, d in pp_group.items():
        g[q] = "all solved" if d["c"] == d["n"] else "none solved" if d["c"] == 0 else "some solved"
    return g


def arm_table(arms, groups, K):
    """One row per arm and group (plus 'all problems'): pass@1, pass@K, S(K), length, cap, repetition, reflection."""
    rows = []
    for name, pp in arms.items():
        probs = sorted(pp)
        for gname in ("all problems", "all solved", "some solved", "none solved"):
            qs = probs if gname == "all problems" else [q for q in probs if groups.get(q) == gname]
            if not qs:
                continue
            ks = list(range(1, K + 1))
            cv = curve(pp, ks, qs)
            toks = [t for q in qs for t in pp[q]["tokens"]]
            rows.append({"arm": name, "group (by base batch A)": gname, "problems": len(qs),
                         "pass@1": round(cv[1], 3), f"pass@{K}": round(cv[K], 3), f"S({K})": round(scalability(cv, K), 3),
                         "mean tokens": round(float(np.mean(toks))), "at cap": round(float(np.mean([t for q in qs for t in pp[q]["truncated"]])), 3),
                         "repetition (a 20-gram x3)": round(float(np.mean([t for q in qs for t in pp[q]["repetition"]])), 3),
                         "heavy repetition (>=20% of tokens)": round(float(np.mean([t for q in qs for t in pp[q]["heavy"]])), 3),
                         "reflection markers / 1k tokens": round(1000 * float(np.sum([t for q in qs for t in pp[q]["reflect"]])) / max(np.sum(toks), 1), 2),
                         "strict reflection / 1k tokens": round(1000 * float(np.sum([t for q in qs for t in pp[q]["reflect_strict"]])) / max(np.sum(toks), 1), 2)})
    return pd.DataFrame(rows)


def budget_table(arms, K):
    """pass@1 and pass@K per arm when a response only counts if it finished within B tokens: the matched-budget view.
    Budgets above an arm's own cap are left out when any of its responses hit that cap (their outcome is unknown)."""
    rows = []
    for name, pp in arms.items():
        cap = max(t for d in pp.values() for t in d["tokens"])
        any_trunc = any(t for d in pp.values() for t in d["truncated"])
        for B in BUDGETS:
            if B > cap and any_trunc:
                continue
            est = {}
            for k in (1, K):
                vals = []
                for d in pp.values():
                    c = sum(1 for ok, t in zip(d["correct"], d["tokens"]) if ok and t <= B)
                    vals.append(pass_at_k(d["n"], c, k))
                est[k] = float(np.nanmean(vals))
            rows.append({"arm": name, "budget (tokens)": B, "pass@1": round(est[1], 3), f"pass@{K}": round(est[K], 3)})
    return pd.DataFrame(rows)


def curve_table(arms, groups, K):
    rows = []
    for name, pp in arms.items():
        cv = curve(pp, range(1, K + 1), sorted(pp))
        rows.append({"arm": name, **{f"pass@{k}": round(v, 3) for k, v in cv.items()}})
    return pd.DataFrame(rows)


def contrast_table(arms, base_name, groups, K, pairs=None):
    """Each other arm against the scoring base batch (or the given (arm, reference) pairs), paired over problems."""
    pairs = pairs or [(name, base_name) for name in arms if name != base_name]
    rows = []
    for name, ref in pairs:
        pp, base = arms[name], arms[ref]
        qs = sorted(set(pp) & set(base))
        for gname in ("all problems", "all solved", "some solved", "none solved"):
            sub = qs if gname == "all problems" else [q for q in qs if groups.get(q) == gname]
            if not sub:
                continue
            row = {"arm": name, "against": ref, "group": gname, "problems": len(sub)}
            if len(sub) < MIN_GROUP:
                row.update({"Δ pass@1 [95% CI]": f"{np.mean([pp[q]['c'] / pp[q]['n'] for q in sub]) - np.mean([base[q]['c'] / base[q]['n'] for q in sub]):+.3f} (too few problems for an interval)",
                            "p (pass@1)": "n/a", f"Δ pass@{K} [95% CI]": "n/a", f"p (pass@{K})": "n/a", f"Tax({K}) = S_ref - S_arm [95% CI]": "n/a"})
            else:
                p1 = boot_diff([pp[q]["c"] / pp[q]["n"] for q in sub], [base[q]["c"] / base[q]["n"] for q in sub])
                pK = boot_diff([pass_at_k(pp[q]["n"], pp[q]["c"], K) for q in sub], [pass_at_k(base[q]["n"], base[q]["c"], K) for q in sub])
                row.update({"Δ pass@1 [95% CI]": fmt(*p1[:3]), "p (pass@1)": round(p1[3], 3), f"Δ pass@{K} [95% CI]": fmt(*pK[:3]),
                            f"p (pass@{K})": round(pK[3], 3), f"Tax({K}) = S_ref - S_arm [95% CI]": fmt(*tax_boot(pp, base, sub, K))})
            rows.append(row)
    return pd.DataFrame(rows)


def transfer_table(arms, group_name, base_name, trained_name):
    """Per-problem movement of the trained model against the base: newly solved (never by either base batch), lost,
    dominant-wrong answer mass, error dispersion; and the same counts between the two base batches as the noise
    baseline."""
    g, b, t = arms[group_name], arms[base_name], arms[trained_name]
    qs = sorted(set(g) & set(b) & set(t))
    new = [q for q in qs if g[q]["c"] == 0 and b[q]["c"] == 0 and t[q]["c"] > 0]
    lost = [q for q in qs if (g[q]["c"] + b[q]["c"]) > 0 and t[q]["c"] == 0]
    noise_new = [q for q in qs if g[q]["c"] == 0 and b[q]["c"] > 0]      # B "discovers" what A never solved
    noise_lost = [q for q in qs if g[q]["c"] > 0 and b[q]["c"] == 0]     # B "loses" what A solved
    dom_rows = []
    for q in qs:
        dw = dominant_wrong(g[q])
        if dw is None:
            continue
        mass = lambda d: float(np.mean([a == dw for a in d["answers"]]))  # noqa: E731
        distinct = lambda d: len({a for a, ok in zip(d["answers"], d["correct"]) if not ok and a is not None})  # noqa: E731
        dom_rows.append(dict(problem=q, base_group=g[q]["c"] / g[q]["n"], dominant_wrong_mass_base=mass(b[q]), dominant_wrong_mass_trained=mass(t[q]),
                             distinct_wrong_base=distinct(b[q]), distinct_wrong_trained=distinct(t[q]),
                             no_answer_base=float(np.mean([a is None for a in b[q]["answers"]])),
                             no_answer_trained=float(np.mean([a is None for a in t[q]["answers"]]))))
    dom = pd.DataFrame(dom_rows)
    summary = {}
    if not dom.empty:
        d = boot_diff(dom.dominant_wrong_mass_trained, dom.dominant_wrong_mass_base)
        summary = {"problems with a dominant wrong answer in batch A": len(dom),
                   "mass on it, base B": round(float(dom.dominant_wrong_mass_base.mean()), 3),
                   "mass on it, trained": round(float(dom.dominant_wrong_mass_trained.mean()), 3),
                   "Δ mass [95% CI]": fmt(*d[:3]) if len(dom) >= MIN_GROUP else "n/a", "p": round(d[3], 3) if len(dom) >= MIN_GROUP else "n/a",
                   "distinct wrong answers per problem, base B": round(float(dom.distinct_wrong_base.mean()), 2),
                   "distinct wrong answers per problem, trained": round(float(dom.distinct_wrong_trained.mean()), 2),
                   "no-answer share, base B": round(float(dom.no_answer_base.mean()), 3),
                   "no-answer share, trained": round(float(dom.no_answer_trained.mean()), 3)}
    summary.update({"problems": len(qs), "apparently newly solved (audit these)": len(new), "newly solved ids": ",".join(new),
                    "noise baseline: solved by base B, never by base A": len(noise_new),
                    "lost (base solved, trained never)": len(lost), "lost ids": ",".join(lost),
                    "noise baseline: solved by base A, never by base B": len(noise_lost)})
    return pd.DataFrame([summary]), dom


def per_problem_table(arms, group_name):
    rows = []
    qs = sorted(set.intersection(*[set(pp) for pp in arms.values()]))
    for q in qs:
        row = {"problem": q, "level": arms[group_name][q]["level"]}
        for name, pp in arms.items():
            row[f"{name}: correct/n"] = f"{pp[q]['c']}/{pp[q]['n']}"
            row[f"{name}: mean tokens"] = round(float(np.mean(pp[q]["tokens"])))
        rows.append(row)
    return pd.DataFrame(rows)


def analyze(group_run, base_run, trained_run, sweeps, K, out_dir=None, show=print):
    arms = {"base A (grouping)": per_problem(common.read_jsonl(Path(group_run) / "rollouts.jsonl")),
            "base B (scoring)": per_problem(common.read_jsonl(Path(base_run) / "rollouts.jsonl")),
            "trained": per_problem(common.read_jsonl(Path(trained_run) / "rollouts.jsonl"))}
    for label, run in sweeps.items():
        arms[f"base T={label}"] = per_problem(common.read_jsonl(Path(run) / "rollouts.jsonl"))
    sets = {name: set(pp) for name, pp in arms.items()}
    if len({frozenset(v) for v in sets.values()}) != 1:
        sizes = {name: len(v) for name, v in sets.items()}
        raise ValueError(f"the arms do not share one problem set ({sizes}; intersection {len(set.intersection(*sets.values()))}); "
                         "all arms need the same --problem-seed and --n-problems")
    groups = groups_of(arms["base A (grouping)"])
    tables = {"arms": arm_table(arms, groups, K), "curves": curve_table(arms, groups, K), "budgets": budget_table(arms, K),
              "contrasts": contrast_table(arms, "base B (scoring)", groups, K)}
    tables["transfer"], tables["dominant_wrong"] = transfer_table(arms, "base A (grouping)", "base B (scoring)", "trained")
    tables["per_problem"] = per_problem_table(arms, "base A (grouping)")
    if sweeps:
        # the matched temperature is chosen on half the problems (held-out tuning) and reported on the other half,
        # and also chosen on all problems (in-sample) for the full-set contrast
        qs = sorted(arms["trained"])
        tune, test = qs[1::2], qs[0::2]
        rate = lambda pp, sub: float(np.mean([pp[q]["c"] / pp[q]["n"] for q in sub]))  # noqa: E731

        def closest(sub):
            target = rate(arms["trained"], sub)
            return min(sweeps, key=lambda lbl: abs(rate(arms[f"base T={lbl}"], sub) - target)), target
        t_held, target_held = closest(tune)
        t_all, target_all = closest(qs)
        tables["sweep_match"] = pd.DataFrame([
            {"choice": "held out: T chosen on odd problems, reported on even", "temperature": t_held, "trained pass@1 (tuning half)": round(target_held, 3),
             f"base T={t_held} pass@1 (tuning half)": round(rate(arms[f"base T={t_held}"], tune), 3)},
            {"choice": "in sample: T chosen on all problems", "temperature": t_all, "trained pass@1 (tuning half)": round(target_all, 3),
             f"base T={t_held} pass@1 (tuning half)": round(rate(arms[f"base T={t_all}"], qs), 3)}])
        held = {name: {q: d for q, d in pp.items() if q in set(test)} for name, pp in arms.items()}
        tables["trained_vs_matched_T"] = pd.concat([
            contrast_table(held, f"base T={t_held}", groups, K, pairs=[("trained", f"base T={t_held}")]).assign(problems_used="even (held out)"),
            contrast_table(arms, f"base T={t_all}", groups, K, pairs=[("trained", f"base T={t_all}")]).assign(problems_used="all (in sample)")],
            ignore_index=True)
    if out_dir:
        out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        for name, t in tables.items():
            t.to_csv(out_dir / f"{name}.csv", index=False)
    if show:
        pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40); pd.set_option("display.max_colwidth", 60)
        for name in ("arms", "curves", "budgets", "contrasts", "transfer", "sweep_match", "trained_vs_matched_T"):
            if name in tables:
                show(f"\n** {name} **"); show(tables[name].to_string(index=False))
    return tables


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--group", required=True, help="base batch A, grouping only")
    ap.add_argument("--base", required=True, help="base batch B, scoring (frozen-model control)")
    ap.add_argument("--trained", required=True)
    ap.add_argument("--sweep", action="append", default=[], metavar="T=RUN", help="base at another temperature")
    ap.add_argument("--k", type=int, default=8, help="largest k of the pass@k curve (at most the batch size)")
    ap.add_argument("--out", default=None, help="where to write the CSV tables")
    args = ap.parse_args(argv)
    sweeps = dict(s.split("=", 1) for s in args.sweep)
    analyze(args.group, args.base, args.trained, sweeps, args.k, args.out)


if __name__ == "__main__":
    main()
