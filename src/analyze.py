#!/usr/bin/env python
"""Analysis (laptop or Colab, no GPU): how useful are the tokens each rule pushes down?

    uv run src/analyze.py --run <run dir>

Per replayed position t with kept token y and token probability p:
  Q_kept      share of kept continuations that end correct
  Q_replaced  the same for the replacement tokens (drawn from p with y removed)
  usefulness  U = Q_kept - Q_replaced. U > 0: the token leads to more correct answers than what replaces it,
              so pushing it down hurts. U < 0: pushing it down helps.
  damage      D = p (1 - p) U, the first-order drop in expected success from one unit of negative step on the
              token's logit (derivation in the notebook)
Means carry 95% bootstrap intervals that resample problems, because positions from one problem are not
independent. Every table is also written to <run>/tables/<name>.csv.

The share of tokens with U > 0 is read against the share sampling noise alone would give (share_null table): at
8 + 8 continuations a single position's U is a coarse estimate, so part of any share is chance.

    uv run src/analyze.py --positions-csv <tables/positions.csv>   # the replay tables from the committed CSV alone
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

RULES = {"opsa": "OPSA: lowest 20% log-prob", "random": "Random 20%", "teacher_neg": "Teacher-negative 20%",
         "nsr": "NSR: failed responses"}
DOUBT = {"wait", "but", "however", "alternatively", "hmm", "actually", "maybe", "perhaps", "though", "although",
         "instead", "check", "verify", "recheck", "double", "mistake", "wrong", "oops", "let"}


# --------------------------------------------------------------------------- loading
def load(run):
    run = Path(run)
    out = dict(rollouts=common.read_jsonl(run / "rollouts.jsonl"))
    for name in ("scores", "selected", "positions"):
        p = run / f"{name}.jsonl"
        out[name] = common.read_jsonl(p) if p.exists() else None
    rep = run / "replays.jsonl"
    rep = rep if rep.exists() else run / "replays.jsonl.partial"
    out["replays"] = common.read_jsonl(rep) if rep.exists() else None
    return out


def position_frame(positions, replays, rollouts):
    """One row per replayed position (positions not yet replayed are left out, so a partial run works)."""
    rates = pd.DataFrame(rollouts).groupby("problem_id")["correct"].mean()
    rep = pd.DataFrame(replays)
    rep["correct"] = rep["correct"].astype(float)
    q = rep.pivot_table(index="pos_id", columns="kind", values="correct", aggfunc="mean")
    n = rep.pivot_table(index="pos_id", columns="kind", values="correct", aggfunc="count")
    pf = pd.DataFrame(positions).set_index("pos_id")
    for kind in ("kept", "redrawn", "dup"):
        pf[f"q_{kind}"] = q[kind] if kind in q else np.nan
        pf[f"n_{kind}"] = (n[kind] if kind in n else pd.Series(dtype=float)).reindex(pf.index).fillna(0).astype(int)
    pf["cont_truncated"] = rep.groupby("pos_id")["truncated"].mean()
    pf = pf[pf.q_kept.notna() & pf.q_redrawn.notna()].copy()
    pf["usefulness"] = pf.q_kept - pf.q_redrawn
    pf["usefulness_dup"] = pf.q_dup - pf.q_redrawn
    pf["damage"] = pf.p_kept * (1 - pf.p_kept) * pf.usefulness
    pf["useful"] = (pf.usefulness > 0).astype(float)
    pf["harmful"] = (pf.usefulness < 0).astype(float)
    pf["rel_t"] = pf.t / pf.n_tokens
    pf["problem_rate"] = pf.problem_id.map(rates)
    pf["stratum"] = np.select([pf.problem_rate == 1, pf.problem_rate == 0], ["all correct", "none correct"], "some correct")
    return pf.reset_index()


def rule_rows(pf, rule):
    return pf[pf.methods.apply(lambda ms: rule in ms)]


# --------------------------------------------------------------------------- bootstrap over problems
def _sums(df, col, clusters):
    g = df.dropna(subset=[col]).groupby("problem_id")[col].agg(["sum", "count"]).reindex(clusters, fill_value=0)
    return g["sum"].to_numpy(float), g["count"].to_numpy(float)


def _weights(n_clusters, n_boot, seed):
    return np.random.default_rng(seed).multinomial(n_clusters, np.full(n_clusters, 1 / n_clusters), size=n_boot)


def mean_ci(df, col, n_boot=2000, seed=0):
    """(mean, lo, hi): problems resampled with replacement."""
    vals = df[col].dropna()
    if vals.empty:
        return np.nan, np.nan, np.nan
    cl = df.problem_id.unique()
    s, c = _sums(df, col, cl)
    w = _weights(len(cl), n_boot, seed)
    with np.errstate(invalid="ignore", divide="ignore"):
        boots = (w @ s) / (w @ c)
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return vals.mean(), lo, hi


def diff_ci(a, b, col, n_boot=2000, seed=0):
    """(mean_a - mean_b, lo, hi, two-sided bootstrap p). Problems are resampled jointly, so a position shared by
    both rules counts on both sides of the same draw."""
    cl = np.union1d(a.problem_id.unique(), b.problem_id.unique())
    sa, ca = _sums(a, col, cl)
    sb, cb = _sums(b, col, cl)
    w = _weights(len(cl), n_boot, seed)
    with np.errstate(invalid="ignore", divide="ignore"):
        d = (w @ sa) / (w @ ca) - (w @ sb) / (w @ cb)
    d = d[np.isfinite(d)]
    lo, hi = np.percentile(d, [2.5, 97.5])
    p = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))
    return a[col].mean() - b[col].mean(), lo, hi, p


def fmt_ci(m, lo, hi, digits=3):
    return "n/a" if pd.isna(m) else f"{m:+.{digits}f} [{lo:+.{digits}f}, {hi:+.{digits}f}]"


# --------------------------------------------------------------------------- tables
def rollout_table(rollouts, scores=None):
    df = pd.DataFrame(rollouts)
    per = df.groupby("problem_id").agg(level=("level", "first"), rate=("correct", "mean"))
    rows = []
    for lvl, g in sorted(df.groupby("level")):
        p = per[per.level == lvl]
        rows.append(dict(level=lvl, problems=len(p), responses=len(g), pass1=round(g.correct.mean(), 3),
                         truncated=round(g.truncated.mean(), 3), mean_tokens=round(g.n_tokens.mean()),
                         all_correct=int((p.rate == 1).sum()), some_correct=int(((p.rate > 0) & (p.rate < 1)).sum()),
                         none_correct=int((p.rate == 0).sum())))
    t = pd.DataFrame(rows)
    tot = dict(level="all", problems=len(per), responses=len(df), pass1=round(df.correct.mean(), 3),
               truncated=round(df.truncated.mean(), 3), mean_tokens=round(df.n_tokens.mean()),
               all_correct=int((per.rate == 1).sum()), some_correct=int(((per.rate > 0) & (per.rate < 1)).sum()),
               none_correct=int((per.rate == 0).sum()))
    return pd.concat([t, pd.DataFrame([tot])], ignore_index=True)


def logprob_agreement(rollouts, scores):
    """vLLM's sampled-token log-probs against the teacher-forced ones the rules use (same model, same tokens)."""
    sc = {r["key"]: r["lp_s"] for r in scores}
    diffs = [abs(a - b) for r in rollouts for a, b in zip(r["vllm_lp"], sc[r["key"]]) if a is not None]
    if not diffs:
        return dict(n=0, mean_abs=np.nan, p99_abs=np.nan)
    d = np.asarray(diffs)
    return dict(n=len(d), mean_abs=float(d.mean()), p99_abs=float(np.percentile(d, 99)))


def usefulness_table(pf, n_boot=2000):
    rows = []
    for rule, label in RULES.items():
        g = rule_rows(pf, rule)
        if g.empty:
            continue
        rows.append({
            "rule": label, "positions": len(g), "problems": g.problem_id.nunique(),
            "median p(token)": round(g.p_kept.median(), 3), "in correct responses": round(g.response_correct.mean(), 2),
            "Q_kept": round(g.q_kept.mean(), 3), "Q_replaced": round(g.q_redrawn.mean(), 3),
            "usefulness U [95% CI]": fmt_ci(*mean_ci(g, "usefulness", n_boot)),
            "share U>0": round(g.useful.mean(), 2), "share U<0": round(g.harmful.mean(), 2),
            "damage p(1-p)U [95% CI]": fmt_ci(*mean_ci(g, "damage", n_boot), digits=4),
        })
    return pd.DataFrame(rows)


def contrast_table(pf, n_boot=2000):
    base = rule_rows(pf, "random")
    rows = []
    for rule in ("opsa", "teacher_neg", "nsr"):
        g = rule_rows(pf, rule)
        if g.empty or base.empty:
            continue
        for col in ("usefulness", "damage"):
            d, lo, hi, p = diff_ci(g, base, col, n_boot)
            rows.append({"contrast": f"{RULES[rule]} minus random", "measure": col,
                         "difference [95% CI]": fmt_ci(d, lo, hi, 3 if col == "usefulness" else 4),
                         "bootstrap p": round(p, 3)})
    return pd.DataFrame(rows)


def strata_table(pf):
    rows = []
    for rule, label in RULES.items():
        g = rule_rows(pf, rule)
        for col, vals in (("response_correct", (True, False)), ("stratum", ("all correct", "some correct", "none correct"))):
            for v in vals:
                s = g[g[col] == v]
                if len(s):
                    name = {True: "correct response", False: "failed response"}.get(v, v) if col == "response_correct" else f"problem: {v}"
                    rows.append(dict(rule=label, subset=name, positions=len(s), usefulness=round(s.usefulness.mean(), 3),
                                     share_useful=round(s.useful.mean(), 2), share_harmful=round(s.harmful.mean(), 2)))
    return pd.DataFrame(rows)


def noise_table(pf, kept_n, seed=0):
    """Two independent kept passes at the same positions. Expected |difference| if the only noise were binomial
    sampling of kept_n continuations, simulated at each position's pooled rate."""
    d = pf[pf.q_dup.notna()]
    if d.empty:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    pooled = ((d.q_kept + d.q_dup) / 2).to_numpy()
    sim = np.abs(rng.binomial(kept_n, pooled, (4000, len(d))) - rng.binomial(kept_n, pooled, (4000, len(d)))) / kept_n
    sign = lambda x: np.sign(np.round(x, 9))  # noqa: E731
    return pd.DataFrame([{
        "positions with a second kept pass": len(d),
        "mean |Q_kept - Q_dup|": round(float((d.q_kept - d.q_dup).abs().mean()), 3),
        "expected from sampling alone": round(float(sim.mean()), 3),
        "corr(Q_kept, Q_dup)": round(float(d.q_kept.corr(d.q_dup)), 2) if len(d) > 2 else np.nan,
        "sign of U agrees": round(float((sign(d.usefulness) == sign(d.usefulness_dup)).mean()), 2),
    }])


P_BANDS = ((0.0, 0.2, "p < 0.2"), (0.2, 0.5, "0.2 to 0.5"), (0.5, 0.8, "0.5 to 0.8"), (0.8, 0.99, "0.8 to 0.99"),
           (0.99, 1.01, "0.99 and up"))


def null_shares(g, n_boot=4000, seed=0):
    """What sampling alone would give for the shares of U > 0 and U < 0 if every token had U = 0: two binomial
    draws per position at its pooled success rate, with the run's own kept and redrawn counts.
    Returns ((mean, lo, hi) for U > 0, (mean, lo, hi) for U < 0), lo and hi the 95% band."""
    rng = np.random.default_rng(seed)
    nk, nr = g.n_kept.to_numpy(int), g.n_redrawn.to_numpy(int)
    pooled = ((g.q_kept * nk + g.q_redrawn * nr) / (nk + nr)).to_numpy()
    u = rng.binomial(nk, pooled, (n_boot, len(g))) / nk - rng.binomial(nr, pooled, (n_boot, len(g))) / nr
    return tuple((float(s.mean()), *np.percentile(s, [2.5, 97.5]).tolist()) for s in ((u > 0).mean(1), (u < 0).mean(1)))


def share_null_table(pf, n_boot=4000, seed=0):
    """The share of a rule's tokens with U > 0 (and U < 0) next to the share sampling noise alone would give if
    every token had U = 0 (null_shares). A share inside the null band is not evidence of useful or harmful tokens.
    The last columns split the spread of U into sampling noise and what is left for real differences between
    tokens: sd of true U = sqrt(var(U) - mean binomial variance), 0 when nothing is left."""
    rows = []
    for rule, label in RULES.items():
        g = rule_rows(pf, rule)
        if len(g) < 2:
            continue
        (pm, plo, phi), (nm, nlo, nhi) = null_shares(g, n_boot, seed)
        nk, nr = g.n_kept.to_numpy(int), g.n_redrawn.to_numpy(int)
        noise = float((g.q_kept * (1 - g.q_kept) / np.maximum(nk - 1, 1)
                       + g.q_redrawn * (1 - g.q_redrawn) / np.maximum(nr - 1, 1)).mean())
        var_u = float(g.usefulness.var())
        rows.append({"rule": label, "positions": len(g),
                     "share U>0": round(g.useful.mean(), 2), "null share U>0 [95%]": f"{pm:.2f} [{plo:.2f}, {phi:.2f}]",
                     "share U<0": round(g.harmful.mean(), 2), "null share U<0 [95%]": f"{nm:.2f} [{nlo:.2f}, {nhi:.2f}]",
                     "sd(U)": round(np.sqrt(var_u), 3), "sampling sd": round(np.sqrt(noise), 3),
                     "sd of true U (est.)": round(np.sqrt(max(var_u - noise, 0.0)), 3)})
    return pd.DataFrame(rows)


def band_table(pf):
    """Each rule's replayed tokens by the student's probability of the kept token. Separates what a rule selects
    at p near 1, where a push barely moves the token and a replacement is off-distribution, from what it selects
    where a push has leverage."""
    rows = []
    for rule, label in RULES.items():
        g = rule_rows(pf, rule)
        for lo, hi, name in P_BANDS:
            s = g[(g.p_kept >= lo) & (g.p_kept < hi)]
            if len(s):
                rows.append({"rule": label, "p(token)": name, "positions": len(s),
                             "in correct responses": round(s.response_correct.mean(), 2),
                             "Q_kept": round(s.q_kept.mean(), 3), "Q_replaced": round(s.q_redrawn.mean(), 3),
                             "usefulness": round(s.usefulness.mean(), 3), "damage": round(s.damage.mean(), 4),
                             "share U>0": round(s.useful.mean(), 2), "share U<0": round(s.harmful.mean(), 2)})
    return pd.DataFrame(rows)


def replay_tables(pf, kept_n, n_boot=2000):
    """Every table that needs only the position frame (stage 3 done)."""
    return {"usefulness": usefulness_table(pf, n_boot), "contrasts": contrast_table(pf, n_boot),
            "strata": strata_table(pf), "noise": noise_table(pf, kept_n),
            "share_null": share_null_table(pf), "bands": band_table(pf)}


def position_frame_from_csv(path):
    """tables/positions.csv read back with its list columns restored, so the replay tables can be rebuilt from
    the committed CSV when the run folder is not at hand."""
    import ast

    pf = pd.read_csv(path)
    for col in ("methods", "in_sets", "alt_strs"):
        if col in pf:
            pf[col] = pf[col].apply(lambda s: ast.literal_eval(s) if isinstance(s, str) and s.startswith("[") else s)
    return pf


def _category(s):
    if s in common.END_TOKENS:
        return "end token"
    w = s.strip()
    if not w:
        return "space/newline"
    if w.isdigit():
        return "digit"
    if w.lower().strip(",.:;!?") in DOUBT:
        return "doubt/fork word"
    if any(ch in w for ch in "\\^_{}=+-*/()<>|$"):
        return "math/LaTeX"
    if w.isalpha():
        return "word"
    return "punctuation/other"


def selection_tables(scores, selected):
    """What each rule selects, over its whole set (not only the replayed sample): token categories, probability,
    position in the response, and overlap between rules."""
    tokens = {r["key"]: r["tokens"] for r in scores}
    lp = {r["key"]: np.asarray(r["lp_s"]) for r in scores}
    sets = {"all tokens": {r["key"]: range(r["n"]) for r in selected}}
    for rule in ("opsa", "random", "teacher_neg"):
        sets[rule] = {r["key"]: r[rule] for r in selected}
    sets["nsr"] = {r["key"]: range(r["n"]) if not r["correct"] else [] for r in selected}
    cats, desc = {}, []
    for name, s in sets.items():
        c, ps, rel = {}, [], []
        n = 0
        for key, ts in s.items():
            L = len(tokens[key])
            for t in ts:
                cat = _category(tokens[key][t])
                c[cat] = c.get(cat, 0) + 1
                ps.append(lp[key][t])
                rel.append(t / L)
                n += 1
        label = RULES.get(name, name)
        cats[label] = {k: v / max(n, 1) for k, v in c.items()}
        desc.append({"rule": label, "tokens": n, "median p(token)": round(float(np.exp(np.median(ps))), 3) if ps else np.nan,
                     "mean position in response": round(float(np.mean(rel)), 2) if rel else np.nan})
    cat_t = pd.DataFrame(cats).fillna(0).round(3)
    cat_t = cat_t.loc[cat_t.max(axis=1).sort_values(ascending=False).index]
    rules = ("opsa", "random", "teacher_neg", "nsr")
    flat = {rule: {(k, t) for k, ts in sets[rule].items() for t in ts} for rule in rules}
    ov = pd.DataFrame({RULES[b]: {RULES[a]: round(len(flat[a] & flat[b]) / max(len(flat[a]), 1), 3) for a in rules}
                       for b in rules})
    return pd.DataFrame(desc), cat_t, ov


def examples(pf, scores, rule="opsa", k=5, width=12):
    """The k most useful and k most harmful replayed tokens of one rule, each with `width` tokens of context on
    either side, the kept token in [brackets] and the most likely replacement after the arrow."""
    tokens = {r["key"]: r["tokens"] for r in scores}
    g = rule_rows(pf, rule).sort_values("usefulness")
    rows = []
    for _, r in pd.concat([g.tail(k).iloc[::-1], g.head(k)]).iterrows():
        tk = tokens[r.key]
        ctx = "".join(tk[max(0, r.t - width):r.t]) + "[" + tk[r.t] + "]" + "".join(tk[r.t + 1:r.t + 1 + width])
        alt = r.top_alts[0][1] if isinstance(r.top_alts, list) and r.top_alts else ""
        rows.append({"U": round(r.usefulness, 2), "p": round(r.p_kept, 3), "Q_kept": r.q_kept, "Q_replaced": r.q_redrawn,
                     "context": ctx.replace("\n", "\\n"), "top replacement": alt.replace("\n", "\\n")})
    return pd.DataFrame(rows)


def mixed_problems(rollouts_table):
    """(mixed, total, per-level text) from rollout_table(): problems with some but not all answers correct."""
    tot = rollouts_table[rollouts_table.level == "all"].iloc[0]
    per = ", ".join(f"level {r.level}: {r.some_correct}/{r.problems}"
                    for r in rollouts_table[rollouts_table.level != "all"].itertuples())
    return int(tot.some_correct), int(tot.problems), per


def gates(tables, pf=None, agreement=None, min_mixed=20, n_boot=2000):
    """The pilot's pass/fail checks, in order: data, wiring, noise, signal. A gate appears once its inputs
    exist, so right after stage 1 the table holds gate 0 alone. The signal gates (4, 5) look at OPSA's own set:
    the contrast with random tokens is reported in the contrasts table but is no gate, because random tokens sit
    at p near 1, where a replacement is off-distribution, so that contrast fires for structural reasons."""
    out = []
    rt = tables.get("rollouts")
    if rt is not None and len(rt):
        mixed, total, per = mixed_problems(rt)
        out.append(("0. enough mixed problems", f"{mixed} of {total} problems have some but not all answers "
                    f"correct" + (f" ({per})" if per else "") + "; replay positions come from these",
                    "PASS" if mixed >= min_mixed else "CHECK"))
    if agreement and agreement["n"]:
        ok = agreement["mean_abs"] < 0.02 and agreement["p99_abs"] < 0.25
        out.append(("1. scoring matches sampling", f"mean |HF - vLLM| log-prob {agreement['mean_abs']:.4f}, "
                    f"99th pct {agreement['p99_abs']:.3f}", "PASS" if ok else "CHECK"))
    if pf is None:
        return pd.DataFrame(out, columns=["gate", "value", "verdict"])
    rnd = rule_rows(pf, "random")
    if len(rnd) > 2:
        r = rnd.q_kept.corr(rnd.problem_rate)
        out.append(("2. replay reproduces the rollout state", f"corr(Q_kept at random positions, problem's rollout "
                    f"success) = {r:.2f}", "PASS" if r > 0.5 else "CHECK"))
    nt = tables.get("noise")
    if nt is not None and not nt.empty:
        o, e = nt.iloc[0]["mean |Q_kept - Q_dup|"], nt.iloc[0]["expected from sampling alone"]
        out.append(("3. no replay noise beyond sampling", f"mean |Q_kept - Q_dup| {o:.3f} vs {e:.3f} expected",
                    "PASS" if o <= 1.25 * e + 0.01 else "CHECK"))
    opsa = rule_rows(pf, "opsa")
    if len(opsa) > 2:
        m, lo, hi = mean_ci(opsa, "damage", n_boot)
        out.append(("4. a push on OPSA's tokens moves success", f"damage D = {fmt_ci(m, lo, hi, 4)}",
                    "DETECTED" if (lo > 0 or hi < 0) else "NOT DETECTED at this size"))
        (pm, plo, phi), _ = null_shares(opsa)
        obs = opsa.useful.mean()
        out.append(("5. OPSA's useful share is outside the noise band",
                    f"share U>0 {obs:.2f} vs {pm:.2f} [{plo:.2f}, {phi:.2f}] if every token had U = 0",
                    "DETECTED" if (obs > phi or obs < plo) else "NOT DETECTED at this size"))
    return pd.DataFrame(out, columns=["gate", "value", "verdict"])


def analyze(run, n_boot=2000, write=True, show=print):
    run = Path(run)
    d = load(run)
    tables = {"rollouts": rollout_table(d["rollouts"])}
    agreement = logprob_agreement(d["rollouts"], d["scores"]) if d["scores"] else dict(n=0, mean_abs=np.nan, p99_abs=np.nan)
    if d["selected"]:
        tables["selection"], tables["categories"], tables["overlap"] = selection_tables(d["scores"], d["selected"])
    pf = None
    if d["positions"] and d["replays"]:
        kept_n = common.read_json(run / "replay.meta.json")["kept"] if (run / "replay.meta.json").exists() else 8
        pf = position_frame(d["positions"], d["replays"], d["rollouts"])
        tables.update(replay_tables(pf, kept_n, n_boot))
    tables["gates"] = gates(tables, pf, agreement, n_boot=n_boot)
    if write:
        (run / "tables").mkdir(exist_ok=True)
        for name, t in tables.items():
            t.to_csv(run / "tables" / f"{name}.csv", index=name in ("categories", "overlap"))
        if pf is not None:
            pf.drop(columns=["alt_ids", "top_alts"], errors="ignore").to_csv(run / "tables" / "positions.csv", index=False)
    if show:
        for name, t in tables.items():
            show(f"\n## {name}\n")
            show(t.to_string(index=name in ("categories", "overlap")))
        show(f"\nlog-prob agreement (vLLM sampling vs teacher-forced scoring): {agreement}")
    return tables, pf


def analyze_positions_csv(path, n_boot=2000, show=print):
    """The replay tables and gates 2 to 5 from a saved positions.csv; the tables are written next to it,
    the gates only printed (gates 0 and 1 need the run folder, so an incomplete gates.csv is not written)."""
    path = Path(path)
    pf = position_frame_from_csv(path)
    tables = replay_tables(pf, int(pf.n_kept.max()), n_boot)
    for name, t in tables.items():
        t.to_csv(path.parent / f"{name}.csv", index=False)
    g = gates(tables, pf, None, n_boot=n_boot)
    if show:
        for name, t in tables.items():
            show(f"\n## {name}\n")
            show(t.to_string(index=False))
        show("\n## gates (2 to 5; 0 and 1 need the run folder)\n")
        show(g.to_string(index=False))
        show(f"\ntables written next to {path}")
    return tables, g, pf


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", help="the run folder")
    ap.add_argument("--positions-csv", help="instead of a run folder: rebuild the replay tables from a saved "
                    "tables/positions.csv and write them next to it (gates 0 and 1 need the run folder)")
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args(argv)
    if args.positions_csv:
        analyze_positions_csv(args.positions_csv, args.boot)
    elif args.run:
        analyze(args.run, args.boot)
    else:
        ap.error("give --run or --positions-csv")


if __name__ == "__main__":
    main()
