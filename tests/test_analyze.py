import json

import math

import numpy as np
import pandas as pd

import analyze
import common
import select_rules as sr


def _fake_run(tmp_path, seed=0):
    """A small run folder with every stage's output, the replays drawn from known success rates."""
    rng = np.random.default_rng(seed)
    rollouts, scores = [], []
    for p in range(12):
        rate = [0.0, 0.5, 1.0][p % 3]
        for s in range(4):
            n = int(rng.integers(8, 30))
            ids = rng.integers(0, 50, n).tolist() + [99]
            lp = (-rng.exponential(1.0, n + 1)).round(4).tolist()
            key = common.response_key(f"prob{p}", s)
            rollouts.append(dict(key=key, problem_id=f"prob{p}", sample_idx=s, level=3 + p % 3, gold="1",
                                 response_ids=ids, n_tokens=n + 1, finish_reason="stop", truncated=False,
                                 text="x", vllm_lp=[v + 0.001 for v in lp], correct=bool(rng.random() < rate)))
            scores.append(dict(key=key, problem_id=f"prob{p}", correct=rollouts[-1]["correct"], lp_s=lp,
                               ent_s=rng.uniform(0, 2, n + 1).round(3).tolist(),
                               lp_t=(np.asarray(lp) + rng.normal(0, 1, n + 1)).round(4).tolist(),
                               tokens=rng.choice(["Wait", " 3", "\\frac", " the", "\n", ","], n).tolist() + ["<|im_end|>"]))
    sets, adv, _ = sr.selected_sets(scores, 8, 0.2, 0)
    selected = [dict(key=r["key"], problem_id=r["problem_id"], correct=r["correct"], n=len(r["lp_s"]),
                     opsa=sets["opsa"][r["key"]], opsa_adv=[adv[r["key"]][t] for t in sets["opsa"][r["key"]]],
                     random=sets["random"][r["key"]], teacher_neg=sets["teacher_neg"][r["key"]]) for r in scores]
    picked = sr.sample_positions(sets, 10, 0)
    by = {r["key"]: r for r in rollouts}
    positions, replays = [], []
    for p in picked:
        r = by[p["key"]]
        pk = float(np.exp(next(s for s in scores if s["key"] == p["key"])["lp_s"][p["t"]]))
        positions.append(dict(p, problem_id=r["problem_id"], token_id=r["response_ids"][p["t"]], token_str=" x",
                              is_end=False, n_tokens=r["n_tokens"], response_correct=r["correct"], lp_s=np.log(pk),
                              lp_t=0.0, ent_s=1.0, opsa_adv=None, in_sets=p["methods"], p_kept=pk, lp_kept=np.log(pk),
                              entropy=1.0, alt_ids=[1, 2, 3, 4], alt_strs=["a"] * 4, top_alts=[]))
        rate = [0.0, 0.5, 1.0][int(r["problem_id"][4:]) % 3]
        for k in range(8):
            replays.append(dict(pos_id=p["pos_id"], kind="kept", j=k, token_id=0, n_cont=5, finish_reason="stop",
                                truncated=False, cont_text="", correct=bool(rng.random() < rate)))
            replays.append(dict(pos_id=p["pos_id"], kind="dup", j=k, token_id=0, n_cont=5, finish_reason="stop",
                                truncated=False, cont_text="", correct=bool(rng.random() < rate)))
        for j in range(4):
            replays.append(dict(pos_id=p["pos_id"], kind="redrawn", j=j, token_id=1, n_cont=5, finish_reason="stop",
                                truncated=False, cont_text="", correct=bool(rng.random() < rate)))
    for name, rows in (("rollouts", rollouts), ("scores", scores), ("selected", selected),
                       ("positions", positions), ("replays", replays)):
        common.write_jsonl_atomic(tmp_path / f"{name}.jsonl", rows)
    common.write_json(tmp_path / "replay.meta.json", dict(kept=8))
    return tmp_path


def test_position_frame_usefulness(tmp_path):
    positions = [dict(pos_id="a#0@1", key="a#0", problem_id="a", t=1, n_tokens=4, p_kept=0.5,
                      response_correct=True, methods=["opsa"])]
    rollouts = [dict(problem_id="a", correct=True), dict(problem_id="a", correct=False)]
    reps = ([dict(pos_id="a#0@1", kind="kept", correct=c, truncated=False) for c in (1, 1, 1, 0)] +
            [dict(pos_id="a#0@1", kind="redrawn", correct=c, truncated=False) for c in (1, 0, 0, 0)])
    pf = analyze.position_frame(positions, reps, rollouts)
    row = pf.iloc[0]
    assert row.q_kept == 0.75 and row.q_redrawn == 0.25
    assert row.usefulness == 0.5
    assert np.isclose(row.damage, 0.5 * 0.5 * 0.5)
    assert row.stratum == "some correct" and row.n_kept == 4 and row.n_dup == 0


def test_mean_ci_brackets_mean():
    df = pd.DataFrame(dict(problem_id=list("aabbcc"), v=[0.0, 1.0, 0.5, 0.5, 1.0, 1.0]))
    m, lo, hi = analyze.mean_ci(df, "v", n_boot=500)
    assert np.isclose(m, 4 / 6) and lo <= m <= hi


def test_diff_ci_zero_for_same_rows():
    df = pd.DataFrame(dict(problem_id=list("aabbcc"), v=[0.0, 1.0, 0.5, 0.5, 1.0, 1.0]))
    d, lo, hi, p = analyze.diff_ci(df, df, "v", n_boot=200)
    assert d == 0 and lo == 0 and hi == 0


def test_analyze_end_to_end(tmp_path):
    run = _fake_run(tmp_path)
    tables, pf = analyze.analyze(run, n_boot=200, show=None)
    for name in ("rollouts", "selection", "categories", "overlap", "usefulness", "contrasts", "strata", "noise",
                 "share_null", "bands", "gates"):
        assert name in tables and len(tables[name]), name
        assert (run / "tables" / f"{name}.csv").exists()
    # replays were drawn from the same rate for kept and redrawn: usefulness should be near zero
    assert abs(pf.usefulness.mean()) < 0.15
    assert set(tables["gates"].verdict) <= {"PASS", "CHECK", "DETECTED", "NOT DETECTED at this size"}
    assert [g[:2] for g in tables["gates"].gate] == ["0.", "1.", "2.", "3.", "4.", "5."]
    # kept and redrawn were drawn from the same rate, so the useful share should sit inside the noise band
    sn = tables["share_null"].set_index("rule").loc[analyze.RULES["opsa"]]
    lo, hi = (float(v) for v in sn["null share U>0 [95%]"].split("[")[1].rstrip("]").split(","))
    assert lo <= sn["share U>0"] <= hi
    assert sn["sd of true U (est.)"] < sn["sampling sd"]
    ex = analyze.examples(pf, common.read_jsonl(run / "scores.jsonl"), "opsa", k=2)
    assert len(ex) <= 4 and "[" in ex.iloc[0]["context"]
    cats = tables["categories"]
    assert abs(cats["all tokens"].sum() - 1.0) < 0.01 and cats.loc["end token", "all tokens"] > 0
    ov = tables["overlap"]
    assert np.isclose(ov.loc[analyze.RULES["opsa"], analyze.RULES["opsa"]], 1.0)


def _pf(rows):
    """A position frame from (rule, p_kept, q_kept, q_redrawn, correct) tuples, 8 + 8 continuations each."""
    pf = pd.DataFrame([dict(pos_id=f"k{i}@1", key=f"k{i}", problem_id=f"prob{i % 4}", t=1, n_tokens=10, methods=[m],
                            p_kept=p, response_correct=c, q_kept=qk, q_redrawn=qr, n_kept=8, n_redrawn=8, q_dup=np.nan,
                            n_dup=0, cont_truncated=0.0) for i, (m, p, qk, qr, c) in enumerate(rows)])
    pf["usefulness"] = pf.q_kept - pf.q_redrawn
    pf["usefulness_dup"] = np.nan
    pf["damage"] = pf.p_kept * (1 - pf.p_kept) * pf.usefulness
    pf["useful"] = (pf.usefulness > 0).astype(float)
    pf["harmful"] = (pf.usefulness < 0).astype(float)
    pf["rel_t"] = 0.1
    pf["problem_rate"] = 0.5
    pf["stratum"] = "some correct"
    return pf


def test_null_shares_deal_the_observed_successes_out_at_random():
    # 4 of 8 kept and 4 of 8 redrawn at every position: the 8 successes land k in kept with k ~ Hypergeom(16, 8, 8),
    # P(k = 4) = C(8,4)^2 / C(16,8) = 0.381, so P(U > 0) = (1 - 0.381) / 2 = 0.310
    g = _pf([("opsa", 0.5, 0.5, 0.5, True)] * 40)
    (pm, lo, hi), (nm, _, _) = analyze.null_shares(g, n_boot=4000)
    assert abs(pm - 0.310) < 0.01 and abs(nm - 0.310) < 0.01 and lo < pm < hi
    # pinned at 0 or 1 there is nothing to deal out: the null share is 0
    g = _pf([("opsa", 0.5, 1.0, 1.0, True)] * 10 + [("opsa", 0.5, 0.0, 0.0, False)] * 10)
    (pm, lo, hi), _ = analyze.null_shares(g)
    assert pm == lo == hi == 0.0


def test_null_shares_stay_calibrated_at_boundary_rates():
    # equal true rates of 0.05 in both branches, U = 0 everywhere: the null share should match the analytic
    # P(X > Y) for X, Y ~ Binomial(8, 0.05), and gate 5's band should fire about one time in twenty, not one in three
    # (fresh binomial draws at each position's estimated pooled rate give a null mean of 0.165 here)
    pmf = np.array([math.comb(8, k) * 0.05 ** k * 0.95 ** (8 - k) for k in range(9)])
    truth = (1 - (pmf ** 2).sum()) / 2
    means, fires = [], 0
    for seed in range(60):
        rng = np.random.default_rng(seed)
        q = rng.binomial(8, 0.05, (64, 2)) / 8
        g = _pf([("opsa", 0.5, qk, qr, False) for qk, qr in q])
        (pm, lo, hi), _ = analyze.null_shares(g, n_boot=1000, seed=seed)
        means.append(pm)
        fires += g.useful.mean() > hi or g.useful.mean() < lo
    assert abs(np.mean(means) - truth) < 0.02, (np.mean(means), truth)
    assert fires <= 9, fires


def _band(cell):
    return tuple(float(v) for v in cell.split("[")[1].rstrip("]").split(","))


def test_share_null_table_separates_signal_from_noise():
    # the same U = 0.75 at every token (7 of 8 kept vs 1 of 8 redrawn): the useful share, 1.0, is far above the
    # null band (two Binomial(8, .5) draws give about 0.40), the sampling sd is sqrt(2 * .875 * .125 / 7) = 0.177,
    # and nothing is left over for differences between tokens
    g = _pf([("opsa", 0.3, 0.875, 0.125, True)] * 24)
    t = analyze.share_null_table(g).iloc[0]
    assert t["share U>0"] == 1.0 and t["share U>0"] > _band(t["null share U>0 [95%]"])[1]
    assert abs(t["sampling sd"] - 0.177) < 0.005 and t["sd of true U (est.)"] == 0
    # half the tokens at U = +0.75, half at -0.75: the spread is real, sd of true U near 0.75
    g = _pf([("opsa", 0.3, 0.875, 0.125, True)] * 12 + [("opsa", 0.3, 0.125, 0.875, False)] * 12)
    t = analyze.share_null_table(g).iloc[0]
    assert t["share U>0"] == 0.5 and 0.7 < t["sd of true U (est.)"] < 0.8


def test_band_table_partitions_positions():
    g = _pf([("opsa", 0.1, 0.5, 0.25, True), ("opsa", 0.45, 0.5, 0.5, False), ("opsa", 0.7, 0.0, 0.5, False),
             ("random", 0.995, 1.0, 0.5, True), ("random", 1.0, 1.0, 0.0, True), ("random", 0.85, 0.5, 0.5, False)])
    t = analyze.band_table(g)
    assert t.groupby("rule").positions.sum().to_dict() == {analyze.RULES["opsa"]: 3, analyze.RULES["random"]: 3}
    r = t[(t.rule == analyze.RULES["random"]) & (t["p(token)"] == "0.99 and up")].iloc[0]
    assert r.positions == 2 and r.usefulness == 0.75 and r["share U>0"] == 1.0


def test_signal_gates_read_opsa_alone():
    rng = np.random.default_rng(0)
    rnd = [("random", 1.0, qk, 0.5, True) for qk in rng.binomial(8, 0.6, 8) / 8]
    # kept always succeeds and redrawn always fails at p = 0.5: D = 0.25 everywhere, so both signal gates detect
    t = analyze.gates({"noise": pd.DataFrame()}, _pf([("opsa", 0.5, 1.0, 0.0, True)] * 16 + rnd), None, n_boot=200)
    by = t.set_index("gate").verdict
    assert by.filter(like="4.").iloc[0] == "DETECTED" and by.filter(like="5.").iloc[0] == "DETECTED"
    # kept and redrawn drawn from the same rate: U scatters around 0, neither gate fires
    q = rng.binomial(8, 0.5, (32, 2)) / 8
    g = _pf([("opsa", 0.5, qk, qr, True) for qk, qr in q] + rnd)
    t = analyze.gates({"noise": pd.DataFrame()}, g, None, n_boot=200)
    assert set(t[t.gate.str[:2].isin(["4.", "5."])].verdict) == {"NOT DETECTED at this size"}
    assert not t.gate.str.contains("random").any()


def test_positions_csv_round_trip(tmp_path):
    run = _fake_run(tmp_path)
    tables, pf = analyze.analyze(run, n_boot=100, show=None)
    out = tmp_path / "copy"
    out.mkdir()
    (out / "positions.csv").write_bytes((run / "tables" / "positions.csv").read_bytes())
    t2, g2, pf2 = analyze.analyze_positions_csv(out / "positions.csv", n_boot=100, show=None)
    assert len(pf2) == len(pf) and list(pf2.methods.iloc[0]) == list(pf.methods.iloc[0])
    pd.testing.assert_frame_equal(t2["usefulness"], tables["usefulness"])
    pd.testing.assert_frame_equal(t2["bands"], tables["bands"])
    assert (out / "share_null.csv").exists() and not (out / "gates.csv").exists()
    assert [g[:2] for g in g2.gate] == ["2.", "3.", "4.", "5."]


def test_gate0_from_rollouts_alone(tmp_path):
    run = _fake_run(tmp_path)
    for name in ("scores", "selected", "positions", "replays"):
        (run / f"{name}.jsonl").unlink()
    tables, pf = analyze.analyze(run, show=None, write=False)
    assert pf is None and list(tables["gates"].gate) == ["0. enough mixed problems"]
    assert tables["gates"].iloc[0].verdict == "CHECK"   # 12 fake problems, at most 4 mixed
    rt = pd.DataFrame([dict(level=3, problems=13, all_correct=6, some_correct=7, none_correct=0),
                       dict(level="all", problems=60, all_correct=10, some_correct=45, none_correct=5)])
    g = analyze.gates({"rollouts": rt})
    assert g.iloc[0].verdict == "PASS" and "level 3: 7/13" in g.iloc[0].value
    assert analyze.gates({"rollouts": rt.assign(some_correct=[7, 19])}).iloc[0].verdict == "CHECK"


def test_partial_replays_are_read(tmp_path):
    run = _fake_run(tmp_path)
    rows = common.read_jsonl(run / "replays.jsonl")
    (run / "replays.jsonl").unlink()
    common.write_jsonl_atomic(run / "replays.jsonl.partial", rows[: len(rows) // 2])
    tables, pf = analyze.analyze(run, n_boot=100, show=None, write=False)
    assert 0 < len(pf) < len(common.read_jsonl(run / "positions.jsonl"))
