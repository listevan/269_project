import json

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
    for name in ("rollouts", "selection", "categories", "overlap", "usefulness", "contrasts", "strata", "noise", "gates"):
        assert name in tables and len(tables[name]), name
        assert (run / "tables" / f"{name}.csv").exists()
    # replays were drawn from the same rate for kept and redrawn: usefulness should be near zero
    assert abs(pf.usefulness.mean()) < 0.15
    assert set(tables["gates"].verdict) <= {"PASS", "CHECK", "DETECTED", "NOT DETECTED at this size"}
    ex = analyze.examples(pf, common.read_jsonl(run / "scores.jsonl"), "opsa", k=2)
    assert len(ex) <= 4 and "[" in ex.iloc[0]["context"]
    cats = tables["categories"]
    assert abs(cats["all tokens"].sum() - 1.0) < 0.01 and cats.loc["end token", "all tokens"] > 0
    ov = tables["overlap"]
    assert np.isclose(ov.loc[analyze.RULES["opsa"], analyze.RULES["opsa"]], 1.0)


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
