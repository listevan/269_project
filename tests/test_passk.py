import math

import numpy as np

import passk


def _rows(problem, outcomes, text_ok="so \\boxed{12}", text_bad="so \\boxed{6}", n_tokens=100):
    rows = []
    for j, ok in enumerate(outcomes):
        rows.append(dict(key=f"{problem}#{j}", problem_id=problem, sample_idx=j, level=3, correct=ok, text=text_ok if ok else text_bad,
                         n_tokens=n_tokens, truncated=False, response_ids=list(range(n_tokens)), finish_reason="stop"))
    return rows


def test_pass_at_k_estimator():
    assert passk.pass_at_k(8, 0, 1) == 0.0 and passk.pass_at_k(8, 8, 1) == 1.0
    assert abs(passk.pass_at_k(8, 1, 1) - 1 / 8) < 1e-12
    assert passk.pass_at_k(8, 1, 8) == 1.0                       # one success among 8: drawing all 8 surely includes it
    assert abs(passk.pass_at_k(4, 2, 2) - (1 - math.comb(2, 2) / math.comb(4, 2))) < 1e-12
    assert np.isnan(passk.pass_at_k(4, 2, 8))


def test_scalability_score():
    cv = {1: 0.5, 2: 0.6, 3: 0.7, 4: 0.8}
    assert abs(passk.scalability(cv, 4) - np.mean([(0.8 - 0.5) / 0.5, (0.8 - 0.6) / 0.5, (0.8 - 0.7) / 0.5])) < 1e-12
    assert np.isnan(passk.scalability({1: 1.0, 2: 1.0}, 2))


def test_norm_answer():
    assert passk.norm_answer("\\boxed{\\dfrac{1}{2}}") == passk.norm_answer("\\boxed{ \\frac{1}{2} }")
    assert passk.norm_answer("no box") is None


def test_groups_transfer_and_dominant_wrong():
    group = passk.per_problem(_rows("p1", [True] * 4) + _rows("p2", [True, False, False, False]) + _rows("p3", [False] * 4))
    base = passk.per_problem(_rows("p1", [True] * 2) + _rows("p2", [False, False]) + _rows("p3", [False] * 2))
    trained = passk.per_problem(_rows("p1", [True] * 2) + _rows("p2", [True, True]) + _rows("p3", [True, False]))
    groups = passk.groups_of(group)
    assert groups == {"p1": "all solved", "p2": "some solved", "p3": "none solved"}
    arms = {"base A (grouping)": group, "base B (scoring)": base, "trained": trained}
    summary, dom = passk.transfer_table(arms, "base A (grouping)", "base B (scoring)", "trained")
    s = summary.iloc[0]
    assert s["apparently newly solved (audit these)"] == 1 and s["newly solved ids"] == "p3"
    assert s["lost (base solved, trained never)"] == 0
    assert set(dom.problem) == {"p2", "p3"}                      # p1 has no wrong answers
    assert dom.set_index("problem").loc["p3", "dominant_wrong_mass_base"] == 1.0
    assert dom.set_index("problem").loc["p3", "dominant_wrong_mass_trained"] == 0.5
    t = passk.arm_table(arms, groups, K=2)
    row = t[(t.arm == "trained") & (t["group (by base batch A)"] == "all problems")].iloc[0]
    assert abs(row["pass@1"] - round((1 + 1 + 0.5) / 3, 3)) < 1e-9 and row["pass@2"] == 1.0   # the table rounds to 3 decimals
    c = passk.contrast_table(arms, "base B (scoring)", groups, K=2)
    row = c[(c.arm == "trained") & (c.group == "all problems")].iloc[0]
    assert row["Δ pass@1 [95% CI]"].startswith("+0.500")
    assert passk.curve_table(arms, groups, 2).shape[0] == 3


def test_norm_answer_canonical_forms_agree():
    a = passk.norm_answer("\\boxed{\\frac{1}{2}}")
    assert a == passk.norm_answer("\\boxed{1/2}") == passk.norm_answer("\\boxed{\\dfrac12}")
    assert passk.norm_answer("\\boxed{12}") == passk.norm_answer("\\boxed{12.0}")
    assert passk.norm_answer("\\boxed{12}") != passk.norm_answer("\\boxed{13}")


def test_repetition_measures():
    assert passk.repeated_share(list(range(100))) == 0.0
    ids = list(range(20)) * 5                      # one 20-gram fills the response
    assert passk.repeated_share(ids) == 1.0 and passk.severe_repetition(ids)
    assert not passk.severe_repetition(list(range(20)) * 2)


def test_strict_reflection_only_counts_sentence_initial_forms():
    assert len(passk.REFLECT_STRICT.findall("How long must she wait? Nothing but net.")) == 0
    assert len(passk.REFLECT_STRICT.findall("So x = 2. Wait, that contradicts step 1.\nBut the sum is 5. Let me check.")) == 3


def test_budget_table_counts_only_responses_within_budget():
    rows = _rows("p1", [True, True], n_tokens=3000) + _rows("p2", [True, False], n_tokens=5000)
    arms = {"x": passk.per_problem(rows)}
    t = passk.budget_table(arms, K=2).set_index("budget (tokens)")
    assert t.loc[4096, "pass@1"] == 0.5 and t.loc[8192, "pass@1"] == 0.75   # p2's correct answer needs more than 4,096 tokens
    rows = _rows("p1", [True, True], n_tokens=4096)
    for r in rows:
        r["truncated"] = True
    assert list(passk.budget_table({"x": passk.per_problem(rows)}, K=2)["budget (tokens)"]) == [1024, 2048, 4096]   # nothing above the cap


def test_analyze_refuses_mismatched_problem_sets(tmp_path):
    import json
    for name, probs in (("a", ["p1", "p2"]), ("b", ["p1", "p2"]), ("t", ["p1", "p3"])):
        (tmp_path / name).mkdir()
        with open(tmp_path / name / "rollouts.jsonl", "w") as f:
            for q in probs:
                for r in _rows(q, [True, False]):
                    f.write(json.dumps(r) + "\n")
    import pytest
    with pytest.raises(ValueError, match="do not share one problem set"):
        passk.analyze(tmp_path / "a", tmp_path / "b", tmp_path / "t", {}, 2, show=None)
    tables = passk.analyze(tmp_path / "a", tmp_path / "b", tmp_path / "b", {}, 2, show=None)
    assert set(tables) >= {"arms", "curves", "budgets", "contrasts", "transfer", "per_problem"}
