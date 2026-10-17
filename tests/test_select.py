import math

import numpy as np
import pytest

import select_rules as sr


def test_lowest_fraction_matches_opsa_definition():
    # OPSA: concatenate the pool, K = max(1, floor(f N)), lowest K log-probs, stable order on ties
    vals = {"a": [-0.1, -5.0, -0.2], "b": [-3.0, -0.01, -5.0, -0.3]}
    out = sr.lowest_fraction(vals, ["a", "b"], 0.3)   # N = 7, K = 2
    assert sum(len(v) for v in out.values()) == math.floor(0.3 * 7)
    assert out == {"a": [1], "b": [2]}               # the two -5.0, "a" first (stable)


def test_budget_at_least_one_and_at_most_n():
    assert sr.budget(3, 0.2) == 1
    assert sr.budget(0, 0.2) == 0
    assert sr.budget(10, 1.0) == 10


def test_teacher_neg_keeps_only_negative_gaps():
    gap = {"a": [0.5, -1.0, 0.2, 0.1, 0.3]}
    out = sr.lowest_fraction(gap, ["a"], 0.4, only_negative=True)   # K = 2: -1.0 and 0.1, only -1.0 is negative
    assert out == {"a": [1]}


def test_random_fraction_same_budget_and_seeded():
    lengths = {"a": 50, "b": 30}
    r1 = sr.random_fraction(lengths, ["a", "b"], 0.2, seed=3)
    r2 = sr.random_fraction(lengths, ["a", "b"], 0.2, seed=3)
    assert r1 == r2
    assert sum(len(v) for v in r1.values()) == 16
    assert all(0 <= t < lengths[k] for k, ts in r1.items() for t in ts)


def test_opsa_advantages_entropy_ranked():
    a = sr.opsa_advantages([0.0, 1.0, 2.0])
    assert np.allclose(a, [-0.5, -0.75, -1.0])
    assert np.allclose(sr.opsa_advantages([0.3, 0.3]), [-1.0, -1.0])


def _rows():
    rng = np.random.default_rng(0)
    rows = []
    for i in range(20):
        n = int(rng.integers(5, 40))
        lp_s = (-rng.exponential(1.0, n)).tolist()
        rows.append(dict(key=f"p{i % 7}#{i}", correct=bool(i % 3), lp_s=lp_s,
                         lp_t=(np.asarray(lp_s) + rng.normal(0, 1, n)).tolist(), ent_s=rng.uniform(0, 3, n).tolist()))
    return rows


def test_selected_sets_shapes():
    rows = _rows()
    sets, adv, pools = sr.selected_sets(rows, pool_size=8, fraction=0.2, seed=0)
    assert sorted(k for p in pools for k in p) == sorted(r["key"] for r in rows)
    n_by_pool = [sum(len(next(r for r in rows if r["key"] == k)["lp_s"]) for k in p) for p in pools]
    assert sum(len(v) for v in sets["opsa"].values()) == sum(sr.budget(n, 0.2) for n in n_by_pool)
    assert sum(len(v) for v in sets["random"].values()) == sum(sr.budget(n, 0.2) for n in n_by_pool)
    assert sum(len(v) for v in sets["teacher_neg"].values()) <= sum(sr.budget(n, 0.2) for n in n_by_pool)
    for r in rows:   # nsr: every token of a failed response, nothing of a correct one
        assert sets["nsr"][r["key"]] == ([] if r["correct"] else list(range(len(r["lp_s"]))))
        assert set(adv.get(r["key"], {})) == set(sets["opsa"][r["key"]])
        assert all(-1.0 <= a <= -0.5 for a in adv.get(r["key"], {}).values())


def test_sample_positions_dedupes_and_is_seeded():
    sets = {"opsa": {"a": [1, 2], "b": [0]}, "random": {"a": [2], "b": [3]}}
    p1 = sr.sample_positions(sets, 10, seed=1)
    assert p1 == sr.sample_positions(sets, 10, seed=1)
    ids = [p["pos_id"] for p in p1]
    assert len(ids) == len(set(ids)) == 4
    shared = next(p for p in p1 if p["pos_id"] == "a@2")
    assert shared["methods"] == ["opsa", "random"]
    assert sr.in_sets(sets, "a", 2) == ["opsa", "random"]


def test_sample_positions_caps_at_set_size():
    sets = {"opsa": {"a": [0, 1, 2]}}
    assert len(sr.sample_positions(sets, 2, 0)) == 2
    assert len(sr.sample_positions(sets, 99, 0)) == 3


if __name__ == "__main__":
    pytest.main([__file__])
