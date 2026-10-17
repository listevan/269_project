import replay

END = [99]


def _pos(t, alts):
    return dict(pos_id=f"p#0@{t}", key="p#0", problem_id="p", t=t, alt_ids=alts)


def test_kept_and_redrawn_prefixes():
    resp = [10, 11, 12, 13, 99]
    reqs = replay.build_requests(_pos(2, [20, 21]), resp, END, kept=4, max_tokens=10, dup_fraction=0.0, seed=0)
    kept = [q for q in reqs if q["kind"] == "kept"]
    red = [q for q in reqs if q["kind"] == "redrawn"]
    assert len(kept) == 1 and kept[0]["n"] == 4 and kept[0]["resp_prefix"] == [10, 11, 12]
    assert [q["resp_prefix"] for q in red] == [[10, 11, 20], [10, 11, 21]]
    assert all(q["n"] == 1 for q in red)
    assert all(q["max_tokens"] == 10 - 3 for q in reqs)   # same total budget as the rollout
    assert not any(q["fixed"] for q in reqs)


def test_end_token_is_fixed():
    resp = [10, 11, 99]
    reqs = replay.build_requests(_pos(2, [99, 30]), resp, END, kept=4, max_tokens=10, dup_fraction=0.0, seed=0)
    by = {(q["kind"], q["j"]): q for q in reqs}
    assert by[("kept", 0)]["fixed"]            # the response ended here
    assert by[("redrawn", 0)]["fixed"]         # a redrawn end token ends it too
    assert not by[("redrawn", 1)]["fixed"]


def test_budget_exhausted_is_fixed():
    resp = list(range(10))
    reqs = replay.build_requests(_pos(9, [50]), resp, END, kept=2, max_tokens=10, dup_fraction=0.0, seed=0)
    assert all(q["fixed"] and q["max_tokens"] == 0 for q in reqs)


def test_dup_pass_and_distinct_seeds():
    reqs = replay.build_requests(_pos(1, [5, 6, 7]), [1, 2, 3], END, kept=3, max_tokens=10, dup_fraction=1.0, seed=0)
    kinds = [q["kind"] for q in reqs]
    assert kinds.count("dup") == 1
    seeds = [q["seed"] for q in reqs]
    assert len(set(seeds)) == len(seeds)
    dup = next(q for q in reqs if q["kind"] == "dup")
    kept = next(q for q in reqs if q["kind"] == "kept")
    assert dup["resp_prefix"] == kept["resp_prefix"] and dup["n"] == kept["n"]


def test_dup_fraction_roughly_respected():
    share = sum(replay.is_dup(f"x@{i}", 0.25) for i in range(4000)) / 4000
    assert 0.22 < share < 0.28
