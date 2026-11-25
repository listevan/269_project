import json

import numpy as np

import rules_analysis as ra
import rules_data as rd


def _rows(sym, split, item_ids, answers_by_item, n_tokens=50):
    rows = []
    for iid, answers in zip(item_ids, answers_by_item):
        it = dict(answer=18, habit=11)
        for j, got in enumerate(answers):
            rows.append(dict(item_id=iid, sym=sym, split=split, kind="direct", operands=[7, 4], answer=it["answer"], habit=it["habit"],
                             sample_idx=j, got=got, correct=got == 18, n_tokens=n_tokens, truncated=False, repeated=0.0, severe_rep=False,
                             text=f"\\boxed{{{got}}}" if got is not None else "no box"))
    return rows


def _write(tmp, name, rows):
    d = tmp / name
    d.mkdir()
    with open(d / "rollouts.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return d


def test_dominant_and_classify():
    rows = _rows("#", "new_numbers", ["#:new_numbers:0"], [[11, 11, 18, 12]])
    dom = ra.dominant_map(rows)
    assert dom == {"#:new_numbers:0": 11}
    ra.classify(rows, dom)
    assert [r["outcome"] for r in rows] == ["dominant_wrong", "dominant_wrong", "correct", "other_wrong"]
    sh = ra.item_shares(rows)
    d = sh[("#", "new_numbers", "#:new_numbers:0")]
    assert d["correct"] == 0.25 and d["dominant_wrong"] == 0.5 and d["other_wrong"] == 0.25 and d["habit"] == 0.5


def test_base_a_is_scored_with_base_b_map(tmp_path):
    a = _rows("#", "new_numbers", ["#:new_numbers:0"], [[11, 11, 12, 18]])      # A's most common wrong: 11
    b = _rows("#", "new_numbers", ["#:new_numbers:0"], [[12, 12, 11, 18]])      # B's most common wrong: 12
    for sp in rd.SPLITS[1:]:
        a += _rows("#", sp, [f"#:{sp}:0"], [[11, 18]])
        b += _rows("#", sp, [f"#:{sp}:0"], [[12, 18]])
    runs = {"base_A": _write(tmp_path, "base_A", a), "base_B": _write(tmp_path, "base_B", b)}
    tables = ra.analyze(runs, None, tmp_path / "tables", show=lambda *a: None)
    by = tables["by_split"].set_index(["split", "arm"])
    assert by.loc[("new_numbers", "base_A"), "dominant_wrong"] == 0.25      # A's 12 counted (B's map), not its two 11s
    assert by.loc[("new_numbers", "base_B"), "dominant_wrong"] == 0.25      # B's 11 counted (A's map)


def test_boot_delta():
    a = np.array([[1.0, 1.0], [0.5, 0.5]])
    b = np.array([[0.0, 0.0], [0.5, 0.5]])
    m, lo, hi = ra.boot_delta(a, b)
    assert m == 0.5 and lo == 0.5 and hi == 0.5
    m, lo, hi = ra.boot_delta(np.array([[0.0, 1.0]]), np.array([[0.0, 0.0]]), n_boot=500)
    assert m == 0.5 and lo == 0.0 and hi == 1.0


def test_analyze_end_to_end(tmp_path):
    syms = ["#", "$"]
    ids = {s: {sp: [f"{s}:{sp}:{i}" for i in range(2)] for sp in rd.SPLITS} for s in syms}
    def batch(correct_prob, seed):
        rng = np.random.default_rng(seed)
        rows = []
        for s in syms:
            for sp in rd.SPLITS:
                answers = [[18 if rng.random() < correct_prob[s] else 11 for _ in range(4)] for _ in ids[s][sp]]
                rows += _rows(s, sp, ids[s][sp], answers)
        return rows
    runs = {"base_A": _write(tmp_path, "base_A", batch({"#": 0.0, "$": 0.5}, 1)),
            "base_B": _write(tmp_path, "base_B", batch({"#": 0.0, "$": 0.5}, 2)),
            "opsa": _write(tmp_path, "opsa", batch({"#": 0.0, "$": 1.0}, 3)),
            "teacher_neg": _write(tmp_path, "teacher_neg", batch({"#": 1.0, "$": 1.0}, 4))}
    read = dict(overall=dict(n=8, p_correct=0.4, p_correct_median=0.3, p_dominant=0.2), per_sym={"#": dict(p_correct=0.1), "$": dict(p_correct=0.7)})
    (runs["opsa"] / "probe.json").write_text(json.dumps(dict(n_probed=8, n_rows=8, own=read, fixed=read)))
    tdir = tmp_path / "train_opsa"
    tdir.mkdir()
    with open(tdir / "train_log.jsonl", "w") as f:
        for step in (1, 2):
            f.write(json.dumps(dict(step=step, n_rollouts=4, truncated=0.0, mean_tokens=40, correct=0.1 * step, habit=0.5, no_answer=0.0,
                                    mean_entropy=1.0, n_tokens=100, n_updated=20, mean_abs_adv=0.7, loss=0.1, grad_norm=2.0,
                                    update_norm=0.01, param_norm=5.0, gen_s=1, score_s=1, train_s=1, seconds=3)) + "\n")
    shown = []
    tables = ra.analyze(runs, None, tmp_path / "tables", show=shown.append, train_dirs={"opsa": tdir})
    ops = tables["operators"].set_index("sym")
    assert ops.loc["#", "base correct"] == 0.0 and ops.loc["#", "definition"] == "2a + b"
    main = tables["correct_new_numbers"].set_index("sym")
    assert main.loc["#", "teacher_neg"] == 1.0 and main.loc["#", "opsa"] == 0.0 and main.loc["mean", "teacher_neg"] == 1.0
    by = tables["by_split"]
    row = by[(by["arm"] == "teacher_neg") & (by["split"] == "new_numbers")].iloc[0]
    assert row["correct"] == 1.0 and row["d_correct"].startswith("+")
    assert "rare (<5%)" in set(tables["reading"]["rarity"])
    assert (tmp_path / "tables" / "dominant.json").exists() and (tmp_path / "tables" / "dominant_B.json").exists()
    assert "probe" in tables and len(tables["probe"]) == 2 and "training" in tables
    assert tables["training"].iloc[0]["rollout correct first/last"] == "0.100 / 0.200"
    assert (tmp_path / "tables" / "train_curves.csv").exists()
