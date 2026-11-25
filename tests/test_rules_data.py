import rules_data as rd


def test_candidates_compute():
    for op in rd.CANDIDATES:
        assert op["fn"](7, 4) == eval_expr(op, 7, 4), op["sym"]
        assert isinstance(op["show"](7, 4), str)
        assert op["habit_fn"](7, 4) is not None


def eval_expr(op, a, b):
    """Check fn against the printed definition for the simple forms."""
    e = op["expr"]
    table = {"2a + b": 2 * a + b, "a + 2b": a + 2 * b, "ab + a": a * b + a, "ab + b": a * b + b, "|a - b|": abs(a - b),
             "a + b - 1": a + b - 1, "2(a + b)": 2 * (a + b), "3a - b": 3 * a - b, "a + 3b": a + 3 * b, "a + b": a + b,
             "ab": a * b, "ab + 1": a * b + 1, "a - b": a - b, "min(a, b)": min(a, b), "max(a, b)": max(a, b),
             "ab - a - b": a * b - a - b, "a^2 + b": a * a + b, "a + b + ab": a + b + a * b, "ab - b": a * b - b,
             "a^2 - b": a * a - b, "2a - b": 2 * a - b}
    return table[e]


def test_items_disjoint_and_deterministic():
    a = rd.make_items("#", 96, 24)
    b = rd.make_items("#", 96, 24)
    assert a == b
    train = {tuple(it["operands"]) for it in a["train"]}
    new = {tuple(it["operands"]) for it in a["new_numbers"]}
    assert len(train) == 96 and len(new) == 24 and not (train & new)
    assert all(rd.TRAIN_RANGE[0] <= x <= rd.TRAIN_RANGE[1] for it in a["train"] for x in it["operands"])
    assert all(rd.NEW_RANGE[0] <= x <= rd.NEW_RANGE[1] for it in a["new_range"] for x in it["operands"])
    assert all(len(it["operands"]) == 3 for it in a["composition"])
    assert [it["operands"] for it in a["with_def"]] == [it["operands"] for it in a["new_numbers"]]
    assert a["with_def"][0]["item_id"].startswith("#:with_def:")
    it = a["train"][0]
    x, y = it["operands"]
    assert it["answer"] == 2 * x + y and it["habit"] == x + y
    c = a["composition"][0]
    x, y, z = c["operands"]
    assert c["answer"] == 2 * x + (2 * y + z)
    assert rd.make_items("#", 0, 4)["train"] == []


def test_pow_habit_guarded():
    items = rd.make_items("↑", 8, 8)
    assert all(it["habit"] is None or abs(it["habit"]) <= 10 ** 15 for it in items["composition"])
    assert all(isinstance(it["answer"], int) for it in items["composition"])


def test_prompts():
    op = rd.BY_SYM["#"]
    it = rd.make_items("#", 4, 4)["train"][0]
    a, b = it["operands"]
    assert rd.prompt_body(op, it) == f"Compute {a} # {b}."
    assert rd.prompt_body(op, it, with_def=True) == f"The operation # is defined by a # b = 2a + b. Compute {a} # {b}."
    comp = rd.make_items("#", 4, 4)["composition"][0]
    x, y, z = comp["operands"]
    assert rd.prompt_body(op, comp) == f"Compute {x} # ({y} # {z})."
    demo = rd.demonstration(op, it)
    assert demo.endswith(f"\\boxed{{{it['answer']}}}") and f"2*{a} + {b}" in demo and "2a + b" in demo


def test_parse_int():
    assert rd.parse_int("18") == 18 and rd.parse_int(" -3 ") == -3 and rd.parse_int("1,234") == 1234
    assert rd.parse_int("\\text{18}") == 18 and rd.parse_int("18.0") == 18 and rd.parse_int("$42$") == 42
    assert rd.parse_int("7 # 4 = 18") == 18 and rd.parse_int("2 \\cdot 7 + 4 = 18") == 18
    assert rd.parse_int("18\\,") == 18 and rd.parse_int("1\\,234") == 1234 and rd.parse_int("18.") == 18
    assert rd.parse_int("18 \\text{ or } 11") is None and rd.parse_int("x") is None and rd.parse_int(None) is None
    assert rd.parse_int("\\frac{1}{2}") is None and rd.parse_int("") is None
    assert rd.answer_of("so the answer is \\boxed{11}. Then \\boxed{18}") == 18
    assert rd.answer_of("no box") is None


def test_outcome_and_dominant():
    item = dict(answer=18)
    assert rd.outcome(18, item, 11) == "correct"
    assert rd.outcome(11, item, 11) == "dominant_wrong"
    assert rd.outcome(12, item, 11) == "other_wrong"
    assert rd.outcome(None, item, 11) == "no_answer"
    assert rd.outcome(11, item, None) == "other_wrong"
    assert rd.dominant_wrong_of([11, 11, 18, 12, None], 18) == 11
    assert rd.dominant_wrong_of([11, 12], 18) == 11          # tie: the smaller
    assert rd.dominant_wrong_of([18, None], 18) is None


def test_pick_operators():
    rates = {"a": 0.0, "b": 0.01, "c": 0.02, "d": 0.1, "e": 0.3, "f": 0.5, "g": 0.9, "h": 0.2}
    picked = rd.pick_operators(rates, n=4, max_rate=0.55, n_zero=1)
    assert "g" not in picked and "a" in picked and "b" in picked and "f" in picked and len(picked) == 4
    assert rd.pick_operators({"a": 0.0, "b": 0.7}, n=4) == ["a"]
    assert rd.pick_operators(rates, n=20) == sorted((s for s, r in rates.items() if r <= 0.55), key=lambda s: (rates[s], s))
    many_zero = {f"z{i}": 0.0 for i in range(10)} | {"p": 0.05, "q": 0.1, "r": 0.2, "s": 0.4, "t": 0.6}
    picked = rd.pick_operators(many_zero, n=12, max_rate=0.55, n_zero=3)
    assert sum(p.startswith("z") for p in picked) == 3 and {"p", "q", "r", "s"} <= set(picked) and "t" not in picked
    only_zero = {f"z{i}": 0.0 for i in range(5)}
    assert len(rd.pick_operators(only_zero, n=12)) == 3          # at most n_zero operators at exactly zero


def test_prompt_text_uses_the_two_instructions():
    class Tok:
        def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False):
            assert enable_thinking is False and len(msgs) == 1 and msgs[0]["role"] == "user"
            return "<user>" + msgs[0]["content"] + "<assistant>"

        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": [ord(c) for c in text]}

    it = rd.make_items("#", 4, 4)["train"][0]
    a, b = it["operands"]
    plain = rd.prompt_text_for(Tok(), it)
    assert plain == f"<user>Compute {a} # {b}.\n{rd.RULE_INSTRUCTION}<assistant>"
    with_def = rd.prompt_text_for(Tok(), it, with_def=True)
    assert with_def.startswith(f"<user>The operation # is defined by a # b = 2a + b. Compute {a} # {b}.\n{rd.DEF_INSTRUCTION}")
    assert rd.prompt_ids_for(Tok(), it) == [ord(c) for c in plain]
