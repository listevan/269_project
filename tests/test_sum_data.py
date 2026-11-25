import sum_data


def test_boxed_kind_and_outcome():
    assert sum_data.boxed_kind("so \\boxed{I don't know.}") == "idk"
    assert sum_data.boxed_kind("\\boxed{\\text{I don’t know}}") == "idk"
    assert sum_data.boxed_kind("\\boxed{\\text{cannot be determined}}") == "hedge"
    assert sum_data.boxed_kind("\\boxed{\\text{Not enough information}}") == "hedge"
    assert sum_data.boxed_kind("\\boxed{\\text{undefined}}") == "answer"   # a mathematical answer, not a refusal
    assert sum_data.boxed_kind("\\boxed{}") == "none" and sum_data.boxed_kind("\\boxed{\\text{}}") == "none"
    for box in ("\\text{Cannot be computed}", "\\text{The answer cannot be uniquely determined}", "\\text{Not determinable}",
                "\\text{unknown}", "?", "\\text{N/A}", "\\text{unsolvable}", "\\text{I do not know}"):
        assert sum_data.boxed_kind("\\boxed{" + box + "}") in ("idk", "hedge"), box
    for box in ("\\text{No solution}", "x = 2", "\\frac{1}{2}", "-3", "(1, 2)", "i", "\\text{undefined}"):
        assert sum_data.boxed_kind("\\boxed{" + box + "}") == "answer", box
    assert sum_data.boxed_kind("\\boxed{42} then \\fbox{I don't know.}") == "idk"      # fbox counts, the last box wins
    assert sum_data.boxed_kind("\\boxed{42}") == "answer"
    assert sum_data.boxed_kind("\\boxed{\\frac{1}{2}}") == "answer"
    assert sum_data.boxed_kind("no box") == "none"
    assert sum_data.boxed_kind("\\boxed{\\text{cannot be determined}} and then \\boxed{7}") == "answer"   # the last box counts
    assert sum_data.outcome("\\boxed{I don't know.}", "unanswerable", False) == "refusal"
    assert sum_data.outcome("\\boxed{3}", "unanswerable", False) == "false_answer"
    assert sum_data.outcome("...", "unanswerable", False) == "no_answer"
    assert sum_data.outcome("\\boxed{3}", "answerable", True) == "correct"
    assert sum_data.outcome("\\boxed{3}", "answerable", False) == "wrong"
    assert sum_data.outcome("\\boxed{I don't know.}", "answerable", False) == "refusal"


def test_wanted_behaviour_per_variant():
    assert sum_data.wanted("refusal", "unanswerable") and not sum_data.wanted("false_answer", "unanswerable")
    assert sum_data.wanted("correct", "answerable") and not sum_data.wanted("refusal", "answerable")


def test_question_split_strips_the_dataset_instruction():
    prompt = "What is 2+2? " + sum_data.SUM_INSTRUCTION
    assert sum_data._SPLIT.sub("", prompt).strip() == "What is 2+2?"
