from common import last_boxed
from grade import Grader, grade_one


def test_last_boxed_nested_and_last():
    assert last_boxed(r"so \boxed{1} then \boxed{\frac{1}{2}}") == r"\frac{1}{2}"
    assert last_boxed(r"\boxed{a{b}c}") == "a{b}c"
    assert last_boxed("no box") is None
    assert last_boxed(r"\boxed{unclosed") is None


def test_grade_one_boxed_only():
    assert grade_one(r"The answer is \boxed{\frac{1}{2}}", r"\frac12") or grade_one(r"\boxed{\frac{1}{2}}", r"\frac{1}{2}")
    assert not grade_one("The answer is 1/2", r"\frac{1}{2}")      # unboxed answers get no credit
    assert grade_one(r"\boxed{5}", "5")
    assert not grade_one(r"\boxed{6}", "5")


def test_grader_serial_and_pool():
    pairs = [(r"\boxed{3}", "3"), (r"\boxed{4}", "3")]
    with Grader(workers=0) as g:
        assert g.grade_many(pairs) == [True, False]
    with Grader(workers=2) as g:
        assert g.grade_many(pairs) == [True, False]
