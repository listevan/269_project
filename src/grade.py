"""Grade responses: math-verify on the last complete \\boxed{...} only (the prompt asks for a boxed answer).

Grading runs in spawned worker processes with a per-item timeout, so one pathological answer cannot stall a
stage (this happened in the 239 project). Falls back to a normalized string match when math-verify is
missing, which only the laptop tests should ever hit.
"""
import multiprocessing as mp
import os
import re

from common import last_boxed

_GRADER = None


def make_grader():
    """(grade(boxed_content, gold) -> bool, name)."""
    try:
        from math_verify import parse, verify  # type: ignore

        def grade(boxed: str, gold: str) -> bool:
            try:
                return bool(verify(parse(f"${gold}$"), parse("\\boxed{" + boxed + "}")))
            except Exception:  # noqa: BLE001
                return False
        return grade, "math_verify"
    except ImportError:
        def norm(s):
            s = re.sub(r"\s+|\\left|\\right|\\!|\\,|\$", "", s or "").replace("dfrac", "frac").replace("tfrac", "frac")
            return (s.lstrip("0") or "0") if s.isdigit() else s

        def grade(boxed: str, gold: str) -> bool:
            return norm(boxed) == norm(gold)
        return grade, "string"


def grade_one(text: str, gold: str) -> bool:
    global _GRADER
    if _GRADER is None:
        _GRADER = make_grader()
    b = last_boxed(text)
    return False if b is None else _GRADER[0](b, gold)


def default_workers() -> int:
    return max(1, min(8, (os.cpu_count() or 2) - 1))


class Grader:
    """grade_many([(text, gold), ...]) -> [bool, ...]. A timed-out item counts as incorrect."""

    def __init__(self, workers=None, timeout=30.0):
        self.workers = default_workers() if workers is None else workers
        self.timeout = timeout
        self.pool = mp.get_context("spawn").Pool(self.workers) if self.workers else None
        self.n_timeout = 0

    def grade_many(self, pairs):
        if self.pool is None:
            return [grade_one(t, g) for t, g in pairs]
        futs = [self.pool.apply_async(grade_one, (t, g)) for t, g in pairs]
        out = []
        for f in futs:
            try:
                out.append(bool(f.get(timeout=self.timeout)))
            except mp.TimeoutError:
                self.n_timeout += 1
                out.append(False)
        return out

    def close(self):
        if self.pool is not None:
            self.pool.terminate()
            self.pool.join()
            self.pool = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def grader_name() -> str:
    return make_grader()[1]
