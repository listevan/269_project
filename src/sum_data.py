"""Synthetic Unanswerable Math (SUM, lime-nlp/Synthetic_Unanswerable_Math) for study 2: unanswerable problems and
their solvable originals, prompted the way the dataset prompts them, and the refusal / false-answer grading.

The test split holds 284 unanswerable prompts, each `question + SUM_INSTRUCTION` with gold "I don't know.".
The train split pairs every unanswerable question with the DeepScaleR original it was made from and that
original's answer; all 284 test questions are in it, so each problem comes in two variants:
  <pair>:u  the unanswerable version, the wanted answer is a refusal
  <pair>:a  the solvable original, the wanted answer is its gold
Both variants get the same instruction, which tells the model it may answer \\boxed{I don't know.}.
Nothing here imports torch or vllm.
"""
import random
import re

from common import last_boxed

SUM_INSTRUCTION = ("Let's think step by step and output the final answer within \\boxed{}. "
                   "If you don't know the answer, reply with \\boxed{I don't know.}.")
REFUSAL_GOLD = "I don't know."
_SPLIT = re.compile(r"\s*Let[’']s think step by step.*$", re.S)


def load_sum(n_problems: int, seed: int):
    """A fixed random subset of SUM's test problems, each as two problem dicts (unanswerable, answerable)."""
    from huggingface_hub import hf_hub_download
    import pandas as pd

    repo = "lime-nlp/Synthetic_Unanswerable_Math"
    test = pd.read_parquet(hf_hub_download(repo, "test.parquet", repo_type="dataset"))
    train = pd.read_parquet(hf_hub_download(repo, "synthetic_unanswerable_math.parquet", repo_type="dataset"))
    by_u = train.assign(u=train.unanswerable_question.str.strip()).drop_duplicates("u").set_index("u")
    rows = []
    for i, prompt in enumerate(test.prompt):
        q = _SPLIT.sub("", prompt).strip()
        if q not in by_u.index:
            raise ValueError(f"SUM test question {i} has no original in the train split: {q[:80]}")
        o = by_u.loc[q]
        rows.append(dict(pair=f"sum{i:03d}", unanswerable=q, answerable=str(o.answerable_question).strip(),
                         gold=str(o.ground_truth)))
    random.Random(seed).shuffle(rows)
    out = []
    for r in rows[:n_problems]:
        out.append(dict(problem_id=f"{r['pair']}:u", pair=r["pair"], variant="unanswerable", level=0, subject="sum",
                        question=r["unanswerable"], gold=REFUSAL_GOLD))
        out.append(dict(problem_id=f"{r['pair']}:a", pair=r["pair"], variant="answerable", level=0, subject="sum",
                        question=r["answerable"], gold=r["gold"]))
    return out


# --------------------------------------------------------------------------- grading
_SEP = r"(?:\s|\*|_)+"            # words may be separated by spaces or markdown emphasis
_APO = "[\u2019']"
_IDK = re.compile(r"\b(?:don" + _APO + r"?t|do" + _SEP + r"not|dont)" + _SEP + r"know\b", re.I)
_HEDGE_BOX = re.compile(
    r"\b(?:cannot|can" + _APO + r"?t|could" + _SEP + r"not|couldn" + _APO + r"?t|unable" + _SEP + r"to|impossible" + _SEP + r"to|not"
    + _SEP + r"possible" + _SEP + r"to)" + _SEP + r"(?:be" + _SEP + r")?(?:\w+ly" + _SEP + r")?(?:determined?|known?|answer(?:ed)?|"
    r"comput(?:e|ed)|calculat(?:e|ed)|solved|found|evaluated?|concluded?)\b"
    r"|\bnot" + _SEP + r"enough" + _SEP + r"(?:information|data|details|info)\b|\binsufficient" + _SEP + r"(?:information|data|details)\b"
    r"|\b(?:undetermined|indeterminate|unanswerable|unknown|unsolvable|undeterminable|not" + _SEP + r"determinable|n/a)\b"
    r"|\bno" + _SEP + r"(?:unique|definite|definitive)" + _SEP + r"(?:answer|solution|value)\b|^\?$", re.I)


def boxed_kind(text: str) -> str:
    """What the last \\boxed{} (or \\fbox{}) holds: 'idk' (the dataset's own refusal, "I don't know"), 'hedge' (another
    statement that the answer cannot be given), 'answer' (anything else), or 'none' (no complete box, or an empty one)."""
    b = last_boxed(text)
    if b is None:
        return "none"
    plain = re.sub(r"\\(text|mathrm|textbf|textit|mbox)\s*\{([^{}]*)\}", r"\2", b)
    plain = plain.replace("\u2019", "'").replace("\\", " ").strip(" .,!")
    if not plain:
        return "none"
    if _IDK.search(plain):
        return "idk"
    if _HEDGE_BOX.search(plain):
        return "hedge"
    return "answer"


REFUSAL_KINDS = ("idk", "hedge")


def outcome(text: str, variant: str, correct_answer: bool) -> str:
    """unanswerable: refusal | false_answer | no_answer.  answerable: correct | wrong | refusal | no_answer.
    correct_answer is the math-verify verdict against the gold (ignored for the unanswerable variant)."""
    kind = boxed_kind(text)
    if kind == "none":
        return "no_answer"
    if kind in REFUSAL_KINDS:
        return "refusal"
    if variant == "unanswerable":
        return "false_answer"
    return "correct" if correct_answer else "wrong"


def wanted(outcome_: str, variant: str) -> bool:
    """The behaviour each variant is graded on: a refusal on an unanswerable problem, a correct answer otherwise."""
    return outcome_ == ("refusal" if variant == "unanswerable" else "correct")
