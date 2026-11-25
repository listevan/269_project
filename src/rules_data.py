"""Experiment 3's task: a hidden binary operation.

Items are prompts such as "Compute 7 # 4." with the definition (a # b = 2a + b) left out, so the correct answer
starts rare and the dominant wrong answer (11, from reading # as plus) is identifiable. CANDIDATES lists more
symbol/definition pairs than the study uses; run_rules.py measures the base model's correct rate on each and
pick_operators keeps about twelve that span base rates from near 0% to about 50% (the initial-rarity dial).

Splits per operator (all deterministic from the seed and the symbol):
  train        operand pairs from TRAIN_RANGE, the prompts every training arm samples from
  new_numbers  held-out pairs from the same range: the primary transfer test
  new_range    pairs from NEW_RANGE
  composition  a # (b # c) with operands from COMP_RANGE
  with_def     the new_numbers items with the definition stated in the prompt: the sanity check
The teacher of experiment 4 is the same model with the definition in its prompt (teacher_prompt). The
demonstration control trains on demonstration(): the definition, the computation and the boxed answer.

Pure Python, no torch, so tests/test_rules_data.py runs anywhere.
"""
import random
import re

from common import last_boxed, stable_seed

# Chosen among five wordings on the base model (2026-11-24, 26 candidates x 4 items x 4 samples, 512-token cap):
# "Put your final answer within \\boxed{}." alone made the model list several readings and box several answers
# (7% truncated, 26% no answer, 288 tokens); this wording gives one committed answer (0.5% truncated, 9% no answer,
# 147 tokens) and the widest spread of base rates. With the definition stated, the first sentence is dropped.
RULE_INSTRUCTION = ("Here the symbol denotes one specific binary operation. Decide what it is and give exactly one final "
                    "answer within \\boxed{}, with at most two sentences of explanation.")
DEF_INSTRUCTION = "Give exactly one final answer within \\boxed{}, with at most two sentences of explanation."
TRAIN_RANGE = (1, 20)
NEW_RANGE = (21, 60)
COMP_RANGE = (1, 12)
SPLITS = ("new_numbers", "new_range", "composition", "with_def")
OUTCOMES = ("correct", "dominant_wrong", "other_wrong", "no_answer")

def _pow(a, b):
    """a^b as the habit reading of ^-like symbols; None when it would be astronomically large (compositions)."""
    return a ** b if 0 <= b <= 64 else None


# sym, symbolic definition, function, numeric rendering for the demonstration, the habit the base model is
# expected to follow (its name and function; only a prediction, the dominant wrong answer is measured)
CANDIDATES = [
    dict(sym="#", expr="2a + b", fn=lambda a, b: 2 * a + b, show=lambda a, b: f"2*{a} + {b}", habit="a + b", habit_fn=lambda a, b: a + b),
    dict(sym="$", expr="a + 2b", fn=lambda a, b: a + 2 * b, show=lambda a, b: f"{a} + 2*{b}", habit="a + b", habit_fn=lambda a, b: a + b),
    dict(sym="@", expr="ab + a", fn=lambda a, b: a * b + a, show=lambda a, b: f"{a}*{b} + {a}", habit="ab", habit_fn=lambda a, b: a * b),
    dict(sym="&", expr="ab + b", fn=lambda a, b: a * b + b, show=lambda a, b: f"{a}*{b} + {b}", habit="ab", habit_fn=lambda a, b: a * b),
    dict(sym="~", expr="|a - b|", fn=lambda a, b: abs(a - b), show=lambda a, b: f"|{a} - {b}|", habit="a - b", habit_fn=lambda a, b: a - b),
    dict(sym="§", expr="a + b - 1", fn=lambda a, b: a + b - 1, show=lambda a, b: f"{a} + {b} - 1", habit="a + b", habit_fn=lambda a, b: a + b),
    dict(sym="¤", expr="2(a + b)", fn=lambda a, b: 2 * (a + b), show=lambda a, b: f"2*({a} + {b})", habit="a + b", habit_fn=lambda a, b: a + b),
    dict(sym="†", expr="3a - b", fn=lambda a, b: 3 * a - b, show=lambda a, b: f"3*{a} - {b}", habit="a - b", habit_fn=lambda a, b: a - b),
    dict(sym="‡", expr="a + 3b", fn=lambda a, b: a + 3 * b, show=lambda a, b: f"{a} + 3*{b}", habit="a + b", habit_fn=lambda a, b: a + b),
    dict(sym="⊕", expr="a + b", fn=lambda a, b: a + b, show=lambda a, b: f"{a} + {b}", habit="a XOR b", habit_fn=lambda a, b: a ^ b),
    dict(sym="⊗", expr="ab", fn=lambda a, b: a * b, show=lambda a, b: f"{a}*{b}", habit="a + b", habit_fn=lambda a, b: a + b),
    dict(sym="⊙", expr="ab + 1", fn=lambda a, b: a * b + 1, show=lambda a, b: f"{a}*{b} + 1", habit="ab", habit_fn=lambda a, b: a * b),
    dict(sym="⊞", expr="a + b", fn=lambda a, b: a + b, show=lambda a, b: f"{a} + {b}", habit="ab", habit_fn=lambda a, b: a * b),
    dict(sym="⊠", expr="ab", fn=lambda a, b: a * b, show=lambda a, b: f"{a}*{b}", habit="a + b", habit_fn=lambda a, b: a + b),
    dict(sym="⊟", expr="a - b", fn=lambda a, b: a - b, show=lambda a, b: f"{a} - {b}", habit="a + b", habit_fn=lambda a, b: a + b),
    dict(sym="∧", expr="min(a, b)", fn=min, show=lambda a, b: f"min({a}, {b})", habit="a^b", habit_fn=lambda a, b: _pow(a, b)),
    dict(sym="∨", expr="max(a, b)", fn=max, show=lambda a, b: f"max({a}, {b})", habit="a + b", habit_fn=lambda a, b: a + b),
    dict(sym="↑", expr="max(a, b)", fn=max, show=lambda a, b: f"max({a}, {b})", habit="a^b", habit_fn=lambda a, b: _pow(a, b)),
    dict(sym="↓", expr="min(a, b)", fn=min, show=lambda a, b: f"min({a}, {b})", habit="a - b", habit_fn=lambda a, b: a - b),
    dict(sym="△", expr="|a - b|", fn=lambda a, b: abs(a - b), show=lambda a, b: f"|{a} - {b}|", habit="a + b", habit_fn=lambda a, b: a + b),
    dict(sym="∇", expr="ab - a - b", fn=lambda a, b: a * b - a - b, show=lambda a, b: f"{a}*{b} - {a} - {b}", habit="ab", habit_fn=lambda a, b: a * b),
    dict(sym="◇", expr="a^2 + b", fn=lambda a, b: a * a + b, show=lambda a, b: f"{a}^2 + {b}", habit="a + b", habit_fn=lambda a, b: a + b),
    dict(sym="⋆", expr="a + b + ab", fn=lambda a, b: a + b + a * b, show=lambda a, b: f"{a} + {b} + {a}*{b}", habit="ab", habit_fn=lambda a, b: a * b),
    dict(sym="♣", expr="ab - b", fn=lambda a, b: a * b - b, show=lambda a, b: f"{a}*{b} - {b}", habit="ab", habit_fn=lambda a, b: a * b),
    dict(sym="♠", expr="a^2 - b", fn=lambda a, b: a * a - b, show=lambda a, b: f"{a}^2 - {b}", habit="a - b", habit_fn=lambda a, b: a - b),
    dict(sym="♦", expr="2a - b", fn=lambda a, b: 2 * a - b, show=lambda a, b: f"2*{a} - {b}", habit="a - b", habit_fn=lambda a, b: a - b),
]
BY_SYM = {o["sym"]: o for o in CANDIDATES}
assert len(BY_SYM) == len(CANDIDATES), "duplicate symbol"


# --------------------------------------------------------------------------- prompts
def question(sym, a, b):
    return f"Compute {a} {sym} {b}."


def comp_question(sym, a, b, c):
    return f"Compute {a} {sym} ({b} {sym} {c})."


def definition(op):
    return f"The operation {op['sym']} is defined by a {op['sym']} b = {op['expr']}."


def prompt_body(op, item, with_def=False):
    """The user message without the instruction: the question, preceded by the definition for the teacher and
    the with_def split."""
    q = comp_question(op["sym"], *item["operands"]) if item["kind"] == "composition" else question(op["sym"], *item["operands"])
    return f"{definition(op)} {q}" if with_def else q


def instruction_for(with_def=False):
    return DEF_INSTRUCTION if with_def else RULE_INSTRUCTION


def prompt_text_for(tok, item, with_def=False):
    """The full chat prompt (thinking off) of an item; with_def states the definition (the teacher, the with_def split)."""
    from common import prompt_text
    return prompt_text(tok, prompt_body(BY_SYM[item["sym"]], item, with_def), instruction_for(with_def))


def prompt_ids_for(tok, item, with_def=False):
    return tok(prompt_text_for(tok, item, with_def), add_special_tokens=False)["input_ids"]


def demonstration(op, item):
    """The demonstration control's target: definition, computation, boxed answer (direct items only)."""
    a, b = item["operands"]
    return (f"Here {op['sym']} is defined by a {op['sym']} b = {op['expr']}, so "
            f"{a} {op['sym']} {b} = {op['show'](a, b)} = {item['answer']}.\n\\boxed{{{item['answer']}}}")


# --------------------------------------------------------------------------- items
def _pairs(lo, hi):
    return [(a, b) for a in range(lo, hi + 1) for b in range(lo, hi + 1) if a != b]


def _item(op, kind, operands, split, idx):
    if kind == "composition":
        a, b, c = operands
        ans = op["fn"](a, op["fn"](b, c))
        inner = op["habit_fn"](b, c)
        habit = None if inner is None else op["habit_fn"](a, inner)
    else:
        ans = op["fn"](*operands)
        habit = op["habit_fn"](*operands)
    if habit is not None and abs(habit) > 10 ** 15:
        habit = None   # the model will not write it anyway
    return dict(item_id=f"{op['sym']}:{split}:{idx}", sym=op["sym"], split=split, kind=kind, operands=list(operands),
                answer=int(ans), habit=None if habit is None else int(habit))


def make_items(sym, n_train, n_eval, seed=0):
    """{split: [items]} for one operator. new_numbers and train are disjoint pairs of TRAIN_RANGE."""
    op = BY_SYM[sym]
    rng = random.Random(stable_seed("rules", seed, sym))
    pairs = _pairs(*TRAIN_RANGE)
    rng.shuffle(pairs)
    if n_train + n_eval > len(pairs):
        raise ValueError(f"{n_train} + {n_eval} items exceed the {len(pairs)} pairs of {TRAIN_RANGE}")
    new_numbers = pairs[:n_eval]
    train = pairs[n_eval:n_eval + n_train]
    far = _pairs(*NEW_RANGE)
    rng.shuffle(far)
    lo, hi = COMP_RANGE
    triples = [(rng.randint(lo, hi), rng.randint(lo, hi), rng.randint(lo, hi)) for _ in range(n_eval)]
    out = {
        "train": [_item(op, "direct", p, "train", i) for i, p in enumerate(train)],
        "new_numbers": [_item(op, "direct", p, "new_numbers", i) for i, p in enumerate(new_numbers)],
        "new_range": [_item(op, "direct", p, "new_range", i) for i, p in enumerate(far[:n_eval])],
        "composition": [_item(op, "composition", t, "composition", i) for i, t in enumerate(triples)],
    }
    out["with_def"] = [dict(it, item_id=f"{sym}:with_def:{i}", split="with_def") for i, it in enumerate(out["new_numbers"])]
    return out


def all_items(syms, n_train, n_eval, seed=0):
    """[items] over operators, each with sym, split, kind, operands, answer, habit."""
    return [it for sym in syms for split, items in make_items(sym, n_train, n_eval, seed).items() for it in items]


# --------------------------------------------------------------------------- answers
_INT = re.compile(r"[+-]?\d+")


def parse_int(boxed):
    """The integer in a boxed answer, or None: one integer after stripping \\text{}, $, spaces and thousands
    commas; with an '=' the part after the last one."""
    if boxed is None:
        return None
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", boxed)
    s = re.sub(r"\\[,;!]", "", s)                   # thin spaces, also used as thousands separators: 1\,234
    s = re.sub(r"\\[a-zA-Z]+", " ", s)             # \quad \cdot ...
    s = s.replace("$", "").replace(",", "").replace("{", " ").replace("}", " ")
    if "=" in s:
        s = s.rsplit("=", 1)[1]
    toks = s.split()
    if len(toks) != 1:
        return None
    t = toks[0].rstrip(".")
    if _INT.fullmatch(t):
        return int(t)
    if re.fullmatch(r"[+-]?\d+\.0+", t):
        return int(float(t))
    return None


def answer_of(text):
    return parse_int(last_boxed(text))


def outcome(answer, item, dominant_wrong):
    """correct / dominant_wrong / other_wrong / no_answer. dominant_wrong: the item's dominant wrong answer
    from the base model's samples (None if the base had no wrong answer)."""
    if answer is None:
        return "no_answer"
    if answer == item["answer"]:
        return "correct"
    if dominant_wrong is not None and answer == dominant_wrong:
        return "dominant_wrong"
    return "other_wrong"


def dominant_wrong_of(answers, correct):
    """The most common wrong integer among sampled answers (ties: the smaller), None if there is none."""
    counts = {}
    for a in answers:
        if a is not None and a != correct:
            counts[a] = counts.get(a, 0) + 1
    if not counts:
        return None
    return sorted(counts, key=lambda a: (-counts[a], a))[0]


# --------------------------------------------------------------------------- operator choice
def pick_operators(rates, n=12, max_rate=0.55, n_zero=3, zero=0.005):
    """From {sym: base correct rate}, keep at most n operators spanning the rarity dial: at most n_zero whose rate is
    below `zero` (many candidates sit at exactly 0, and they all say the same thing), then the operators with a
    measurable rate at or below max_rate, spread evenly from the lowest to the highest (the two extremes, then the
    symbol nearest each evenly spaced target rate). Fewer than n come back when few operators have a measurable rate."""
    zeros = sorted((s for s, r in rates.items() if r < zero), key=lambda s: (rates[s], s))
    cands = sorted(((r, s) for s, r in rates.items() if zero <= r <= max_rate), key=lambda x: (x[0], x[1]))
    chosen_zero = [(rates[s], s) for s in zeros[:n_zero]]
    m = n - len(chosen_zero)
    if len(cands) <= m:
        return [s for _, s in sorted(chosen_zero + cands)]
    chosen = [cands[0], cands[-1]]
    lo, hi = cands[0][0], cands[-1][0]
    targets = [lo + (hi - lo) * (i + 1) / (m - 1) for i in range(m - 2)]
    for t in targets:
        rest = [c for c in cands if c not in chosen]
        chosen.append(min(rest, key=lambda c: (abs(c[0] - t), c[1])))
    return [s for _, s in sorted(chosen_zero + chosen)]
