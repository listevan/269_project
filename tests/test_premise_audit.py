import numpy as np

import premise_audit as pa


class FakeTok:
    """Each id decodes to a fixed string; 101 and 102 are two byte pieces that only decode together, to one character."""
    STR = {1: "The", 2: " answer", 3: " is", 4: " \\boxed", 5: "{", 6: "I", 7: " don't", 8: " know", 9: ".", 10: "}", 11: "<|im_end|>",
           12: " cannot", 13: " be", 14: " determined", 15: "42", 16: ",", 17: "{\\", 18: "text", 19: "{I", 20: " don't know.}"}

    def decode(self, ids, skip_special_tokens=False):
        out, i = "", 0
        while i < len(ids):
            if ids[i] == 101 and i + 1 < len(ids) and ids[i + 1] == 102:
                out += "\u2297"; i += 2
            elif ids[i] >= 100:                      # a lone byte piece decodes to nothing
                i += 1
            else:
                out += self.STR[ids[i]]; i += 1
        return out


def test_token_spans_concatenate_and_map_back():
    tok = FakeTok()
    ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    text, spans = pa.token_spans(tok, ids)
    assert text == "The answer is \\boxed{I don't know.}<|im_end|>"
    assert [text[a:b] for a, b in spans] == [FakeTok.STR[i] for i in ids]
    box = pa.last_box(text)
    assert text[box[0]:box[1]] == "I don't know."
    assert pa.token_at(spans, box[0]) == 5          # the "I"
    assert pa.tokens_in(spans, *box) == [5, 6, 7, 8]   # I don't know .
    assert pa.token_at(spans, box[0]) == ids.index(6)


def test_decision_token_skips_brace_backslash_glue():
    tok = FakeTok()
    text, spans = pa.token_spans(tok, [1, 4, 5, 6, 7, 8, 9, 10])          # \boxed{I don't know.}
    t, glue = pa.decision_token(text, spans, pa.last_box(text))
    assert (text[slice(*spans[t])], glue) == ("I", "")
    text, spans = pa.token_spans(tok, [1, 4, 17, 18, 19, 20, 10])         # \boxed{\text{I don't know.}}
    t, glue = pa.decision_token(text, spans, pa.last_box(text))
    assert (text[slice(*spans[t])], glue) == ("text", "\\")
    text, spans = pa.token_spans(tok, [1, 4, 5, 10])                      # \boxed{} : no decision
    assert pa.decision_token(text, spans, pa.last_box(text)) == (None, "")
    assert pa.decision_token("no box", [(0, 6)], None) == (None, "")


def test_token_spans_fall_back_to_prefix_decoding_for_split_bytes():
    tok = FakeTok()
    ids = [1, 101, 102, 2]                       # two half-byte pieces that only decode together to "⊗"
    text, spans = pa.token_spans(tok, ids)
    assert text == "The\u2297 answer"
    assert [text[a:b] for a, b in spans] == ["The", "", "\u2297", " answer"]


def test_last_box_ignores_an_unclosed_box():
    assert pa.last_box("so \\boxed{42") is None
    assert pa.last_box("\\boxed{1} then \\boxed{\\frac{1}{2}}") == (len("\\boxed{1} then \\boxed{"), len("\\boxed{1} then \\boxed{\\frac{1}{2}"))


def test_hedge_spans_only_before_the_box():
    text = "The value cannot be determined from this. \\boxed{I don't know.}"
    cut = text.index("\\boxed{")
    assert [h[2] for h in pa.hedge_spans(text, cut)] == ["cannot be determined"]
    assert pa.hedge_spans("We assume x = 2 and compute.", None) == []


def test_hedge_regex_on_qwen_phrasings():
    hits = ["we **cannot** compute", "we **cannot** determine", "we don\u2019t know", "can\u2019t determine", "We don't have enough information",
            "the problem does not specify the value", "cannot be uniquely determined", "cannot be directly computed",
            "without more information", "the constant N is missing", "lack of information", "the value of N is not given",
            "it is not stated what N is", "this is ambiguous", "the problem is not well-defined", "We don't know N"]
    for s in hits:
        assert pa.hedge_spans(s), s
    misses = ["unambiguous", "indeterminate form 0/0", "the function is not defined at x=0", "we cannot solve this directly, so",
              "we cannot find an integer solution", "we compute the value", "assume N = 2"]
    for s in misses:
        assert not pa.hedge_spans(s), s


def test_severe_repetition():
    assert not pa.severe_repetition(list(range(100)))
    assert pa.severe_repetition(list(range(20)) * 3)
    assert not pa.severe_repetition(list(range(20)) * 2)


def test_boot_share_clusters():
    m, lo, hi = pa.boot_share([1, 1, 0, 0], ["a", "a", "b", "b"])
    assert m == 0.5 and lo <= 0.5 <= hi and lo >= 0 and hi <= 1
    assert np.isnan(pa.boot_share([], [])[0])


def _synthetic():
    tok = FakeTok()
    rows, scores, selected, decisions = [], [], [], []
    for pair, (ids, variant, outcome) in enumerate([
            ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], "unanswerable", "refusal"),
            ([1, 2, 12, 13, 14, 16, 4, 5, 6, 7, 8, 9, 10, 11], "unanswerable", "refusal"),
            ([1, 2, 3, 4, 5, 15, 10, 11], "unanswerable", "false_answer"),
            ([1, 2, 3, 4, 5, 15, 10, 11], "answerable", "correct")]):
        key = f"p{pair}:{variant[0]}#0"
        text, spans = pa.token_spans(tok, ids)
        box = pa.last_box(text)
        t_first, glue = pa.decision_token(text, spans, box)
        t_brace = pa.token_at(spans, box[0] - 1)
        rows.append(dict(key=key, problem_id=f"p{pair}:{variant[0]}", pair=f"p{pair}", sample_idx=0, variant=variant, outcome=outcome,
                         text=text.replace("<|im_end|>", ""), response_ids=ids, n_tokens=len(ids), truncated=False,
                         correct=outcome in ("refusal", "correct")))
        lp = [-0.1] * len(ids); lp[t_first] = -2.0                     # the first content token is unlikely
        scores.append(dict(key=key, problem_id=rows[-1]["problem_id"], correct=rows[-1]["correct"], lp_s=lp, ent_s=[1.0] * len(ids),
                           tokens=[text[a:b] for a, b in spans]))
        selected.append(dict(key=key, problem_id=rows[-1]["problem_id"], n=len(ids), opsa=[t_first], opsa_adv=[-1.0], random=[0]))
        after_bare = {"I": 0.4 if variant == "unanswerable" else 0.0, "digit": 0.5, "top": [], "entropy": 1.0}
        decisions.append(dict(key=key, problem_id=rows[-1]["problem_id"], variant=variant, outcome=outcome, has_box=True, box_chars=list(box),
                              hedges=[list(h) for h in pa.hedge_spans(text, box[0] - 1)],
                              decision=dict(brace=dict(t=t_brace, kept_id=ids[t_brace], kept_str="{", p_kept=0.9, p_bare=0.9, p_backslash=0.1,
                                                       entropy=0.5, top=[]),
                                            after_bare=after_bare, after_backslash={"text": 0.3, "frac": 0.6, "top": [], "entropy": 1.0},
                                            first=dict(t=t_first, kept_id=ids[t_first], kept_str=FakeTok.STR[ids[t_first]], glue=glue),
                                            box_content=text[box[0]:box[1]])))
    return rows, scores, selected, decisions


def test_rates_and_audit_tables_on_synthetic_run():
    rows, scores, selected, decisions = _synthetic()
    rates = pa.rates_table(rows)
    un = rates[rates.variant == "unanswerable"].iloc[0]
    assert un.responses == 3 and un.refusal.startswith("0.667") and un.false_answer.startswith("0.333")
    an = rates[rates.variant == "answerable"].iloc[0]
    assert an.correct.startswith("1.000")
    premise, decision, first_tokens, mass, mass_pos, worded = pa.audit_tables(rows, scores, selected, decisions, 0.2)
    p = premise.set_index("tokens")
    first = p.loc["refusal: first content token of the boxed refusal"]
    assert first.n == 2 and first["share in OPSA's set [95% CI]"].startswith("1.000")    # both decision tokens were selected
    every = p.loc["refusal: every token of the boxed refusal"]
    assert every.n == 8 and every["share in OPSA's set [95% CI]"].startswith("0.250")
    hedge = p.loc["hedge phrase in the reasoning of refusing responses: first token"]
    assert hedge.n == 1 and hedge["share in OPSA's set [95% CI]"].startswith("0.000")
    assert p.loc["hedge phrase in the reasoning of false-answer responses: first token"].n == 0
    assert p.loc["false answer: first content token of the box (unanswerable)"].n == 1
    assert list(first_tokens.iloc[:, 0]) == ["|I"]
    dec = decision.set_index("responses")
    assert dec.loc["unanswerable, refused", "n"] == 2
    assert abs(dec.loc["unanswerable, refused", "P(refusal path '{' 'I'), mean"] - 0.36) < 1e-6     # 0.9 * 0.4
    assert dec.loc["original, correct", "P(refusal path '{' 'I'), mean"] == 0.0
    assert mass_pos["opsa_tokens_in_refusing_responses"] == 2 and mass_pos["refusing_responses_with_no_hedge_detected"] == 1
    assert worded.empty
