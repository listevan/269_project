import math

import numpy as np
import pytest

import neg_train as nt
import select_rules as sr


def _rows(n=4, length=10, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        lp_s = (-rng.random(length) * 4).round(4).tolist()
        rows.append(dict(key=f"r{i}", correct=i % 2 == 0, lp_s=lp_s, ent_s=rng.random(length).round(4).tolist(),
                         lp_t=(np.asarray(lp_s) + rng.normal(0, 1, length)).round(4).tolist()))
    return rows


def test_opsa_and_random_budget():
    rows = _rows()
    adv = nt.advantages("opsa", rows, pool_size=4, fraction=0.2, clip=5.0, seed=0)
    flat = np.concatenate([adv[r["key"]] for r in rows])
    sel = flat != 0
    assert sel.sum() == 8                                   # 20% of 40 tokens
    lps = np.concatenate([r["lp_s"] for r in rows])
    assert set(np.where(sel)[0]) == set(np.argsort(lps, kind="stable")[:8])
    assert np.all(flat[sel] <= -0.5) and np.all(flat[sel] >= -1.0) and math.isclose(flat[sel].min(), -1.0)
    rnd = nt.advantages("random", rows, pool_size=4, fraction=0.2, clip=5.0, seed=0)
    rflat = np.concatenate([rnd[r["key"]] for r in rows])
    assert (rflat != 0).sum() == 8 and np.all(rflat[rflat != 0] <= -0.5)
    assert not np.array_equal(rflat != 0, sel)             # a different set (with overwhelming probability)
    assert nt.pool_summary(adv) == dict(n_tokens=40, n_updated=8, mean_abs_adv=round(float(-flat[sel].mean()), 4))


def test_teacher_arms():
    rows = _rows()
    neg = nt.advantages("teacher_neg", rows, 4, 0.2, clip=5.0, seed=0)
    both = nt.advantages("teacher_both", rows, 4, 0.2, clip=5.0, seed=0)
    for r in rows:
        gap = np.asarray(r["lp_t"]) - np.asarray(r["lp_s"])
        assert np.allclose(neg[r["key"]], np.clip(gap, -5, 0))
        assert np.allclose(both[r["key"]], np.clip(gap, -5, 5))
    sel = nt.advantages("teacher_sel", rows, 4, 0.2, clip=5.0, seed=0)
    flat = np.concatenate([sel[r["key"]] for r in rows])
    gaps = np.concatenate([np.asarray(r["lp_t"]) - np.asarray(r["lp_s"]) for r in rows])
    chosen = np.where(flat != 0)[0]
    assert len(chosen) <= 8 and np.all(gaps[chosen] < 0)
    assert set(chosen) == {i for i in np.argsort(gaps, kind="stable")[:8] if gaps[i] < 0}


def test_answer_key_and_sft():
    rows = _rows()
    ak = nt.advantages("answer_key", rows, 4, 0.2, clip=5.0, seed=0)
    for r in rows:
        if r["correct"]:
            assert not np.any(ak[r["key"]])
    n_sel = sum(int((ak[r["key"]] != 0).sum()) for r in rows)
    assert n_sel == 8                                      # the whole pool's budget, from the wrong responses
    wrong_lps = np.concatenate([r["lp_s"] for r in rows if not r["correct"]])
    cut = np.sort(wrong_lps)[7]
    for r in rows:
        if not r["correct"]:
            assert all(r["lp_s"][t] <= cut for t in np.where(ak[r["key"]] != 0)[0])
    all_right = [dict(r, correct=True) for r in rows]
    assert not any(np.any(v) for v in nt.advantages("answer_key", all_right, 4, 0.2, 5.0, 0).values())
    sft = nt.advantages("sft", rows, 4, 0.2, 5.0, 0)
    assert all(np.all(v == 1.0) for v in sft.values())


def test_lowest_fraction_k_override():
    vals = {"a": [-1, -2, -3], "b": [-4, -5, -6]}
    assert sr.lowest_fraction(vals, ["a", "b"], 0.2, k=2) == {"a": [], "b": [1, 2]}
    assert sr.lowest_fraction(vals, ["b"], 0.2, k=10) == {"b": [0, 1, 2]}


class FakeTok:
    pad_token_id = 0
    padding_side = "left"

    def pad(self, enc, padding=True, return_tensors="pt"):
        import torch
        seqs = enc["input_ids"]
        L = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), L), self.pad_token_id, dtype=torch.long)
        mask = torch.zeros((len(seqs), L), dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, L - len(s):] = torch.tensor(s)
            mask[i, L - len(s):] = 1
        return _Batch(input_ids=ids, attention_mask=mask)


class _Batch(dict):
    def to(self, device):
        return self


def _tiny_model():
    torch = pytest.importorskip("torch")
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, head_dim=8, max_position_embeddings=256, tie_word_embeddings=False)
    model = Qwen3ForCausalLM(cfg)
    model = get_peft_model(model, LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0, target_modules=nt.LORA_TARGETS, task_type="CAUSAL_LM"))
    return model


def test_loss_matches_manual_and_moves_lora():
    torch = pytest.importorskip("torch")
    model = _tiny_model()
    pairs = [([1, 2, 3], [4, 5, 6, 7]), ([2, 3], [8, 9, 10])]
    adv = [np.array([0.0, -1.0, 0.0, -0.5]), np.array([2.0, 0.0, -3.0])]
    model.eval()
    lps, ents = nt.response_logprobs(model, pairs, pad_id=0, batch=1, entropy=True)
    assert [len(x) for x in lps] == [4, 3] and all(v <= 0 for x in lps for v in x) and all(v >= 0 for x in ents for v in x)
    lps2, _ = nt.response_logprobs(model, pairs, pad_id=0, batch=2)
    assert np.allclose(np.concatenate(lps), np.concatenate(lps2), atol=1e-3)     # padding does not change the scores
    n = 7
    manual = -sum(float((a * np.asarray(lp)).sum()) for a, lp in zip(adv, lps)) / n
    model.train()
    loss = nt.response_logprobs(model, pairs, pad_id=0, batch=2, grad=True, adv=adv, n_norm=n)
    assert abs(loss - manual) < 1e-3
    params = nt.lora_params(model)
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in params)
    opt = torch.optim.AdamW(params, lr=1e-2)
    before = [p.detach().clone() for p in params]
    opt.step()
    assert any(not torch.equal(p, b) for p, b in zip(params, before))


def test_generate_cuts_at_end_token():
    pytest.importorskip("torch")
    model = _tiny_model().eval()
    tok = FakeTok()
    end_ids = [7]
    out = nt.generate(model, tok, [[1, 2, 3], [4, 5]], end_ids, max_tokens=12, batch=2, seed=0)
    assert len(out) == 2
    for ids, fin in out:
        assert 1 <= len(ids) <= 12
        if fin == "stop":
            assert ids[-1] == 7 and 7 not in ids[:-1]
        else:
            assert len(ids) == 12 and 7 not in ids


def test_probe_number_probability():
    torch = pytest.importorskip("torch")
    import rules_probe as rp

    model = _tiny_model().eval()
    digits = [10, 11, 12]                      # pretend these three ids are the digit tokens
    ctx, num = [1, 2, 3], [20, 10, 11]         # "{", then two digits
    lp = rp.candidate_logprobs(model, [(ctx, num)], pad_id=0, digits=digits, batch=4)[0]
    ids = torch.tensor([ctx + num])
    with torch.no_grad():
        logp = torch.log_softmax(model(input_ids=ids, use_cache=False).logits[0].float(), -1)
    manual = float(logp[2, 20] + logp[3, 10] + logp[4, 11] + torch.log1p(-logp[5, digits].exp().sum()))
    assert abs(lp - manual) < 1e-4 and lp < float(logp[2, 20] + logp[3, 10] + logp[4, 11])


def test_probe_number_ids_with_real_tokenizer_shape():
    class Tok:
        table = {"{": 1, "{-": 2, "1": 3, "8": 4, "3": 5}

        def __call__(self, text, add_special_tokens=False):
            toks = []
            if text.startswith("{-"):
                toks.append("{-"); text = text[2:]
            elif text.startswith("{"):
                toks.append("{"); text = text[1:]
            toks += list(text)
            return {"input_ids": [self.table[t] for t in toks]}

        def decode(self, ids):
            inv = {v: k for k, v in self.table.items()}
            return "".join(inv[i] for i in ids)

    import rules_probe as rp
    assert rp.number_ids(Tok(), 18) == [1, 3, 4]
    assert rp.number_ids(Tok(), -3) == [2, 5]


def test_checkpoint_resume_roundtrip(tmp_path):
    torch = pytest.importorskip("torch")
    import json
    model = _tiny_model()
    params = nt.lora_params(model)
    opt = torch.optim.AdamW(params, lr=1e-2)
    pairs = [([1, 2, 3], [4, 5, 6, 7])]
    model.train()
    nt.response_logprobs(model, pairs, 0, 1, grad=True, adv=[np.array([1.0, -1.0, 0.5, -0.5])], n_norm=4)
    opt.step()
    rng = np.random.default_rng(3)
    order = [5, 1, 2]
    nt.checkpoint(model, opt, tmp_path / "ckpt", 7, order, rng)
    log = tmp_path / "train_log.jsonl"
    log.write_text("".join(json.dumps(dict(step=s)) + "\n" for s in range(1, 10)))
    saved = [p.detach().clone() for p in params]
    model2 = _tiny_model()
    opt2 = torch.optim.AdamW(nt.lora_params(model2), lr=1e-2)
    assert any(not torch.equal(a, b) for a, b in zip(nt.lora_params(model2), saved))
    nxt = nt.resume(model2, opt2, tmp_path / "ckpt", log)
    assert nxt == 8
    assert all(torch.equal(a, b) for a, b in zip(nt.lora_params(model2), saved))
    assert opt2.state_dict()["state"] and len(log.read_text().splitlines()) == 7
    state = torch.load(tmp_path / "ckpt" / "state.pt", weights_only=False)
    assert state["order"] == order and state["rng"].integers(0, 100) == rng.integers(0, 100)
