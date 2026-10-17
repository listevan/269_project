"""The four rules for which response tokens a negative-only method pushes down, and the replay sample.

All of them act on a pool of responses, as OPSA does: its selector concatenates the response tokens of one
data-parallel rank's packed batch (64 prompts x 1 sample split over 4 actor GPUs, so 16 responses for the
Qwen3-1.7B preset) and takes the K = max(1, floor(f N)) tokens with the lowest actor log-probability,
f = 0.2, stable order, no EOS or "think" mask (DripNowhy/On-Policy-Self-Adaptation,
slime/backends/megatron_utils/opsa.py, read 2026-10-16).

  opsa         the lowest-K student log-probabilities in the pool
  random       K tokens drawn uniformly from the pool (does picking the unlikely ones matter?)
  teacher_neg  the lowest-K values of log p_teacher - log p_student, negative ones only: the tokens the
               teacher likes least relative to the student, i.e. the negative half of the on-policy
               distillation signal, cut to OPSA's budget
  nsr          every token of a failed response (negative sample reinforcement uses the answer key)

Pure numpy, no GPU, so tests/test_select.py checks it on the laptop.
"""
import math
import random

import numpy as np

from common import stable_seed


def make_pools(keys, pool_size: int, seed: int):
    keys = sorted(keys)
    random.Random(seed).shuffle(keys)
    return [keys[i:i + pool_size] for i in range(0, len(keys), pool_size)]


def _flat(values_by_key, keys):
    vals = [np.asarray(values_by_key[k], dtype=np.float64) for k in keys]
    flat = np.concatenate(vals) if vals else np.zeros(0)
    owner = np.concatenate([np.full(len(v), i) for i, v in enumerate(vals)]) if vals else np.zeros(0, int)
    pos = np.concatenate([np.arange(len(v)) for v in vals]) if vals else np.zeros(0, int)
    return flat, owner, pos


def budget(n_tokens: int, fraction: float) -> int:
    return 0 if n_tokens == 0 else min(n_tokens, max(1, math.floor(fraction * n_tokens)))


def lowest_fraction(values_by_key, keys, fraction: float, only_negative=False):
    """OPSA's selector on one pool. Returns {key: sorted positions}."""
    flat, owner, pos = _flat(values_by_key, keys)
    k = budget(len(flat), fraction)
    idx = np.argsort(flat, kind="stable")[:k]
    if only_negative:
        idx = idx[flat[idx] < 0]
    out = {key: [] for key in keys}
    for i in idx:
        out[keys[owner[i]]].append(int(pos[i]))
    return {key: sorted(v) for key, v in out.items()}


def random_fraction(lengths_by_key, keys, fraction: float, seed: int):
    """K = OPSA's budget, drawn uniformly from the pool's tokens."""
    n = sum(lengths_by_key[k] for k in keys)
    k = budget(n, fraction)
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=k, replace=False) if k else np.zeros(0, int)
    owner = np.concatenate([np.full(lengths_by_key[key], i) for i, key in enumerate(keys)])
    pos = np.concatenate([np.arange(lengths_by_key[key]) for key in keys])
    out = {key: [] for key in keys}
    for i in idx:
        out[keys[owner[i]]].append(int(pos[i]))
    return {key: sorted(v) for key, v in out.items()}


def opsa_advantages(entropies, a_min=-1.0, a_max=-0.5):
    """OPSA's per-token advantage for the selected tokens of one pool: entropy min-max ranked,
    A = a_max + (a_min - a_max) r, so the most uncertain selected token gets -1.0, the least -0.5;
    all -1.0 if the entropies are equal."""
    h = np.asarray(entropies, dtype=np.float64)
    if h.size == 0:
        return h
    rng = h.max() - h.min()
    r = np.ones_like(h) if rng <= 1e-12 else (h - h.min()) / rng
    return a_max + (a_min - a_max) * r


def selected_sets(rows, pool_size=16, fraction=0.2, seed=0):
    """rows: one dict per response with key, correct, lp_s, lp_t, ent_s (per-token lists of equal length).
    Returns ({method: {key: positions}}, {key: {position: opsa advantage}}, pools)."""
    by_key = {r["key"]: r for r in rows}
    pools = make_pools(by_key, pool_size, seed)
    lp_s = {k: r["lp_s"] for k, r in by_key.items()}
    gap = {k: (np.asarray(r["lp_t"]) - np.asarray(r["lp_s"])).tolist() for k, r in by_key.items()}
    lengths = {k: len(r["lp_s"]) for k, r in by_key.items()}
    sets = {m: {} for m in ("opsa", "random", "teacher_neg", "nsr")}
    adv = {}
    for p_idx, keys in enumerate(pools):
        opsa = lowest_fraction(lp_s, keys, fraction)
        sets["opsa"].update(opsa)
        sel = [(k, t) for k in keys for t in opsa[k]]
        a = opsa_advantages([by_key[k]["ent_s"][t] for k, t in sel])
        for (k, t), v in zip(sel, a):
            adv.setdefault(k, {})[t] = float(v)
        sets["random"].update(random_fraction(lengths, keys, fraction, stable_seed("random", seed, p_idx)))
        sets["teacher_neg"].update(lowest_fraction(gap, keys, fraction, only_negative=True))
    for k, r in by_key.items():
        sets["nsr"][k] = [] if r["correct"] else list(range(lengths[k]))
    return sets, adv, pools


def sample_positions(sets, n_per_method: int, seed: int):
    """Uniform sample (without replacement) of n positions from each method's selected set. A position drawn
    by several methods is replayed once. Returns [{pos_id, key, t, methods}], sorted by pos_id."""
    picked = {}
    for m in sorted(sets):
        pool = [(k, t) for k in sorted(sets[m]) for t in sets[m][k]]
        rng = random.Random(stable_seed("positions", seed, m))
        for k, t in rng.sample(pool, min(n_per_method, len(pool))):
            pid = f"{k}@{t}"
            picked.setdefault(pid, dict(pos_id=pid, key=k, t=t, methods=[]))["methods"].append(m)
    return [picked[p] for p in sorted(picked)]


def in_sets(sets, key, t):
    """Every method whose selected set holds (key, t)."""
    return [m for m in sorted(sets) if t in set(sets[m].get(key, ()))]
