"""Shared pieces of the token replay pilot: models, prompt, MATH-500 subset, JSONL files, answer extraction.

Nothing here imports torch or vllm, so the laptop tests and the analysis can use it.
"""
import glob
import hashlib
import json
import os
import random
import shutil
import sys
from pathlib import Path

STUDENT = "Qwen/Qwen3-1.7B"   # the model OPSA trains (released OPSA checkpoint: Tuwhy/Qwen3-1.7B-OPSA)
TEACHER = "Qwen/Qwen3-4B"     # same tokenizer, so its per-token scores line up with the student's tokens

# Qwen's own math instruction. OPSA trains and evaluates Qwen3 with thinking disabled.
INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."
END_TOKENS = ("<|im_end|>", "<|endoftext|>")

METHODS = ("opsa", "random", "teacher_neg", "nsr")


# --------------------------------------------------------------------------- prompt and tokens
def prompt_text(tok, question: str) -> str:
    msgs = [{"role": "user", "content": f"{question}\n{INSTRUCTION}"}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)


def prompt_ids(tok, question: str) -> list[int]:
    return tok(prompt_text(tok, question), add_special_tokens=False)["input_ids"]


def end_token_ids(tok) -> list[int]:
    """Ids of the end-of-turn tokens. vLLM strips special tokens before matching stop strings, so these
    must be passed as stop_token_ids."""
    vocab = tok.get_vocab()
    ids = [vocab[t] for t in END_TOKENS if t in vocab]
    if not ids:
        raise ValueError("tokenizer has none of " + ", ".join(END_TOKENS))
    return ids


# --------------------------------------------------------------------------- local GPU
def ensure_c_compiler():
    """Triton builds a small C launcher for every kernel it compiles (vLLM's torch.compile and its own kernels),
    so the vLLM stages need a C compiler. On a machine with none (and no sudo to install one), point Triton at
    tools/rocm_cc, which wraps the clang that the pip ROCm SDK ships. Call before importing vllm or torch."""
    if os.environ.get("CC") or shutil.which("gcc") or shutil.which("clang"):
        return
    clang = sorted(glob.glob(f"{sys.prefix}/lib/python3*/site-packages/_rocm_sdk_core/lib/llvm/bin/clang"))
    if clang:
        os.environ["ROCM_SDK_CLANG"] = clang[0]
        os.environ["CC"] = str(Path(__file__).resolve().parents[1] / "tools" / "rocm_cc")


# --------------------------------------------------------------------------- data
def load_math500(levels, n_problems: int, seed: int):
    """A fixed random subset of MATH-500 at the given levels (same subset for the same arguments)."""
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    rows = sorted((r for r in ds if int(r["level"]) in set(levels)), key=lambda r: r["unique_id"])
    random.Random(seed).shuffle(rows)
    return [dict(problem_id=r["unique_id"], level=int(r["level"]), subject=r["subject"],
                 question=r["problem"], gold=r["answer"]) for r in rows[:n_problems]]


def response_key(problem_id: str, sample_idx: int) -> str:
    return f"{problem_id}#{sample_idx}"


def stable_seed(*parts) -> int:
    """A 31-bit seed from any parts, the same on every machine (Python's hash() is salted per process)."""
    h = hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()
    return int(h[:8], 16) & 0x7FFFFFFF


# --------------------------------------------------------------------------- files
def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl_atomic(path, rows):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    os.replace(tmp, path)


def append_jsonl(path, rows):
    with open(path, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
        f.flush()
        os.fsync(f.fileno())


def read_json(path):
    with open(path) as f:
        return json.load(f)


def write_json(path, obj):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    os.replace(tmp, path)


# --------------------------------------------------------------------------- answers
def last_boxed(text: str):
    """Content of the last complete \\boxed{...} (or \\fbox{...}), or None."""
    if not text:
        return None
    best = None
    for macro in ("\\boxed", "\\fbox"):
        i = text.rfind(macro + "{")
        if i < 0:
            continue
        j, depth = i + len(macro) + 1, 1
        while j < len(text) and depth:
            depth += {"{": 1, "}": -1}.get(text[j], 0)
            j += 1
        if depth == 0 and (best is None or i > best[0]):
            best = (i, text[i + len(macro) + 1:j - 1])
    return best[1] if best else None
