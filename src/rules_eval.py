#!/usr/bin/env python
"""Experiment 3 evaluation (GPU, vLLM): sample a checkpoint's answers to the hidden-rule items and parse them.

    python src/rules_eval.py --run <dir> --items <items.jsonl> --splits new_numbers,new_range,composition,with_def --samples 16
    python src/rules_eval.py --run <dir> --items ... --adapter runs/<arm>/adapter_final      # a trained arm

T = 1.0, no top-p or top-k, thinking off, at most --max-tokens response tokens; the with_def split states the
definition in the prompt. A LoRA adapter is served by vLLM as an adapter (--adapter-mode lora, the default: the
fp32 LoRA weights are only rounded to bf16 themselves) or merged into the base weights on the CPU in bf16 and
written next to the run, then deleted unless --keep-merged (--adapter-mode merge; after a short training run the
update is small against the base weights' bf16 spacing, so merging rounds much of it away: 87% of its norm after
the smoke's two steps). Writes rollouts.jsonl (item, sample, text, parsed integer answer, length, truncation,
repetition) and rollout.meta.json; skips if the meta file exists.
"""
import argparse
import gc
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import rules_data as rd  # noqa: E402
from passk import repeated_share, severe_repetition  # noqa: E402


def merge_adapter(model_name, adapter, out):
    """Base + LoRA -> merged bf16 weights and tokenizer in `out` (on the CPU, before vLLM takes the GPU)."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out = Path(out)
    if (out / "config.json").exists():
        return out
    base = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16)
    merged = PeftModel.from_pretrained(base, adapter).merge_and_unload()
    merged.save_pretrained(out, safe_serialization=True)
    AutoTokenizer.from_pretrained(model_name).save_pretrained(out)
    del merged, base
    gc.collect()
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--items", required=True)
    ap.add_argument("--model", default=common.STUDENT)
    ap.add_argument("--adapter", default=None, help="LoRA adapter directory")
    ap.add_argument("--adapter-mode", choices=("lora", "merge"), default="lora")
    ap.add_argument("--splits", default=",".join(rd.SPLITS))
    ap.add_argument("--samples", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--keep-ids", action="store_true", help="store response token ids (the teacher audit needs them)")
    ap.add_argument("--keep-merged", action="store_true")
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    args = ap.parse_args(argv)

    run = Path(args.run)
    run.mkdir(parents=True, exist_ok=True)
    if (run / "rollout.meta.json").exists():
        print(f"evaluation already done: {run / 'rollout.meta.json'}")
        return
    common.ensure_c_compiler()
    t0 = time.time()
    splits = [s for s in args.splits.split(",") if s]
    items = [it for it in common.read_jsonl(args.items) if it["split"] in splits]
    if not items:
        sys.exit(f"no items in splits {splits}")
    merge = bool(args.adapter) and args.adapter_mode == "merge"
    model_path = str(merge_adapter(args.model, args.adapter, run / "merged")) if merge else args.model
    merge_s = time.time() - t0
    lora_kw, lora_req = {}, None
    if args.adapter and not merge:
        from vllm.lora.request import LoRARequest
        rank = int(common.read_json(Path(args.adapter) / "adapter_config.json")["r"])
        lora_kw = dict(enable_lora=True, max_lora_rank=max(8, rank), max_loras=1)
        lora_req = LoRARequest("adapter", 1, str(Path(args.adapter).resolve()))

    import torch
    import vllm
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(args.model)
    end_ids = common.end_token_ids(tok)
    for it in items:
        it["prompt_ids"] = rd.prompt_ids_for(tok, it, with_def=it["split"] == "with_def")
    max_prompt = max(len(it["prompt_ids"]) for it in items)
    dtype = "bfloat16" if torch.cuda.get_device_capability()[0] >= 8 else "float16"
    llm = LLM(model=model_path, tokenizer=args.model, dtype=dtype, seed=args.seed, max_model_len=max_prompt + args.max_tokens + 8,
              gpu_memory_utilization=args.gpu_mem, enable_prefix_caching=True, generation_config="vllm", **lora_kw)
    sps = [SamplingParams(n=args.samples, temperature=1.0, top_p=1.0, max_tokens=args.max_tokens, stop_token_ids=end_ids,
                          seed=common.stable_seed("rules", args.seed, it["item_id"])) for it in items]
    prompts = [{"prompt_token_ids": it["prompt_ids"]} for it in items]
    outs = llm.generate(prompts, sps, lora_request=lora_req) if lora_req else llm.generate(prompts, sps)
    gen_s = time.time() - t0 - merge_s

    rows = []
    for it, out in zip(items, outs):
        for j, c in enumerate(out.outputs):
            ids = list(c.token_ids)
            text = tok.decode(ids, skip_special_tokens=True)
            got = rd.answer_of(text)
            row = dict(item_id=it["item_id"], sym=it["sym"], split=it["split"], kind=it["kind"], operands=it["operands"],
                       answer=it["answer"], habit=it["habit"], sample_idx=j, got=got, correct=got == it["answer"],
                       n_tokens=len(ids), truncated=c.finish_reason == "length",
                       repeated=round(repeated_share(ids), 3), severe_rep=severe_repetition(ids), text=text)
            if args.keep_ids:
                row["response_ids"] = ids
            rows.append(row)
    common.write_jsonl_atomic(run / "rollouts.jsonl", rows)
    by_split = {}
    for r in rows:
        by_split.setdefault(r["split"], []).append(r)
    meta = dict(stage="rules_eval", model=args.model, adapter=args.adapter, adapter_mode=args.adapter_mode if args.adapter else None,
                splits=splits, samples=args.samples,
                max_tokens=args.max_tokens, temperature=1.0, top_p=1.0, top_k=None, thinking=False, seed=args.seed,
                n_items=len(items), n_responses=len(rows), vllm=vllm.__version__, dtype=dtype,
                correct={s: round(sum(r["correct"] for r in v) / len(v), 4) for s, v in by_split.items()},
                truncated=round(sum(r["truncated"] for r in rows) / len(rows), 4),
                no_answer=round(sum(r["got"] is None for r in rows) / len(rows), 4),
                mean_tokens=round(sum(r["n_tokens"] for r in rows) / len(rows), 1),
                merge_seconds=round(merge_s), gen_seconds=round(gen_s), seconds=round(time.time() - t0),
                example_prompt=rd.prompt_text_for(tok, items[0]), example_prompt_with_def=rd.prompt_text_for(tok, items[0], with_def=True))
    common.write_json(run / "rollout.meta.json", meta)
    if merge and not args.keep_merged:
        shutil.rmtree(run / "merged", ignore_errors=True)
    print(f"{len(rows)} responses; correct {meta['correct']}; truncated {meta['truncated']:.1%}; no answer {meta['no_answer']:.1%}; "
          f"mean {meta['mean_tokens']:.0f} tokens; {meta['seconds']}s")


if __name__ == "__main__":
    main()
