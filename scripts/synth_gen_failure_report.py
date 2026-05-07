#!/usr/bin/env python3
"""Generate on SYNTH (JSONL ``messages``) with a checkpoint (e.g. ft-c-v_geo) and analyze failures.

Writes every row + metrics to JSONL; prints strict/relaxed rates and failure **buckets**.

Strict-failure buckets (first match wins in ``_classify_strict_failure``):

  * ``hit_token_budget`` — new-token count reached ``max_new_tokens`` before natural stop
  * ``empty_generation`` — stripped model output empty
  * ``relaxed_only_ok`` — engineering-relaxed match, strict EM fails (delimiter/format bleed)
  * ``starts_with_gt_alnum_tail`` — gold prefix + extra alphanumeric continuation
  * ``wrong_answer`` — incorrect regex vs gold

Example:
  conda run -n controlJEPA python scripts/synth_gen_failure_report.py \\
    --model_path Llama3.2-1B-Instruct/ft-c-v_geo-synth-g0.80-t1e-3-2e-5-0.01-0-82 \\
    --original_model_name meta-llama/Llama-3.2-1B-Instruct \\
    --input_file datasets/synth_test.jsonl \\
    --out_jsonl results/synth_vgeo_failures.jsonl \\
    --max_examples 500 --device_map cuda:0
"""
from __future__ import annotations

import argparse
import json
import gc
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from evaluate import format_conversation, get_messages, load_model_and_tokenizer  # noqa: E402
from synth_metrics import synth_engineering_relaxed_match, synth_strict_match  # noqa: E402


def _read_jsonl(path: Path, limit: int | None) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def _classify_strict_failure(
    gen: str, gt: str, new_tok_len: int, max_new_tokens: int
) -> str:
    g = (gen or "").strip()
    gt_s = (gt or "").strip()
    if synth_strict_match(gen, gt):
        return "ok_strict"
    if new_tok_len >= max_new_tokens:
        return "hit_token_budget"
    if not g:
        return "empty_generation"
    if synth_engineering_relaxed_match(gen, gt) and not synth_strict_match(gen, gt):
        return "relaxed_only_ok"
    if g.startswith(gt_s) and len(g) > len(gt_s):
        rest = g[len(gt_s):]
        if rest and rest[0].isalnum():
            return "starts_with_gt_alnum_tail"
    return "wrong_answer"


def _write_failure_preview(jsonl_path: Path, out_txt: Path, n_prev: int) -> None:
    lines_out: list[str] = []
    with jsonl_path.open() as f:
        for raw in f:
            r = json.loads(raw)
            if r.get("strict_ok"):
                continue
            lines_out.append(
                f"--- #{r['index']} tag={r['failure_tag']} new_toks={r['new_token_count']} ---\n"
                f"USER: {str(r['user'])[:280]}\n"
                f"GOLD: {r['gold']}\n"
                f"PRED: {r['prediction']}\n"
            )
            if len(lines_out) >= n_prev:
                break
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_txt.write_text("\n".join(lines_out))


@torch.inference_mode()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=str, required=True)
    ap.add_argument("--original_model_name", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    ap.add_argument("--input_file", type=str, default="datasets/synth_test.jsonl")
    ap.add_argument("--out_jsonl", type=str, default="results/synth_gen_failure_report.jsonl")
    ap.add_argument("--max_examples", type=int, default=None)
    ap.add_argument("--max_new_tokens", type=int, default=96)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--device_map", type=str, default="auto")
    args = ap.parse_args()

    rows = _read_jsonl(Path(args.input_file), args.max_examples)
    print(f"Loaded {len(rows)} examples from {args.input_file}")

    model, tokenizer = load_model_and_tokenizer(
        args.model_path, args.original_model_name, device_map=args.device_map
    )
    model.eval()
    dev = next(model.parameters()).device
    eot_id = tokenizer.eos_token_id

    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    buckets_fail = Counter()
    strict_ok = 0
    relaxed_ok = 0

    with out_path.open("w") as out_f:
        for idx, ex in enumerate(rows):
            messages = ex["messages"]
            user_text = messages[1]["content"]
            gt = messages[2]["content"]

            conv = get_messages(args.original_model_name, messages)
            prompt = format_conversation(conv, tokenizer, plain=False)
            enc = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=args.max_length,
                add_special_tokens=True,
            )
            input_ids = enc["input_ids"].to(dev)
            attn = enc["attention_mask"].to(dev)
            plen = int(input_ids.shape[1])

            out = model.generate(
                input_ids=input_ids,
                attention_mask=attn,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=eot_id,
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
            )
            gen_ids = out[0, plen:].tolist()
            n_new = len(gen_ids)

            response = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            s_ok = synth_strict_match(response, gt)
            r_ok = synth_engineering_relaxed_match(response, gt)
            if s_ok:
                strict_ok += 1
            if r_ok:
                relaxed_ok += 1

            tag = _classify_strict_failure(response, gt, n_new, args.max_new_tokens)
            if not s_ok:
                buckets_fail[tag] += 1

            record: dict[str, Any] = {
                "index": idx,
                "strict_ok": s_ok,
                "relaxed_ok": r_ok,
                "failure_tag": "none" if s_ok else tag,
                "new_token_count": n_new,
                "max_new_tokens": args.max_new_tokens,
                "contains_eot_in_gen": bool(eot_id in gen_ids) if eot_id is not None else None,
                "user": user_text[:500],
                "gold": gt,
                "prediction": response,
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

            if (idx + 1) % 100 == 0:
                print(f"  processed {idx + 1}/{len(rows)}")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    n = len(rows)
    total_fail = n - strict_ok
    print(f"\n=== Summary ({args.model_path}) ===")
    print(f"examples: {n}")
    print(f"strict_EM:  {strict_ok}/{n}  ({strict_ok/n:.4f})")
    print(f"relaxed_engineering: {relaxed_ok}/{n} ({relaxed_ok/n:.4f})")
    print(f"wrote JSONL: {out_path}")

    print("\n=== Strict failure taxonomy ===")
    for k in (
        "wrong_answer",
        "starts_with_gt_alnum_tail",
        "relaxed_only_ok",
        "hit_token_budget",
        "empty_generation",
        "ok_strict",
    ):
        c = buckets_fail.get(k, 0)
        if c:
            print(
                f"  {k}: {c}  ({c/n:.4f} of all examples, "
                f"{c/max(total_fail,1):.4f} of failures)"
            )

    fail_txt = out_path.with_suffix(out_path.suffix + ".failures_preview.txt")
    _write_failure_preview(out_path, fail_txt, n_prev=min(60, total_fail))
    print(f"wrote readable failure preview (up to 60 failures): {fail_txt}")


if __name__ == "__main__":
    main()
