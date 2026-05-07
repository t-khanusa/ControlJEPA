#!/usr/bin/env python3
"""Compare SYNTH generation length / stopping between Llama STP (ft-j) and control_v_geo (ft-c-v_geo).

Loads each checkpoint separately (to limit VRAM), replays the same prompts as evaluate.py
(system+user + generation prompt), greedy-decodes, then reports:

  * new-token count distribution (mean/median/p95/max)
  * fraction that hit --max_new_tokens (length-capped, not early-stopped)
  * whether generated ids contain tokenizer.eos_token_id
  * P(eos_at_oracle) := softmax(logits after teacher-forced *gold answer*) on <|eot_id|> (or tokenizer.eos_token_id)

The last quantity isolates: "after emitting the reference regex string, how likely is immediate end-of-turn?"
If v_geo systematically assigns lower mass to the turn-end token here, that explains longer continuations.

Paired per-example token deltas are printed at the end (v_geo - stp).

Usage (from repo root):
  conda run -n controlJEPA python scripts/diagnose_llama_synth_gen_length.py \\
    --stp_ckpt Llama3.2-1B-Instruct/ft-j-synth-2e-5-0.02-0-82 \\
    --vgeo_ckpt Llama3.2-1B-Instruct/ft-c-v_geo-synth-g0.80-t1e-3-2e-5-0.01-0-82 \\
    --original_model_name meta-llama/Llama-3.2-1B-Instruct \\
    --input_file datasets/synth_test.jsonl --max_examples 256
"""
from __future__ import annotations

import argparse
import json
import gc
import statistics
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from evaluate import format_conversation, get_messages, load_model_and_tokenizer  # noqa: E402


def _read_jsonl(path: Path, limit: int | None):
    rows = []
    with path.open() as f:
        for line in f:
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def _pctl(xs: list[int], q: float) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    idx = min(len(s) - 1, max(0, int(round((len(s) - 1) * q))))
    return float(s[idx])


@torch.inference_mode()
def run_one_model(
    *,
    model_path: str,
    original: str,
    examples: list,
    max_new_tokens: int,
    max_length: int,
    device_hint: str,
):
    model, tokenizer = load_model_and_tokenizer(
        model_path, original, device_map=device_hint
    )
    model.eval()

    dev = next(model.parameters()).device

    eot_id = tokenizer.eos_token_id
    if eot_id is None:
        raise RuntimeError("tokenizer.eos_token_id is None")

    new_lens = []
    hit_cap = []
    gen_has_eot = []
    oracle_peot = []
    decoded_lens = []
    strict_ok = []

    for ex in examples:
        messages = ex["messages"]
        conv = get_messages(original, messages)
        prompt = format_conversation(conv, tokenizer, plain=False)
        enc = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            add_special_tokens=True,
        )
        input_ids = enc["input_ids"].to(dev)
        attn = enc["attention_mask"].to(dev)
        prompt_len = int(input_ids.shape[1])

        out = model.generate(
            input_ids=input_ids,
            attention_mask=attn,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=eot_id,
            do_sample=False,
            max_new_tokens=max_new_tokens,
        )
        gen_ids = out[0, prompt_len:].tolist()
        n_new = len(gen_ids)
        new_lens.append(n_new)
        hit_cap.append(1 if n_new >= max_new_tokens else 0)
        gen_has_eot.append(1 if eot_id in gen_ids else 0)

        response = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        decoded_lens.append(len(response))
        gt = (messages[2]["content"] or "").strip()
        strict_ok.append(1 if response == gt else 0)

        gold = messages[2]["content"]
        gold_ids = tokenizer.encode(gold, add_special_tokens=False)
        if not gold_ids:
            oracle_peot.append(float("nan"))
            continue
        oracle = torch.cat(
            [input_ids, torch.tensor([gold_ids], device=dev, dtype=torch.long)],
            dim=1,
        )
        o_am = torch.ones_like(oracle, dtype=torch.long)
        logits = model(input_ids=oracle, attention_mask=o_am).logits
        last = logits[0, -1].float()
        probs = F.softmax(last, dim=-1)
        oracle_peot.append(float(probs[eot_id].item()))

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    def _summ(name, xs: list[float]):
        xs_clean = [x for x in xs if x == x]  # drop nan
        if not xs_clean:
            return f"{name}: n/a"
        return (
            f"{name}: mean={statistics.mean(xs_clean):.4f} "
            f"median={statistics.median(xs_clean):.4f}"
        )

    return {
        "path": model_path,
        "n": len(examples),
        "eot_id": int(eot_id),
        "new_lens": list(new_lens),
        "decoded_lens": list(decoded_lens),
        "strict_ok": list(strict_ok),
        "new_len_mean": statistics.mean(new_lens),
        "new_len_median": statistics.median(new_lens),
        "new_len_p95": _pctl(new_lens, 0.95),
        "new_len_max": max(new_lens),
        "dec_len_mean": statistics.mean(decoded_lens),
        "frac_hit_cap": sum(hit_cap) / len(hit_cap),
        "frac_gen_contains_eot_token": sum(gen_has_eot) / len(gen_has_eot),
        "frac_strict_em": sum(strict_ok) / len(strict_ok),
        "oracle_peot_mean": statistics.mean([x for x in oracle_peot if x == x]),
        "oracle_peot_median": statistics.median([x for x in oracle_peot if x == x]),
        "lines": (
            f"path={model_path}  eot_id={eot_id}\n"
            f"  new_tokens: mean={statistics.mean(new_lens):.2f}  "
            f"median={statistics.median(new_lens):.1f}  "
            f"p95={_pctl(new_lens, 0.95):.1f}  "
            f"max={max(new_lens)}  (budget={max_new_tokens})\n"
            f"  decoded_chars (skip_special_tokens): mean={statistics.mean(decoded_lens):.2f}\n"
            f"  strict_EM (greedy decode vs gold): {sum(strict_ok)/len(strict_ok):.4f}\n"
            f"  frac_hit_max_new_tokens={sum(hit_cap)/len(hit_cap):.4f}\n"
            f"  frac_generation_ids_contain_eot={sum(gen_has_eot)/len(gen_has_eot):.4f}\n"
            f"  {_summ('oracle P(eos) after teacher-forced gold', oracle_peot)}"
        ),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stp_ckpt", type=str, required=True)
    p.add_argument("--vgeo_ckpt", type=str, required=True)
    p.add_argument("--original_model_name", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    p.add_argument("--input_file", type=str, default="datasets/synth_test.jsonl")
    p.add_argument("--max_examples", type=int, default=256)
    p.add_argument("--max_new_tokens", type=int, default=96)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--device_map", type=str, default="auto")
    args = p.parse_args()

    examples = _read_jsonl(Path(args.input_file), args.max_examples)
    print(f"Loaded {len(examples)} examples from {args.input_file}")

    a = run_one_model(
        model_path=args.stp_ckpt,
        original=args.original_model_name,
        examples=examples,
        max_new_tokens=args.max_new_tokens,
        max_length=args.max_length,
        device_hint=args.device_map,
    )
    print("\n=== STP (ft-j) ===\n" + a["lines"])

    b = run_one_model(
        model_path=args.vgeo_ckpt,
        original=args.original_model_name,
        examples=examples,
        max_new_tokens=args.max_new_tokens,
        max_length=args.max_length,
        device_hint=args.device_map,
    )
    print("\n=== control_v_geo (ft-c-v_geo) ===\n" + b["lines"])

    print("\n=== Delta (v_geo - stp) ===")
    print(f"  mean_new_tokens: {b['new_len_mean'] - a['new_len_mean']:+.3f}")
    print(f"  frac_hit_cap: {b['frac_hit_cap'] - a['frac_hit_cap']:+.4f}")
    print(
        "  oracle P(eos): "
        f"{b['oracle_peot_mean'] - a['oracle_peot_mean']:+.4f} "
        f"(v_geo_mean={b['oracle_peot_mean']:.4f} vs stp_mean={a['oracle_peot_mean']:.4f})"
    )

    paired = [vb - va for va, vb in zip(a["new_lens"], b["new_lens"])]
    print("\n=== Paired (v_geo - stp) new_token counts ===")
    print(f"  mean_delta: {statistics.mean(paired):+.4f}")
    print(f"  frac(v_geo strictly longer): {sum(1 for d in paired if d > 0) / len(paired):.4f}")
    print(f"  frac(same length): {sum(1 for d in paired if d == 0) / len(paired):.4f}")

    mism = [(i, a["new_lens"][i], b["new_lens"][i]) for i in range(len(paired)) if paired[i] != 0]
    if mism:
        print(f"  examples with != length: {len(mism)} (show up to 8 idx: stp_len, vgeo_len)")
        for t in mism[:8]:
            print(f"    ex {t[0]}: stp_new={t[1]} vgeo_new={t[2]} delta={t[2]-t[1]}")


if __name__ == "__main__":
    main()
