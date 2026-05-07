#!/usr/bin/env python3
"""
Stress test 2 — Sequence length extrapolation (error accumulation).

Protocol:
  * Models are trained with some max context (e.g. L_train = 512 in your runs);
    this script does not train — it loads finished checkpoints.
  * Build evaluation sequences of length L_eval by streaming tokens from
    chat-formatted JSONL rows (same template as training eval).
  * Zero-shot style: for each L_eval in a user-provided list, measure **only
    the final-token prediction**: NLL, PPL = exp(NLL), and greedy accuracy.

Compare NTP vs STP vs control_JEPA by passing any subset of:
  --ckpt_ntp --ckpt_stp --ckpt_control

Typical use:
  python stress_test_length_extrapolation.py \\
    --ckpt_ntp path/to/regular \\
    --ckpt_stp path/to/stp \\
    --ckpt_control path/to/control_jepa \\
    --eval_jsonl datasets/synth_test.jsonl \\
    --lengths 512 1024 2048 4096 \\
    --out_json stress_length_extrapolation.json \\
    --out_plot stress_length_extrapolation.png

If the concatenated corpus is shorter than max(lengths), the script **tiles**
the stream (repeats the same token ids) so there is no ambiguous PAD tail.

Requires: torch, transformers, matplotlib, numpy.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_jsonl_messages(path: Path, max_lines: int) -> List[dict]:
    rows: List[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "messages" in obj:
                rows.append(obj)
            if len(rows) >= max_lines:
                break
    return rows


def stream_token_ids(
    tokenizer,
    examples: List[dict],
    target_len: int,
) -> Tuple[torch.Tensor, bool]:
    """
    Concatenate token ids from successive chat-formatted examples until
    ``target_len`` is reached. If the corpus is still short, **tile** the
    accumulated stream (no PAD tokens) so every index is a real token id —
    this keeps the last-token CE well-defined.
    Returns ([1, target_len], tiled_flag).
    """
    ids: List[int] = []
    for ex in examples:
        text = tokenizer.apply_chat_template(
            ex["messages"], tokenize=False, add_generation_prompt=False
        )
        piece = tokenizer.encode(text, add_special_tokens=False)
        ids.extend(piece)
        if len(ids) >= target_len:
            ids = ids[:target_len]
            return torch.tensor([ids], dtype=torch.long), False
    if not ids:
        raise RuntimeError("No tokens collected from eval JSONL (empty messages?).")
    if len(ids) < target_len:
        base = ids[:]
        while len(ids) < target_len:
            ids.extend(base)
        ids = ids[:target_len]
        return torch.tensor([ids], dtype=torch.long), True
    return torch.tensor([ids[:target_len]], dtype=torch.long), False


def last_token_metrics(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    last_idx: int,
) -> Tuple[float, float, float]:
    """
    Supervise only the prediction of token at position ``last_idx`` (0-based).
    Returns (nll_natural, ppl, acc01) for that single position.
    """
    device = next(model.parameters()).device
    b, L = input_ids.shape
    assert last_idx == L - 1
    labels = torch.full_like(input_ids, -100)
    # logits[i] predicts token i+1; so token at L-1 is predicted from logits[L-2]
    labels[:, L - 2] = input_ids[:, L - 1]

    out = model(
        input_ids=input_ids.to(device),
        attention_mask=attention_mask.to(device),
        labels=labels.to(device),
    )
    logits = out.logits.float()  # [B, L, V]
    # manual CE at position L-2
    logit = logits[:, L - 2, :]
    target = input_ids[:, L - 1].to(device)
    nll = F.cross_entropy(logit, target, reduction="mean").item()
    pred = logit.argmax(dim=-1)
    acc = float((pred == target).float().mean().item())
    return nll, math.exp(nll), acc


@torch.no_grad()
def sweep_lengths_for_ckpt(
    ckpt_dir: Path,
    tokenizer_base: Optional[str],
    examples: List[dict],
    lengths: List[int],
    device: torch.device,
    dtype: torch.dtype,
) -> Dict[int, Dict[str, float]]:
    tok_path = str(ckpt_dir) if tokenizer_base is None else tokenizer_base
    tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(ckpt_dir),
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    model.eval()
    model.to(device)

    results: Dict[int, Dict[str, float]] = {}
    for L in lengths:
        ids, tiled = stream_token_ids(tokenizer, examples, L)
        am = torch.ones_like(ids)
        nll, ppl, acc = last_token_metrics(model, ids, am, L - 1)
        results[int(L)] = {
            "last_token_nll": nll,
            "last_token_ppl": ppl,
            "last_token_acc": acc,
            "tiled_stream_to_length": float(tiled),
        }
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return results


def plot_results(
    all_ckpts: Dict[str, Dict[int, Dict[str, float]]],
    lengths: List[int],
    out_path: Path,
    metric: str,
) -> None:
    plt.figure(figsize=(8, 4.5))
    for label, by_len in all_ckpts.items():
        ys = [by_len[L][metric] for L in lengths]
        plt.plot(lengths, ys, marker="o", linewidth=2, label=label)
    plt.xlabel(r"Evaluation context length $L_{eval}$ (tokens)")
    key = metric
    if metric == "last_token_ppl":
        plt.ylabel("Last-token PPL")
        plt.yscale("log")
    elif metric == "last_token_nll":
        plt.ylabel("Last-token NLL (nats)")
    else:
        plt.ylabel("Last-token greedy accuracy")
        plt.ylim(-0.05, 1.05)
    plt.title(f"Length extrapolation ({key})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main() -> None:
    p = argparse.ArgumentParser(description="Length-extrapolation stress test (last-token metrics).")
    p.add_argument("--ckpt_ntp", type=str, default="", help="Regular / NTP checkpoint dir.")
    p.add_argument("--ckpt_stp", type=str, default="", help="STP (random_span) checkpoint dir.")
    p.add_argument("--ckpt_control", type=str, default="", help="control_JEPA checkpoint dir.")
    p.add_argument(
        "--tokenizer_path",
        type=str,
        default="",
        help="Optional tokenizer source (default: each checkpoint dir).",
    )
    p.add_argument("--eval_jsonl", type=str, default="datasets/synth_test.jsonl")
    p.add_argument(
        "--stream_max_lines",
        type=int,
        default=10_000,
        help="Max JSONL rows to read when concatenating tokens for long contexts.",
    )
    p.add_argument(
        "--lengths",
        type=int,
        nargs="+",
        default=[512, 1024, 2048, 4096],
        help="Evaluation lengths L_eval (must be >= 2 for last-token loss).",
    )
    p.add_argument("--L_train_note", type=int, default=512, help="Recorded in JSON only (your train max_length).")
    p.add_argument("--out_json", type=str, default="stress_length_extrapolation.json")
    p.add_argument("--out_plot", type=str, default="stress_length_extrapolation.png")
    p.add_argument(
        "--plot_metric",
        type=str,
        default="last_token_ppl",
        choices=("last_token_ppl", "last_token_nll", "last_token_acc"),
    )
    args = p.parse_args()

    ckpts: List[Tuple[str, str]] = []
    if args.ckpt_ntp.strip():
        ckpts.append(("NTP", args.ckpt_ntp.strip()))
    if args.ckpt_stp.strip():
        ckpts.append(("STP", args.ckpt_stp.strip()))
    if args.ckpt_control.strip():
        ckpts.append(("control_JEPA", args.ckpt_control.strip()))
    if not ckpts:
        raise SystemExit("Provide at least one of --ckpt_ntp / --ckpt_stp / --ckpt_control")

    lengths = sorted(set(args.lengths))
    if min(lengths) < 2:
        raise SystemExit("Each length must be >= 2 (need a predecessor position for last-token CE).")

    eval_path = Path(args.eval_jsonl)
    examples = load_jsonl_messages(eval_path, args.stream_max_lines)
    if not examples:
        raise SystemExit(f"No messages in {eval_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    tok_override = args.tokenizer_path.strip() or None

    all_results: Dict[str, Dict[int, Dict[str, float]]] = {}
    for label, path in ckpts:
        print(f"=== {label} :: {path} ===", flush=True)
        all_results[label] = sweep_lengths_for_ckpt(
            Path(path),
            tok_override,
            examples,
            lengths,
            device,
            dtype,
        )
        for L in lengths:
            row = all_results[label][L]
            print(
                f"  L={L:5d}  PPL={row['last_token_ppl']:.4f}  "
                f"acc={row['last_token_acc']:.4f}  tiled={bool(row['tiled_stream_to_length'])}",
                flush=True,
            )

    out_json = Path(args.out_json)
    serial = {
        "L_train_note": args.L_train_note,
        "eval_jsonl": str(eval_path),
        "lengths": lengths,
        "plot_metric": args.plot_metric,
        "results": {
            label: {str(L): v for L, v in by_l.items()} for label, by_l in all_results.items()
        },
    }
    out_json.write_text(json.dumps(serial, indent=2))
    plot_results(all_results, lengths, Path(args.out_plot), args.plot_metric)
    print(f"Wrote {out_json} and {args.out_plot}", flush=True)


if __name__ == "__main__":
    main()
