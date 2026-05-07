#!/usr/bin/env python3
"""
Load a pretrained checkpoint and plot V_t at every token position
for sample sequences from the SYNTH dataset.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.tmin_experiment import compute_V_all_vgeo, find_start_end


def load_examples(jsonl_path: Path, max_examples: int = 10) -> List[Dict]:
    rows = []
    with jsonl_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if len(rows) >= max_examples:
                break
    return rows


@torch.no_grad()
def get_vt_for_sample(
    model: torch.nn.Module,
    tokenizer,
    messages: List[Dict],
    max_length: int = 512,
    layer: int = -1,
) -> Optional[Dict]:
    """Compute V_t at every position for one sample."""
    full_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    tok = tokenizer(
        full_text, truncation=True, max_length=max_length,
        padding=False, return_tensors=None, add_special_tokens=True,
    )
    input_ids = tok["input_ids"]
    attention_mask = [1] * len(input_ids)
    seq_len = len(input_ids)

    # Find user and assistant spans
    user_msg = next((m for m in messages if m["role"] == "user"), None)
    asst_msg = next((m for m in messages if m["role"] == "assistant"), None)
    if not user_msg or not asst_msg:
        return None

    try:
        u0, ue = find_start_end(user_msg["content"], tokenizer, input_ids, attention_mask)
        a0, ae = find_start_end(asst_msg["content"], tokenizer, input_ids, attention_mask)
    except Exception:
        return None

    u_s, u_e = u0 + 1, ue
    a_s, a_e = a0 + 1, ae

    if not (0 <= u_s < u_e < seq_len and 0 <= a_s < a_e < seq_len):
        return None
    if a_e <= a_s + 1:
        return None

    device = next(model.parameters()).device
    iid = torch.tensor([input_ids], dtype=torch.long, device=device)
    am_t = torch.tensor([attention_mask], dtype=torch.long, device=device)
    out = model(input_ids=iid, attention_mask=am_t, output_hidden_states=True, use_cache=False)
    h = out.hidden_states[layer][0].float()

    V_all = compute_V_all_vgeo(h, u_s, u_e, a_s, a_e)
    V_np = V_all.cpu().numpy()

    # Decode tokens for labeling
    tokens_decoded = [tokenizer.decode([t]) for t in input_ids]

    return {
        "V_all": V_np,
        "u_s": u_s,
        "u_e": u_e,
        "a_s": a_s,
        "a_e": a_e,
        "seq_len": seq_len,
        "tokens": tokens_decoded,
        "user_text": user_msg["content"][:80],
        "asst_text": asst_msg["content"][:80],
    }


def plot_vt_samples(
    results: List[Dict],
    gamma: float,
    tau: float,
    out_path: Path,
) -> None:
    """Plot V_t for each sample, showing user/assistant span regions."""
    n = len(results)
    cols = 2
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(14, 4 * rows), dpi=120)
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes[np.newaxis, :]
    elif cols == 1:
        axes = axes[:, np.newaxis]

    v_star = tau / (1.0 - gamma)

    for idx, res in enumerate(results):
        r, c = divmod(idx, cols)
        ax = axes[r, c]

        V = res["V_all"]
        positions = np.arange(len(V))
        u_s, u_e = res["u_s"], res["u_e"]
        a_s, a_e = res["a_s"], res["a_e"]

        # Plot V_t
        ax.plot(positions, V, color="#1f77b4", lw=1.5, alpha=0.9)

        # Shade regions
        ax.axvspan(u_s, u_e, alpha=0.1, color="blue", label="User span")
        ax.axvspan(a_s, a_e, alpha=0.1, color="green", label="Assistant span")

        # Mark key points
        ax.axvline(u_s, color="blue", ls="--", lw=0.8, alpha=0.5)
        ax.axvline(u_e, color="blue", ls="--", lw=0.8, alpha=0.5)
        ax.axvline(a_s, color="green", ls="--", lw=0.8, alpha=0.5)
        ax.axvline(a_e, color="green", ls="--", lw=0.8, alpha=0.5)

        # Theoretical lines
        ax.axhline(v_star, color="red", ls="--", lw=1.5, alpha=0.7,
                   label=f"$V^* = {v_star:.6f}$")

        # V_ts at assistant start
        v_ts = V[a_s]
        ax.plot(a_s, v_ts, "ro", markersize=6, zorder=5)
        ax.annotate(f"V_ts={v_ts:.4f}", xy=(a_s, v_ts),
                    xytext=(a_s + 3, v_ts + 0.005), fontsize=7, color="red")

        ax.set_xlabel("Token position")
        ax.set_ylabel("$V_t$")
        ax.set_title(f"Sample {idx+1}: \"{res['user_text'][:50]}...\"", fontsize=9)
        if idx == 0:
            ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.2)
        ax.set_xlim(0, res["seq_len"])

    # Hide unused axes
    for idx in range(n, rows * cols):
        r, c = divmod(idx, cols)
        axes[r, c].set_visible(False)

    fig.suptitle(
        f"V_t per token — γ={gamma}, τ={tau}, V*={v_star:.6f}\n"
        f"Checkpoint: ft-c-v_geo-synth-g0.95-t1e-4",
        fontsize=12, y=1.01,
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    checkpoint = "test/llm-jepa/ft-c-v_geo-synth-g0.95-t1e-4-2e-5-0.05-0-84"
    eval_jsonl = "test/llm-jepa/datasets/synth_test.jsonl"
    gamma = 0.95
    tau = 1e-4
    n_samples = 10
    max_length = 512
    layer = -1
    out_path = Path("vt_per_token_synth.png")

    print(f"Loading model: {checkpoint}")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_cuda = torch.cuda.is_available()
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        torch_dtype=torch.bfloat16 if use_cuda else torch.float32,
        device_map="auto" if use_cuda else None,
        trust_remote_code=True,
    )
    model.eval()
    print(f"Model loaded on {'CUDA' if use_cuda else 'CPU'}")

    print(f"\nLoading {n_samples} samples from: {eval_jsonl}")
    examples = load_examples(Path(eval_jsonl), max_examples=n_samples * 2)

    print("Computing V_t for each sample...")
    results = []
    for ex in examples:
        if len(results) >= n_samples:
            break
        res = get_vt_for_sample(model, tokenizer, ex["messages"],
                                max_length=max_length, layer=layer)
        if res is not None:
            results.append(res)
            print(f"  [{len(results)}/{n_samples}] seq_len={res['seq_len']}, "
                  f"user=[{res['u_s']}:{res['u_e']}], asst=[{res['a_s']}:{res['a_e']}], "
                  f"V_ts={res['V_all'][res['a_s']]:.4f}, V_final={res['V_all'][res['a_e']-1]:.4f}")

    if not results:
        raise SystemExit("No valid sequences found.")

    print(f"\nPlotting {len(results)} samples...")
    plot_vt_samples(results, gamma, tau, out_path)


if __name__ == "__main__":
    main()
