"""
Post-hoc diagnostic: measure hidden-state geometry of trained checkpoints
to explain why different V_t normalizations succeed or fail.

For each checkpoint we run a forward pass over a fixed set of evaluation
prompts (default: first N examples of ``datasets/synth_test.jsonl``), locate
the user and assistant spans with the exact ``find_start_end`` logic used by
``stp.py``, pull the last-layer hidden states, and compute four geometric
quantities per valid token:

    h[t]                       raw hidden state
    d_t                        displacement from user anchor (with the chord
                               gap skipped — see LyapunovControlLoss)
    v_geo                      secant chord = (h[u_e] - h[u_s]) + (h[a_e] - h[a_s])
    p_t = <d_t, v_hat> v_hat   projection onto chord  (SIGNAL along v_geo)
    e_t = d_t - p_t            transversal residual    (NOISE off v_geo)

We then aggregate over valid tokens and examples to report

    mean_h2    = E ||h[t]||^2          hidden-state scale
    mean_d2    = E ||d_t||^2           displacement energy
    mean_vgeo2 = E ||v_geo||^2         chord length squared
    mean_p2    = E ||p_t||^2           signal (along-chord) energy
    mean_e2    = E ||e_t||^2           noise (transversal) energy
    snr        = mean_p2 / mean_e2     signal-to-noise ratio
    sin2_theta = E ||e_t||^2 / ||d_t||^2   angle form V_t (d_t-normed)
    v_over_h   = mean_vgeo2 / mean_h2      chord-to-state ratio

This is the right lens to diagnose ``d_model`` failure vs ``v_geo`` / ``d_t``
success:
  * If scale-gauge collapse is at work, ``mean_h2`` and ``mean_vgeo2`` will be
    depressed relative to the regular-NTP baseline (the reference), while the
    angle form ``sin2_theta`` may look fine — i.e. e_t goes down by shrinking,
    not by orienting.
  * If a variant is doing real geometric work, ``mean_h2`` and ``mean_vgeo2``
    stay on the NTP baseline and ``sin2_theta`` (or ``snr``) moves in the
    right direction (lower / higher) relative to NTP.

**Appendix H (Gaussian channel):** pooled ``SNR_geom = mean(||p||^2)/mean(||e||^2)``
can be substituted into Shannon's ``½·ln(1+SNR)`` only as an *engineering proxy*
matching the Semantic Tube orthogonal-noise narrative — see ``semantic_tube_snr.py``.

Outputs:
  * A JSON of per-model stats (``signal_noise_stats.json``)
  * A 2×3 grid of bar charts (``signal_noise_bars.png``) comparing the 5 models
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from semantic_tube_snr import accumulate_tube_projection_stats, appendix_h_linked_summary


def load_examples(jsonl_path: Path, n: int) -> List[Dict]:
    items = []
    with jsonl_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
            if len(items) >= n:
                break
    return items


def find_start_end(
    content: str,
    tokenizer,
    input_ids: List[int],
    attention_mask: List[int],
) -> Tuple[int, int]:
    """Port of stp.py::find_start_end (Llama / non-OpenELM path).

    Returns (start, end) where `start` is the index of the token *before* the
    first content token and `end` is the index of the last content token.
    This matches the convention used by ``RepresentationTrainer`` (the
    LyapunovControlLoss then does ``u_s = user_start_end[0] + 1``, so the
    actual span used by the loss is [start+1, end] inclusive).
    """
    tokens = tokenizer.encode(content, add_special_tokens=False)
    decoded_content = [tokenizer.decode(t) for t in tokens]
    decoded_input = [tokenizer.decode(t) for t in input_ids]
    for i in range(len(input_ids) - len(tokens), -1, -1):
        if (
            attention_mask[i] == 1
            and decoded_input[i : i + len(tokens)] == decoded_content
        ):
            assert i > 0, f"start index 0 is not safe; content={content!r}"
            return i - 1, i + len(tokens) - 1
    raise RuntimeError(f"Cannot locate content {content!r} in tokenized sequence")


@torch.no_grad()
def measure_one_model(
    ckpt_dir: Path,
    base_tokenizer_name: str,
    examples: List[Dict],
    max_length: int = 256,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
    layer: int = -1,
    eps: float = 1e-8,
) -> Dict[str, float]:
    """Run forward over all examples and return aggregate geometry stats."""
    tokenizer = AutoTokenizer.from_pretrained(base_tokenizer_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        str(ckpt_dir),
        torch_dtype=dtype,
        attn_implementation="eager",
    ).to(device).eval()

    partials = []
    count_examples = 0

    for ex in examples:
        msgs = ex["messages"]
        formatted = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=False
        )
        tok = tokenizer(
            formatted,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors=None,
            add_special_tokens=True,
        )
        input_ids = tok["input_ids"]
        attention_mask = tok["attention_mask"]

        try:
            user_start, user_end = find_start_end(
                msgs[1]["content"], tokenizer, input_ids, attention_mask
            )
            asst_start, asst_end = find_start_end(
                msgs[2]["content"], tokenizer, input_ids, attention_mask
            )
        except RuntimeError:
            continue

        u_s = user_start + 1
        u_e = user_end
        a_s = asst_start + 1
        a_e = asst_end
        T = len(input_ids)
        if not (0 <= u_s <= u_e < T and 0 <= a_s <= a_e < T):
            continue
        if u_e <= u_s and a_e <= a_s:
            # no valid transitions
            continue

        iid = torch.tensor([input_ids], device=device)
        amk = torch.tensor([attention_mask], device=device)
        out = model(
            input_ids=iid,
            attention_mask=amk,
            output_hidden_states=True,
            use_cache=False,
        )
        h = out.hidden_states[layer][0]  # (T, D)

        st = accumulate_tube_projection_stats(
            h, u_s=u_s, u_e=u_e, a_s=a_s, a_e=a_e, eps=eps
        )
        partials.append(st)
        count_examples += 1

    del model
    torch.cuda.empty_cache()

    if not partials:
        raise RuntimeError(f"No valid tokens found for {ckpt_dir}")

    sum_p2 = sum(p["sum_p2_signal"] for p in partials)
    sum_e2 = sum(p["sum_e2_noise"] for p in partials)
    sum_d2 = sum(p["sum_d2"] for p in partials)
    sum_h2 = sum(p["sum_h2"] for p in partials)
    sum_sin2 = sum(p["sum_sin2_theta"] for p in partials)
    sum_vgeo2 = sum(p["sum_vgeo2"] for p in partials)
    count_tokens = sum(p["n_tokens"] for p in partials)

    mean_h2 = sum_h2 / count_tokens
    mean_d2 = sum_d2 / count_tokens
    mean_p2 = sum_p2 / count_tokens
    mean_e2 = sum_e2 / count_tokens
    mean_sin2 = sum_sin2 / count_tokens
    mean_vgeo2 = sum_vgeo2 / max(count_examples, 1)

    appendix = appendix_h_linked_summary(
        mean_p2 / max(mean_e2, eps),
        vocab_size=getattr(tokenizer, "vocab_size", len(tokenizer)) or None,
        m_seen=count_tokens,
    )

    return {
        "mean_h2": mean_h2,
        "mean_d2": mean_d2,
        "mean_p2_signal": mean_p2,
        "mean_e2_noise": mean_e2,
        "snr_p_over_e": mean_p2 / max(mean_e2, eps),
        "mean_sin2_theta": mean_sin2,
        "mean_vgeo2": mean_vgeo2,
        "vgeo2_over_h2": mean_vgeo2 / max(mean_h2, eps),
        "n_tokens": count_tokens,
        "n_examples": count_examples,
        **{f"appx_{k}": v for k, v in appendix.items()},
    }


def plot_bars(stats: Dict[str, Dict[str, float]], save_path: Path) -> None:
    names = list(stats.keys())
    keys_and_titles = [
        ("mean_h2", "||h_t||²   (hidden-state scale)"),
        ("mean_vgeo2", "||v_geo||²   (chord length²)"),
        ("mean_d2", "||d_t||²   (displacement)"),
        ("mean_p2_signal", "||p_t||²   SIGNAL (along chord)"),
        ("mean_e2_noise", "||e_t||²   NOISE (transversal)"),
        ("mean_sin2_theta", "V_t = ||e_t||² / ||d_t||²   (angle form)"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.0))
    colors = plt.get_cmap("tab10")(np.arange(len(names)) % 10)
    for ax, (key, title) in zip(axes.ravel(), keys_and_titles):
        vals = [stats[n][key] for n in names]
        bars = ax.bar(range(len(names)), vals, color=colors)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", linestyle=":", alpha=0.5)
        vmax = max(vals) if vals else 1.0
        for b, v in zip(bars, vals):
            txt = f"{v:.2g}" if vmax < 1e3 else f"{v:.2e}"
            ax.text(
                b.get_x() + b.get_width() / 2,
                b.get_height(),
                txt,
                ha="center", va="bottom", fontsize=7,
            )
        if vmax > 0 and vmax / max(min(v for v in vals if v > 0), 1e-12) > 50:
            ax.set_yscale("log")
    fig.suptitle(
        "Hidden-state geometry on synth_test (last-layer)  —  signal vs noise",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(save_path, dpi=130)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--runs", nargs="+", required=True,
        help="checkpoint_dir[:label] pairs, e.g. ft-r-...:regular",
    )
    p.add_argument(
        "--base_model", default="meta-llama/Llama-3.2-1B-Instruct",
        help="tokenizer source; must match training",
    )
    p.add_argument(
        "--data", default="datasets/synth_test.jsonl",
        help="jsonl file of {messages: [...]}",
    )
    p.add_argument("--n", type=int, default=64, help="number of examples")
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--layer", type=int, default=-1, help="hidden_states layer idx")
    p.add_argument("--out_dir", default=".", help="where to save json + png")
    p.add_argument("--tag", default="signal_noise", help="output filename stem")
    args = p.parse_args()

    runs: List[Tuple[str, Path]] = []
    for spec in args.runs:
        if ":" in spec:
            path, label = spec.split(":", 1)
        else:
            path = spec
            label = Path(spec).name
        runs.append((label, Path(path)))

    examples = load_examples(Path(args.data), args.n)
    print(f"Loaded {len(examples)} examples from {args.data}")

    stats: Dict[str, Dict[str, float]] = {}
    for label, ckpt in runs:
        print(f"[{label}] forward pass on {ckpt} ...")
        s = measure_one_model(
            ckpt, args.base_model, examples,
            max_length=args.max_length, layer=args.layer,
        )
        stats[label] = s
        print(f"  {s}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / f"{args.tag}_stats.json"
    with json_path.open("w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved stats: {json_path}")

    png_path = out_dir / f"{args.tag}_bars.png"
    plot_bars(stats, png_path)
    print(f"Saved figure: {png_path}")


if __name__ == "__main__":
    main()
