#!/usr/bin/env python3
"""
Stress test 1 — Impulse noise / hidden-state perturbation (recovery curve).

Protocol (matches the spec you described):
  1) Clean forward: save last-layer hidden states H^clean over the sequence.
  2) Perturbed forward: same inputs; after decoder block ``inject_layer``,
     add z ~ N(0, sigma^2 I) only at token index t_0; no further noise.
  3) For t > t_0, Delta_t = ||h_t^pert - h_t^clean||_2 (L2 per position).

A trajectory that contracts back toward the clean manifold should show
Delta_t decreasing (on average) for t > t_0.

Typical use (compare control_JEPA vs STP checkpoints):
  python stress_test_impulse_recovery.py \\
    --checkpoint_a path/to/control_ckpt --label_a control_JEPA \\
    --checkpoint_b path/to/stp_ckpt --label_b STP \\
    --eval_jsonl datasets/synth_test.jsonl --max_examples 32

Requires: torch, transformers, matplotlib, numpy.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)


def load_jsonl_messages(path: Path, n: int) -> List[dict]:
    rows: List[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "messages" in obj:
                rows.append(obj)
            if len(rows) >= n:
                break
    return rows


def get_decoder_layers(model: nn.Module) -> nn.ModuleList:
    """Return the stack of transformer decoder blocks (Llama / Mistral / Qwen-style)."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    if hasattr(model, "transformer") and hasattr(model.transformer, "layers"):
        return model.transformer.layers
    raise RuntimeError(
        "Unsupported architecture: could not find decoder layers "
        "(expected model.model.layers or transformer.h)."
    )


def real_token_span(attention_mask_1d: torch.Tensor) -> Tuple[int, int]:
    """First / one-past-last indices where mask is 1 (handles left- or right-padded batches)."""
    nz = (attention_mask_1d != 0).nonzero(as_tuple=True)[0]
    if nz.numel() == 0:
        return 0, 0
    return int(nz[0].item()), int(nz[-1].item()) + 1


def tokenize_chat(
    tokenizer,
    messages: List[dict],
    max_length: int,
) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
    """Returns input_ids, attention_mask, real_start, real_end (exclusive; end-start = #real tokens)."""
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    tok = tokenizer(
        formatted,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt",
        add_special_tokens=True,
    )
    ids = tok["input_ids"]
    am = tok["attention_mask"]
    start, end = real_token_span(am[0])
    return ids, am, start, end


def forward_last_hidden(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Last tensor in hidden_states tuple: [B, T, D]."""
    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
    )
    return out.hidden_states[-1]


def resolve_inject_layer(n_layers: int, inject_layer: int) -> int:
    """
    Resolve user layer index into [0, n_layers-1].

    Important: for this recovery metric we need post-perturbation propagation
    along depth, so default (-1) intentionally maps to the *first* layer, not
    the last layer.
    """
    if n_layers <= 0:
        return 0
    if inject_layer == -1:
        return 0
    if inject_layer < 0:
        inject_layer = n_layers + inject_layer
    if inject_layer < 0:
        return 0
    if inject_layer >= n_layers:
        return n_layers - 1
    return inject_layer


def make_one_shot_noise_hook(
    t0: int,
    sigma: float,
    generator: torch.Generator,
) -> Tuple[callable, callable]:
    """
    Forward hook on a decoder layer output (first return is hidden [B,T,D]).
    Adds Gaussian noise only at batch positions [:, t0, :], once per forward.
    """
    done = [False]

    def hook(_module, _inp, output):
        if done[0]:
            return output
        if isinstance(output, tuple):
            h = output[0]
            rest = output[1:]
        else:
            h = output
            rest = ()
        if t0 < 0 or t0 >= h.shape[1]:
            return output
        noise = torch.randn(
            h.shape[0],
            h.shape[-1],
            device=h.device,
            dtype=h.dtype,
            generator=generator,
        ) * float(sigma)
        h_new = h.clone()
        h_new[:, t0, :] = h_new[:, t0, :] + noise
        done[0] = True
        if isinstance(output, tuple):
            return (h_new,) + rest
        return h_new

    def reset():
        done[0] = False

    return hook, reset


@torch.no_grad()
def curve_one_example(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    real_start: int,
    real_end: int,
    t0: int,
    sigma: float,
    inject_layer: int,
    seed: int,
    device: torch.device,
) -> Optional[torch.Tensor]:
    """
    Returns Δ_t for global indices t in {t0+1, ..., real_end-1} (only within the
    non-padded span). ``t0`` is a **global** sequence index (same index space as
    ``input_ids``), and must satisfy real_start <= t0 < real_end - 1.
    """
    if real_end - real_start < 2:
        return None
    if t0 < real_start or t0 >= real_end - 1:
        return None
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    # Clone so a second forward cannot overwrite activations (HF may reuse buffers).
    h_clean = forward_last_hidden(model, input_ids, attention_mask)[0].detach().clone()

    layers = get_decoder_layers(model)
    inject_layer = resolve_inject_layer(len(layers), inject_layer)

    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))
    hook_fn, reset = make_one_shot_noise_hook(t0, sigma, gen)
    handle = layers[inject_layer].register_forward_hook(hook_fn)
    try:
        reset()
        h_pert = forward_last_hidden(model, input_ids, attention_mask)[0].detach().clone()
    finally:
        handle.remove()

    d = (h_pert - h_clean).float()
    dist = d.norm(dim=-1)  # [seq_len]
    return dist[t0 + 1 : real_end].cpu()


def aggregate_curves(
    model: nn.Module,
    tokenizer,
    examples: List[dict],
    max_length: int,
    t0: int,
    sigma: float,
    inject_layer: int,
    base_seed: int,
    device: torch.device,
) -> Tuple[torch.Tensor, int]:
    """Mean curve over examples; pads shorter tails with nan then nanmean."""
    curves: List[torch.Tensor] = []
    for i, ex in enumerate(examples):
        ids, am, r0, r1 = tokenize_chat(tokenizer, ex["messages"], max_length)
        span = r1 - r0
        t0_eff = (r0 + max(0, span // 4)) if t0 < 0 else t0
        c = curve_one_example(
            model,
            ids,
            am,
            r0,
            r1,
            t0_eff,
            sigma,
            inject_layer,
            base_seed + i,
            device,
        )
        if c is None or c.numel() == 0:
            continue
        curves.append(c)
    if not curves:
        raise RuntimeError("No valid examples produced recovery curves (check t0 vs lengths).")
    max_t = max(x.numel() for x in curves)
    mat = torch.full((len(curves), max_t), float("nan"))
    for ri, c in enumerate(curves):
        mat[ri, : c.numel()] = c
    mean_curve = torch.nanmean(mat, dim=0)
    return mean_curve, len(curves)


def parse_int_csv(text: str) -> List[int]:
    vals = [x.strip() for x in text.split(",") if x.strip()]
    if not vals:
        return []
    return [int(x) for x in vals]


def mean_std_over_curves(curves: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pad variable-length curves with nan; return nanmean/nanstd over rows."""
    if not curves:
        raise RuntimeError("No curves to aggregate.")
    max_t = max(c.numel() for c in curves)
    mat = torch.full((len(curves), max_t), float("nan"))
    for i, c in enumerate(curves):
        mat[i, : c.numel()] = c
    mean = torch.nanmean(mat, dim=0)
    if len(curves) <= 1:
        std = torch.zeros_like(mean)
    else:
        # Backward-compatible nanstd for older torch versions (no torch.nanstd).
        valid = ~torch.isnan(mat)
        centered = torch.where(valid, mat - mean.unsqueeze(0), torch.zeros_like(mat))
        counts = valid.sum(dim=0).clamp_min(1)
        var = (centered * centered).sum(dim=0) / counts
        std = torch.sqrt(var)
    return mean, std


def normalize_curve(curve: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    if curve.numel() == 0:
        return curve
    d1 = float(curve[0].item())
    if abs(d1) < eps:
        return torch.full_like(curve, float("nan"))
    return curve / d1


def auc_at_k(curve: torch.Tensor, k: int) -> float:
    """Mean over first k points (k clipped by available length). Lower is better."""
    if curve.numel() == 0:
        return float("nan")
    kk = max(1, min(int(k), int(curve.numel())))
    return float(curve[:kk].mean().item())


def safe_nanmean_std_1d(values: torch.Tensor) -> Tuple[float, float]:
    """Return (nanmean, nanstd) using ops available in older torch."""
    if values.numel() == 0:
        return float("nan"), float("nan")
    valid = ~torch.isnan(values)
    count = int(valid.sum().item())
    if count == 0:
        return float("nan"), float("nan")
    v = values[valid]
    mean = float(v.mean().item())
    if count == 1:
        return mean, 0.0
    var = float(((v - mean) ** 2).mean().item())
    return mean, var ** 0.5


def plot_comparison(
    raw_mean: Dict[str, torch.Tensor],
    raw_std: Dict[str, torch.Tensor],
    norm_mean: Dict[str, torch.Tensor],
    norm_std: Dict[str, torch.Tensor],
    out_path: Path,
    title: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.5))

    # Left: raw Delta
    ax = axes[0]
    steps = range(1, 1 + next(iter(raw_mean.values())).numel())
    for label, y in raw_mean.items():
        yy = y.numpy()
        xs = list(steps)[: len(yy)]
        ax.plot(xs, yy, label=label, linewidth=2)
        if label in raw_std:
            ss = raw_std[label].numpy()
            ax.fill_between(xs, yy - ss, yy + ss, alpha=0.18)
    ax.set_xlabel("Steps after perturbation (t - t0)")
    ax.set_ylabel(r"$\Delta_t = \|h_t^{pert} - h_t^{clean}\|_2$")
    ax.set_title("Raw recovery")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Right: normalized Delta/Delta1
    ax2 = axes[1]
    steps_n = range(1, 1 + next(iter(norm_mean.values())).numel())
    for label, y in norm_mean.items():
        yy = y.numpy()
        xs = list(steps_n)[: len(yy)]
        ax2.plot(xs, yy, label=label, linewidth=2)
        if label in norm_std:
            ss = norm_std[label].numpy()
            ax2.fill_between(xs, yy - ss, yy + ss, alpha=0.18)
    ax2.set_xlabel("Steps after perturbation (t - t0)")
    ax2.set_ylabel(r"$\Delta_t / \Delta_{t_0+1}$")
    ax2.set_title("Normalized recovery")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    fig.suptitle(title)
    fig.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main() -> None:
    p = argparse.ArgumentParser(description="Impulse-noise hidden-state recovery stress test.")
    p.add_argument("--checkpoint_a", type=str, required=True, help="First model dir (e.g. control_JEPA).")
    p.add_argument("--checkpoint_b", type=str, default="", help="Second model dir (e.g. STP); omit to only run A.")
    p.add_argument("--label_a", type=str, default="model_A")
    p.add_argument("--label_b", type=str, default="model_B")
    p.add_argument("--eval_jsonl", type=str, default="datasets/synth_test.jsonl")
    p.add_argument(
        "--original_model_name",
        type=str,
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="Used only when --use_evaluate_loader is set (to match evaluate.py loading path).",
    )
    p.add_argument(
        "--use_evaluate_loader",
        action="store_true",
        help="Load with evaluate.load_model_and_tokenizer (same path as stress_test_synth_eval.py).",
    )
    p.add_argument("--device_map", type=str, default="auto", help="Used with --use_evaluate_loader.")
    p.add_argument("--max_examples", type=int, default=32)
    p.add_argument("--max_length", type=int, default=512, help="Tokenizer truncation cap (same as L_train-style eval window).")
    p.add_argument(
        "--t0",
        type=int,
        default=-1,
        help=(
            "Global token index for noise (same indexing as input_ids). Must lie in the "
            "non-padded span and before the last real token. Use -1 to pick "
            "real_start + floor(span/4) per example (robust to left padding)."
        ),
    )
    p.add_argument("--sigma", type=float, default=0.2, help="Noise std (same units as hidden activations).")
    p.add_argument(
        "--inject_layer",
        type=int,
        default=-1,
        help=(
            "Decoder block index to perturb after. For this script, -1 means FIRST block "
            "(default; enables forward propagation through remaining layers). "
            "Other negatives use Python-style indexing, e.g. -2 = second-last."
        ),
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--seeds",
        type=str,
        default="",
        help="Comma-separated seeds for multi-seed aggregation. If empty, use --seed only.",
    )
    p.add_argument(
        "--auc_ks",
        type=str,
        default="10,20",
        help="Comma-separated K values for AUC@K (mean first K steps) on raw and normalized curves.",
    )
    p.add_argument("--out_json", type=str, default="stress_impulse_recovery.json")
    p.add_argument("--out_plot", type=str, default="stress_impulse_recovery.png")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    eval_path = Path(args.eval_jsonl)
    examples = load_jsonl_messages(eval_path, args.max_examples)
    if not examples:
        raise SystemExit(f"No messages loaded from {eval_path}")
    seeds = parse_int_csv(args.seeds) if args.seeds.strip() else [int(args.seed)]
    auc_ks = parse_int_csv(args.auc_ks)
    if not auc_ks:
        raise SystemExit("--auc_ks must provide at least one integer.")

    def run_ckpt(path: str, label: str) -> Dict[int, torch.Tensor]:
        ckpt = Path(path)
        if args.use_evaluate_loader:
            import evaluate as ev

            model, tok = ev.load_model_and_tokenizer(
                str(ckpt),
                args.original_model_name,
                device_map=args.device_map,
            )
            model.eval()
        else:
            tok = AutoTokenizer.from_pretrained(str(ckpt), trust_remote_code=True)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            model = AutoModelForCausalLM.from_pretrained(
                str(ckpt),
                torch_dtype=dtype,
                trust_remote_code=True,
            )
            model.eval()
            model.to(device)
        curves_by_seed: Dict[int, torch.Tensor] = {}
        for s in seeds:
            curve, n_used = aggregate_curves(
                model,
                tok,
                examples,
                args.max_length,
                args.t0,
                args.sigma,
                args.inject_layer,
                s,
                device,
            )
            curves_by_seed[int(s)] = curve
            print(
                f"[{label}] seed={s} averaged {n_used} examples; curve length {curve.numel()}",
                flush=True,
            )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return curves_by_seed

    curves_by_label_seed: Dict[str, Dict[int, torch.Tensor]] = {
        args.label_a: run_ckpt(args.checkpoint_a, args.label_a)
    }
    if args.checkpoint_b.strip():
        curves_by_label_seed[args.label_b] = run_ckpt(args.checkpoint_b, args.label_b)

    raw_mean: Dict[str, torch.Tensor] = {}
    raw_std: Dict[str, torch.Tensor] = {}
    norm_mean: Dict[str, torch.Tensor] = {}
    norm_std: Dict[str, torch.Tensor] = {}
    auc_summary: Dict[str, Dict[str, Dict[str, float]]] = {}

    for label, by_seed in curves_by_label_seed.items():
        seed_order = sorted(by_seed.keys())
        seed_curves = [by_seed[s] for s in seed_order]
        m_raw, s_raw = mean_std_over_curves(seed_curves)
        raw_mean[label] = m_raw
        raw_std[label] = s_raw

        seed_norm = [normalize_curve(c) for c in seed_curves]
        m_norm, s_norm = mean_std_over_curves(seed_norm)
        norm_mean[label] = m_norm
        norm_std[label] = s_norm

        auc_summary[label] = {"raw": {}, "normalized": {}}
        for k in auc_ks:
            vals_raw = torch.tensor([auc_at_k(c, k) for c in seed_curves], dtype=torch.float32)
            vals_norm = torch.tensor([auc_at_k(c, k) for c in seed_norm], dtype=torch.float32)
            rm, rs = safe_nanmean_std_1d(vals_raw)
            nm, ns = safe_nanmean_std_1d(vals_norm)
            auc_summary[label]["raw"][f"auc@{k}_mean"] = rm
            auc_summary[label]["raw"][f"auc@{k}_std"] = rs
            auc_summary[label]["normalized"][f"auc@{k}_mean"] = nm
            auc_summary[label]["normalized"][f"auc@{k}_std"] = ns

    out_json = Path(args.out_json)
    payload = {
        "eval_jsonl": str(eval_path),
        "max_examples": args.max_examples,
        "max_length": args.max_length,
        "t0": args.t0,
        "t0_note": "If t0 is -1, injection index is real_start + span//4 per example (not stored per-step in JSON).",
        "sigma": args.sigma,
        "inject_layer": args.inject_layer,
        "inject_layer_note": "inject_layer=-1 maps to first decoder block in this script.",
        "seed": args.seed,
        "seeds": seeds,
        "auc_ks": auc_ks,
        "curves_by_seed": {
            label: {str(s): [float(x) for x in c.tolist()] for s, c in by_seed.items()}
            for label, by_seed in curves_by_label_seed.items()
        },
        "curves_raw_mean": {k: [float(x) for x in v.tolist()] for k, v in raw_mean.items()},
        "curves_raw_std": {k: [float(x) for x in v.tolist()] for k, v in raw_std.items()},
        "curves_norm_mean": {k: [float(x) for x in v.tolist()] for k, v in norm_mean.items()},
        "curves_norm_std": {k: [float(x) for x in v.tolist()] for k, v in norm_std.items()},
        "auc_summary": auc_summary,
        "x_positions": list(range(1, 1 + next(iter(raw_mean.values())).numel())),
        "x_positions_note": "Relative recovery steps after perturbation: (t - t0), starting from 1.",
    }
    out_json.write_text(json.dumps(payload, indent=2))

    plot_comparison(
        raw_mean,
        raw_std,
        norm_mean,
        norm_std,
        Path(args.out_plot),
        title=(
            f"Impulse recovery (t0={args.t0}, sigma={args.sigma}, "
            f"inject_layer={args.inject_layer}, n_seeds={len(seeds)})"
        ),
    )
    print("\n=== AUC summary (lower is better) ===")
    for label in auc_summary:
        print(f"\n[{label}]")
        for k in auc_ks:
            rm = auc_summary[label]["raw"][f"auc@{k}_mean"]
            rs = auc_summary[label]["raw"][f"auc@{k}_std"]
            nm = auc_summary[label]["normalized"][f"auc@{k}_mean"]
            ns = auc_summary[label]["normalized"][f"auc@{k}_std"]
            print(f"  raw auc@{k}: {rm:.4f} ± {rs:.4f}")
            print(f"  norm auc@{k}: {nm:.4f} ± {ns:.4f}")
    print(f"Wrote {out_json} and {args.out_plot}", flush=True)


if __name__ == "__main__":
    main()
