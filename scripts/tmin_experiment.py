#!/usr/bin/env python3
"""
Minimum-context (Tmin) validation experiment from "Tmin experiment.pdf".

Implements the six phases:
  1) derive theoretical Tmin / V* from (gamma, tau, delta, V_ts),
  2) load a trained checkpoint,
  3) split test examples by sequence length (< Tmin vs >= Tmin),
  4) measure per-sequence geometry metrics M1..M5,
  5) report group tables,
  6) export the two-panel figure.

Key design decisions (matching LyapunovControlLoss / version 2):
  - V_t = ||e_t||^2 / ||v_geo||^2  (norm_mode="v_geo", matching training)
  - V_ts calibrated at assistant start a_s (NOT at u_s which is degenerate 0)
  - V_final measured at a_e - 1 (last non-trivial position; a_e is tautologically 0)
  - Envelope: V_pred(k) = gamma^k * V_ts + V*(1 - gamma^k)  (tight ISS bound)
  - Horizon T = a_e - u_s (token offset from user pivot to last assistant token)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import evaluate as eval_mod
from eval_testset import prediction_matches_gold


# ---------------------------------------------------------------------------
# Phase 1 — Theory (computed from hyperparameters only)
# ---------------------------------------------------------------------------


def v_star_steady(gamma: float, tau: float) -> float:
    return tau / (1.0 - gamma)


def t_min_tokens(v_ts: float, gamma: float, tau: float, delta: float) -> float:
    """T_min = log(V_ts (1-gamma) / (delta tau)) / log(1/gamma)."""
    if gamma <= 0.0 or gamma >= 1.0:
        raise ValueError("gamma must be in (0, 1)")
    if tau <= 0.0 or delta <= 0.0:
        raise ValueError("tau and delta must be positive")
    num = v_ts * (1.0 - gamma) / (delta * tau)
    if num <= 1.0:
        return 0.0
    return math.log(num) / math.log(1.0 / gamma)


def v_pred_at_offset(k: float, v_ts: float, gamma: float, tau: float) -> float:
    """PDF Eq.(4): V_pred(k) = V* + gamma^k * V_ts (theoretical upper bound)."""
    if k < 0:
        return float("nan")
    v_star = v_star_steady(gamma, tau)
    return v_star + (gamma ** k) * v_ts


def tube_satisfied(v_final: float, delta: float, gamma: float, tau: float) -> bool:
    v_star = v_star_steady(gamma, tau)
    return v_final <= (1.0 + delta) * v_star


# ---------------------------------------------------------------------------
# Geometry: compute V_all matching LyapunovControlLoss norm_mode="v_geo"
# ---------------------------------------------------------------------------


def compute_V_all_vgeo(
    h: torch.Tensor,
    u_s: int,
    u_e: int,
    a_s: int,
    a_e: int,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute V_t = ||e_t||^2 / ||v_geo||^2 at all T positions.

    This exactly reproduces LyapunovControlLoss with norm_mode="v_geo".
    Returns tensor of shape (T,) with V at each position.
    Positions outside [u_s, u_e] union [a_s, a_e] have V=0 (no displacement defined).
    """
    T, D = h.shape
    device = h.device
    dtype = h.dtype

    d_all = torch.zeros(T, D, device=device, dtype=dtype)
    v_user = h[u_e] - h[u_s]
    d_all[u_s : u_e + 1] = h[u_s : u_e + 1] - h[u_s]
    d_all[a_s : a_e + 1] = v_user + (h[a_s : a_e + 1] - h[a_s])
    v_geo = v_user + (h[a_e] - h[a_s])

    L_sq = (v_geo * v_geo).sum().clamp(min=eps)
    v_hat = v_geo / L_sq.sqrt()

    proj_scalar = (d_all @ v_hat).unsqueeze(-1)
    p_all = proj_scalar * v_hat.unsqueeze(0)
    e_all = d_all - p_all
    e_sq = (e_all ** 2).sum(dim=-1)

    V_all = e_sq / L_sq
    return V_all


# ---------------------------------------------------------------------------
# Data loading and span detection
# ---------------------------------------------------------------------------


def load_examples(jsonl_path: Path, max_examples: Optional[int]) -> List[Dict]:
    rows: List[Dict] = []
    with jsonl_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            if "messages" not in ex:
                continue
            rows.append(ex)
            if max_examples is not None and len(rows) >= max_examples:
                break
    return rows


def find_start_end(content: str, tokenizer, input_ids: List[int], attention_mask: List[int]) -> Tuple[int, int]:
    """Locate content span in tokenized sequence. Returns (start-1, end)."""
    tokens = tokenizer.encode(content, add_special_tokens=False)
    decoded_content = [tokenizer.decode(t) for t in tokens]
    decoded_input = [tokenizer.decode(t) for t in input_ids]
    n = len(tokens)

    for i in range(len(input_ids) - n, -1, -1):
        if attention_mask[i] == 1 and decoded_input[i : i + n] == decoded_content:
            if i <= 0:
                continue
            return i - 1, i + n - 1

    if n > 0:
        target = tokenizer.decode(tokens).strip()
        if target:
            s = len(input_ids)
            lo_delta = max(1 - n, -4)
            hi_delta = 4
            for i in range(s - 1, 0, -1):
                if attention_mask[i] != 1:
                    continue
                for delta in range(lo_delta, hi_delta + 1):
                    l_win = n + delta
                    if l_win <= 0 or i + l_win > s:
                        continue
                    if attention_mask[i + l_win - 1] != 1:
                        continue
                    if tokenizer.decode(input_ids[i : i + l_win]).strip() == target:
                        return i - 1, i + l_win - 1

    raise RuntimeError("Failed to locate content span in tokenized sequence.")


# ---------------------------------------------------------------------------
# Per-example geometry extraction
# ---------------------------------------------------------------------------


@torch.no_grad()
def geometry_for_example(
    model: torch.nn.Module,
    tokenizer,
    messages: List[Dict],
    *,
    max_length: int,
    layer: int,
    eps: float = 1e-8,
) -> Optional[Dict]:
    """Compute V_all and per-sequence metrics for one chat example.

    Returns dict with:
      V_all_offsets: V at offsets k=0..horizon from u_s (for the curve)
      V_ts: V at assistant start a_s (tube entry)
      V_final: V at a_e - 1 (last non-trivial position)
      horizon: a_e - u_s (total token offset)
      u_s, u_e, a_s, a_e: span indices
    """
    if len(messages) < 3:
        return None

    full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    tok = tokenizer(
        full_text,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors=None,
        add_special_tokens=True,
    )
    input_ids = tok["input_ids"]
    attention_mask = tok["attention_mask"]

    try:
        u0, ue = find_start_end(messages[1]["content"], tokenizer, input_ids, attention_mask)
        a0, ae = find_start_end(messages[2]["content"], tokenizer, input_ids, attention_mask)
    except Exception:
        return None

    u_s, u_e = u0 + 1, ue
    a_s, a_e = a0 + 1, ae
    tlen = len(input_ids)
    if not (0 <= u_s <= u_e < tlen and 0 <= a_s <= a_e < tlen):
        return None
    if a_e <= a_s:
        return None

    device = next(model.parameters()).device
    iid = torch.tensor([input_ids], dtype=torch.long, device=device)
    am = torch.tensor([attention_mask], dtype=torch.long, device=device)
    out = model(input_ids=iid, attention_mask=am, output_hidden_states=True, use_cache=False)
    h = out.hidden_states[layer][0].float()  # (T, D) in float32

    V_all = compute_V_all_vgeo(h, u_s, u_e, a_s, a_e, eps=eps)

    horizon = a_e - u_s
    # V at assistant start (tube entry point)
    v_ts_i = float(V_all[a_s].item())
    # V at a_e - 1 (last meaningful position; a_e itself is always 0)
    v_final_i = float(V_all[a_e - 1].item())

    # Extract curve: V at offsets k=0..horizon from u_s
    # (skip a_e since it's trivially 0)
    v_offsets: List[float] = []
    for t in range(u_s, a_e):
        v_offsets.append(float(V_all[t].item()))

    return {
        "V_offsets": v_offsets,
        "V_ts": v_ts_i,
        "V_final": v_final_i,
        "horizon": int(horizon),
        "u_s": u_s,
        "u_e": u_e,
        "a_s": a_s,
        "a_e": a_e,
    }


# ---------------------------------------------------------------------------
# Accuracy (optional)
# ---------------------------------------------------------------------------


@torch.no_grad()
def exact_match_for_example(
    model: torch.nn.Module,
    tokenizer,
    *,
    messages: List[Dict],
    base_model_name: str,
    max_length: int,
    max_new_tokens: int,
    eval_stem: str,
) -> float:
    full = eval_mod.get_messages(base_model_name, messages)
    prompt = eval_mod.format_conversation(full, tokenizer, include_assistant=False, plain=False)
    gen_cfg = type("GenCfg", (), {"max_length": max_length})()
    generated = eval_mod.generate_response(model, tokenizer, prompt, gen_cfg, max_new_tokens)
    return float(prediction_matches_gold(generated, messages, eval_stem))


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------


def mean_or_nan(values: Sequence[float]) -> float:
    if not values:
        return float("nan")
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def summarize_group(rows: Sequence[Dict]) -> Dict[str, float]:
    vfinal = [r["V_final"] for r in rows]
    pred_err = [abs(r["V_pred"] - r["V_final"]) for r in rows]
    sat = [r["Sat"] for r in rows]
    acc = [r["Acc"] for r in rows if not math.isnan(r["Acc"])]
    return {
        "count": float(len(rows)),
        "mean_V_final": mean_or_nan(vfinal),
        "mean_abs_Vpred_minus_Vfinal": mean_or_nan(pred_err),
        "tube_satisfaction_rate": mean_or_nan(sat),
        "accuracy": mean_or_nan(acc),
    }


def mean_curve_sqrt(raw: DefaultDict[int, List[float]]) -> Tuple[np.ndarray, np.ndarray]:
    """Mean of sqrt(V) at each offset k (PDF Phase 6: mean_i sqrt(V^(i)_t))."""
    if not raw:
        return np.array([]), np.array([])
    ks = sorted(raw.keys())
    means = [
        float(np.nanmean([math.sqrt(max(v, 1e-30)) for v in raw[k]]))
        for k in ks
    ]
    return np.array(ks, dtype=float), np.array(means, dtype=float)


# ---------------------------------------------------------------------------
# Plotting (Phase 6)
# ---------------------------------------------------------------------------


def plot_main_figure(
    *,
    out_path: Path,
    gamma: float,
    tau: float,
    tmin: int,
    v_star: float,
    v_ts_mean: float,
    curve_g2: DefaultDict[int, List[float]],
    curve_g1: DefaultDict[int, List[float]],
    g1_summary: Dict[str, float],
    g2_summary: Dict[str, float],
) -> None:
    k_g2, sqrt_g2 = mean_curve_sqrt(curve_g2)
    k_g1, sqrt_g1 = mean_curve_sqrt(curve_g1)

    k_max = int(max(
        np.max(k_g2) if len(k_g2) else 0,
        np.max(k_g1) if len(k_g1) else 0,
        tmin + 2,
    ))
    k_theory = np.arange(0, k_max + 1, dtype=float)
    v_theory = np.array([v_pred_at_offset(float(k), v_ts_mean, gamma, tau) for k in k_theory])
    sqrt_v_theory = np.sqrt(np.maximum(v_theory, 0.0))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=150,
                             gridspec_kw={"width_ratios": [1.4, 1.0], "wspace": 0.35})

    ax = axes[0]
    if len(k_g2) > 0:
        ax.plot(k_g2, sqrt_g2, color="#1f77b4", lw=2.0, label=r"Empirical G2 ($T \geq T_{\min}$)")
    if len(k_g1) > 0:
        ax.plot(k_g1, sqrt_g1, color="#1f77b4", lw=2.0, ls="--",
                label="G1 (short / truncated)")
    ax.plot(k_theory, sqrt_v_theory, color="#ff7f0e", lw=2.0, ls="--",
            label=r"Envelope $\sqrt{V_{\mathrm{pred}}(k)}$")
    ax.axhline(math.sqrt(max(v_star, 0.0)), color="#d62728", lw=1.8,
               label=r"$\sqrt{V^*}$ threshold")
    ax.axvline(tmin, color="black", ls="--", lw=1.2, alpha=0.7, label="$T_{\\min}$")
    ax.set_xlabel("Token offset $k = t - t_s$")
    ax.set_ylabel(r"Mean $\sqrt{V_t}$")
    ax.set_title("Geometric validation (Phase 6)")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, loc="upper right")

    ax2 = axes[1]
    x = np.array([0, 1])
    g1_acc = g1_summary["accuracy"]
    g2_acc = g2_summary["accuracy"]
    acc_vals = [
        g1_acc * 100 if not math.isnan(g1_acc) else 0.0,
        g2_acc * 100 if not math.isnan(g2_acc) else 0.0,
    ]
    bars = ax2.bar(x, acc_vals, width=0.45, color=["#aec7e8", "#1f78b4"],
                   edgecolor="black", label="Accuracy (%)")
    ax2.set_xticks(x)
    ax2.set_xticklabels([r"G1 ($T < T_{\min}$)", r"G2 ($T \geq T_{\min}$)"])
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_ylim(0, 105)
    ax2.set_title("Performance")
    ax2.grid(True, axis="y", alpha=0.25)

    ax3 = ax2.twinx()
    g1_vf = g1_summary["mean_V_final"]
    g2_vf = g2_summary["mean_V_final"]
    if not math.isnan(g1_vf) and not math.isnan(g2_vf):
        ax3.plot(x, [g1_vf, g2_vf], color="#ff7f0e", marker="o", lw=1.5, markersize=8,
                 label=r"Mean $V_{\mathrm{final}}$")
        ax3.set_ylabel(r"Mean $V_{\mathrm{final}}$")
        ax3.legend(loc="upper right", fontsize=8)

    fig.suptitle(
        f"$\\gamma={gamma}$, $\\tau={tau}$, $T_{{\\min}}={tmin}$, "
        f"$V^*={v_star:.4g}$",
        fontsize=11, y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate the Tmin minimum-context corollary.")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--eval_jsonl", type=str, default="datasets/gsm8k_test.jsonl")
    p.add_argument("--base_model_name", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    p.add_argument("--gamma", type=float, required=True)
    p.add_argument("--tau", type=float, required=True)
    p.add_argument("--delta", type=float, default=0.01)
    p.add_argument("--vts_override", type=float, default=-1.0,
                   help="If > 0, use this V_ts instead of calibrating at a_s.")
    p.add_argument("--calibration_n", type=int, default=32,
                   help="Number of examples to estimate V_ts from.")
    p.add_argument("--max_examples", type=int, default=128)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--layer", type=int, default=-1)
    p.add_argument("--accuracy_mode", choices=("exact_match", "none"), default="exact_match")
    p.add_argument("--out_json", type=str, default="tmin_experiment_report.json")
    p.add_argument("--out_plot", type=str, default="tmin_experiment_main_figure.png")
    p.add_argument("--save_per_sequence", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    eval_path = Path(args.eval_jsonl)
    max_examples = None if args.max_examples <= 0 else int(args.max_examples)
    examples = load_examples(eval_path, max_examples=max_examples)
    if not examples:
        raise SystemExit(f"No valid examples found in {eval_path}")

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    use_cuda = torch.cuda.is_available()
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint,
        torch_dtype=torch.bfloat16 if use_cuda else torch.float32,
        device_map="auto" if use_cuda else None,
        trust_remote_code=True,
    )
    model.eval()

    # -----------------------------------------------------------------------
    # Pass 1: geometry extraction (V_all at every position)
    # -----------------------------------------------------------------------
    geom_rows: List[Dict] = []
    for idx, ex in enumerate(examples):
        g = geometry_for_example(
            model, tokenizer, ex["messages"],
            max_length=args.max_length, layer=args.layer,
        )
        if g is None:
            continue
        geom_rows.append({"index": idx, **g})

    if not geom_rows:
        raise SystemExit("No valid examples after geometry extraction.")

    # -----------------------------------------------------------------------
    # Phase 1: Calibrate V_ts at assistant start (a_s)
    # -----------------------------------------------------------------------
    if args.vts_override > 0:
        v_ts_mean = float(args.vts_override)
    else:
        cal_samples = [r["V_ts"] for r in geom_rows[:max(1, args.calibration_n)]]
        v_ts_mean = float(np.mean(cal_samples))

    t_min_float = t_min_tokens(v_ts_mean, args.gamma, args.tau, args.delta)
    tmin = max(1, int(math.ceil(t_min_float)))
    v_star = v_star_steady(args.gamma, args.tau)

    print("=== Phase 1: Theory ===")
    print(f"V_ts (calibrated at a_s): {v_ts_mean:.6g}")
    print(f"T_min = {t_min_float:.2f} -> ceil = {tmin}")
    print(f"V* = tau/(1-gamma) = {v_star:.6g}")
    print(f"sqrt(V*) = {math.sqrt(v_star):.6g}")

    # -----------------------------------------------------------------------
    # Phase 3: Coverage check & split
    # -----------------------------------------------------------------------
    horizons = [r["horizon"] for r in geom_rows]
    n_long = sum(h >= tmin for h in horizons)
    n_short = sum(h < tmin for h in horizons)
    print(f"\n=== Phase 3: Split ===")
    print(f"Horizons: min={min(horizons)} mean={np.mean(horizons):.1f} max={max(horizons)}")
    print(f"G1 (T < {tmin}): {n_short} examples")
    print(f"G2 (T >= {tmin}): {n_long} examples")

    if n_long == 0:
        raise SystemExit(
            f"No examples with horizon >= Tmin={tmin}. "
            "Increase --max_length or use a longer dataset."
        )

    # -----------------------------------------------------------------------
    # Phase 4: Per-sequence metrics
    # -----------------------------------------------------------------------
    rows: List[Dict] = []
    for r in geom_rows:
        horizon = r["horizon"]
        group = "G1" if horizon < tmin else "G2"
        # V_ts per sequence (at a_s)
        v_ts_i = r["V_ts"]
        # V_final at a_e - 1 (last non-trivial position)
        v_final_i = r["V_final"]
        # Theoretical prediction at the final offset
        k_final = horizon - 1  # offset of a_e-1 from u_s
        v_pred_i = v_pred_at_offset(float(k_final), v_ts_i, args.gamma, args.tau)
        sat = tube_satisfied(v_final_i, args.delta, args.gamma, args.tau)
        rows.append({
            "index": r["index"],
            "group": group,
            "horizon": horizon,
            "V_ts": v_ts_i,
            "V_final": v_final_i,
            "V_pred": v_pred_i,
            "Sat": float(sat),
            "Acc": float("nan"),
            "V_offsets": r["V_offsets"],
        })

    # -----------------------------------------------------------------------
    # Phase 4 (M5): Accuracy (optional, after geometry)
    # -----------------------------------------------------------------------
    if args.accuracy_mode == "exact_match":
        eval_stem = eval_path.stem
        print(f"\n=== Phase 4: Accuracy (exact match) ===")
        for r in rows:
            ex = examples[int(r["index"])]
            r["Acc"] = exact_match_for_example(
                model, tokenizer,
                messages=ex["messages"],
                base_model_name=args.base_model_name,
                max_length=args.max_length,
                max_new_tokens=args.max_new_tokens,
                eval_stem=eval_stem,
            )

    # -----------------------------------------------------------------------
    # Phase 5: Group summaries
    # -----------------------------------------------------------------------
    g1_rows = [r for r in rows if r["group"] == "G1"]
    g2_rows = [r for r in rows if r["group"] == "G2"]
    g1_summary = summarize_group(g1_rows)
    g2_summary = summarize_group(g2_rows)

    # Build matched control: truncate G2 arcs to < Tmin
    curve_g2: DefaultDict[int, List[float]] = defaultdict(list)
    curve_g1: DefaultDict[int, List[float]] = defaultdict(list)

    for r in g2_rows:
        offsets = r["V_offsets"]
        for k, v in enumerate(offsets):
            curve_g2[k].append(v)
        # Truncated version: same trajectory but only up to offset < tmin
        for k in range(min(len(offsets), tmin - 1)):
            curve_g1[k].append(offsets[k])

    for r in g1_rows:
        offsets = r["V_offsets"]
        for k, v in enumerate(offsets):
            curve_g1[k].append(v)

    # Matched control summary (G2 truncated at Tmin-1)
    matched_rows: List[Dict] = []
    for r in g2_rows:
        offsets = r["V_offsets"]
        trunc_k = min(len(offsets) - 1, tmin - 2)
        if trunc_k < 0:
            continue
        v_final_trunc = offsets[trunc_k]
        v_pred_trunc = v_pred_at_offset(float(trunc_k), r["V_ts"], args.gamma, args.tau)
        sat_trunc = tube_satisfied(v_final_trunc, args.delta, args.gamma, args.tau)
        matched_rows.append({
            "V_final": v_final_trunc,
            "V_pred": v_pred_trunc,
            "Sat": float(sat_trunc),
            "Acc": r["Acc"],
        })
    matched_summary = summarize_group(matched_rows)

    # -----------------------------------------------------------------------
    # Phase 5: Report
    # -----------------------------------------------------------------------
    print(f"\n=== Phase 5: Results ===")
    print(f"G1 (T < Tmin={tmin}): {g1_summary}")
    print(f"G2 (T >= Tmin={tmin}): {g2_summary}")
    print(f"Matched control (G2 truncated at k={tmin-2}): {matched_summary}")

    out = {
        "phase1_theory": {
            "gamma": args.gamma,
            "tau": args.tau,
            "delta": args.delta,
            "V_ts_calibrated_at_a_s": v_ts_mean,
            "Tmin": tmin,
            "Tmin_float": t_min_float,
            "V_star": v_star,
            "sqrt_V_star": math.sqrt(v_star),
        },
        "data": {
            "checkpoint": args.checkpoint,
            "eval_jsonl": str(eval_path),
            "n_evaluated": len(rows),
            "n_skipped": len(examples) - len(geom_rows),
            "accuracy_mode": args.accuracy_mode,
        },
        "phase3_coverage": {
            "n_short_lt_tmin": n_short,
            "n_long_ge_tmin": n_long,
            "horizon_min": int(min(horizons)),
            "horizon_max": int(max(horizons)),
            "horizon_mean": float(np.mean(horizons)),
        },
        "groups": {
            "G1_T_less_Tmin": g1_summary,
            "G2_T_ge_Tmin": g2_summary,
        },
        "matched_control_G2_truncated": matched_summary,
    }
    if args.save_per_sequence:
        out["per_sequence"] = [
            {k: v for k, v in r.items() if k != "V_offsets"} for r in rows
        ]

    out_json = Path(args.out_json)
    out_json.write_text(json.dumps(out, indent=2))
    print(f"\nWrote JSON: {out_json}")

    # -----------------------------------------------------------------------
    # Phase 6: Figure
    # -----------------------------------------------------------------------
    plot_main_figure(
        out_path=Path(args.out_plot),
        gamma=args.gamma,
        tau=args.tau,
        tmin=tmin,
        v_star=v_star,
        v_ts_mean=v_ts_mean,
        curve_g2=curve_g2,
        curve_g1=curve_g1,
        g1_summary=g1_summary,
        g2_summary=g2_summary,
    )
    print(f"Wrote plot: {args.out_plot}")


if __name__ == "__main__":
    main()
