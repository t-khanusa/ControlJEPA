#!/usr/bin/env python3
"""
Tmin Experiment — δ-Sweep & Theory Demonstration

Demonstrates the Minimum Context-Length Corollary:
  For any δ > 0, define threshold = (1+δ)V* and T_min(δ).
  Theory predicts:
    - For T < T_min: V_t CANNOT have converged below threshold (still in transient)
    - For T ≥ T_min: V_t SHOULD be ≤ threshold (steady-state guarantee)

This script:
  1) Extracts V_t at EVERY position in the assistant span for each sequence
  2) Sweeps δ and computes satisfaction rates at different offsets
  3) Tests whether sequences satisfying the tube condition have better accuracy
  4) Produces a multi-panel visualization contrasting theory vs empirical behavior
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import evaluate as eval_mod
from eval_testset import prediction_matches_gold
from scripts.tmin_experiment import (
    compute_V_all_vgeo,
    find_start_end,
    load_examples,
    t_min_tokens,
    v_star_steady,
)


def v_pred_at_offset(k: float, v_ts: float, gamma: float, tau: float) -> float:
    """PDF Eq.(4): V_pred(k) = V* + gamma^k * V_ts"""
    v_star = v_star_steady(gamma, tau)
    return v_star + (gamma ** k) * v_ts


@torch.no_grad()
def extract_full_geometry(
    model: torch.nn.Module,
    tokenizer,
    examples: List[Dict],
    *,
    max_length: int,
    layer: int,
) -> List[Dict]:
    """Extract per-position V_t for the assistant span of each example."""
    device = next(model.parameters()).device
    results: List[Dict] = []

    for idx, ex in enumerate(examples):
        messages = ex.get("messages", [])
        if len(messages) < 3:
            continue

        full_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        tok = tokenizer(
            full_text, truncation=True, max_length=max_length,
            padding="max_length", return_tensors=None, add_special_tokens=True,
        )
        input_ids = tok["input_ids"]
        attention_mask = tok["attention_mask"]

        try:
            u0, ue = find_start_end(messages[1]["content"], tokenizer, input_ids, attention_mask)
            a0, ae = find_start_end(messages[2]["content"], tokenizer, input_ids, attention_mask)
        except Exception:
            continue

        u_s, u_e = u0 + 1, ue
        a_s, a_e = a0 + 1, ae
        tlen = len(input_ids)
        if not (0 <= u_s <= u_e < tlen and 0 <= a_s <= a_e < tlen):
            continue
        if a_e <= a_s + 2:
            continue

        iid = torch.tensor([input_ids], dtype=torch.long, device=device)
        am = torch.tensor([attention_mask], dtype=torch.long, device=device)
        out = model(input_ids=iid, attention_mask=am, output_hidden_states=True, use_cache=False)
        h = out.hidden_states[layer][0].float()

        V_all = compute_V_all_vgeo(h, u_s, u_e, a_s, a_e)

        # V_t in assistant span only: from a_s to a_e-1 (excluding trivial endpoint)
        assist_len = a_e - a_s
        v_assistant = [float(V_all[a_s + k].item()) for k in range(assist_len)]

        horizon = a_e - u_s
        v_ts_i = float(V_all[a_s].item())
        v_final_i = float(V_all[a_e - 1].item())

        results.append({
            "index": idx,
            "horizon": horizon,
            "assist_len": assist_len,
            "V_ts": v_ts_i,
            "V_final": v_final_i,
            "V_assistant": v_assistant,
        })

        if (idx + 1) % 10 == 0:
            print(f"  [{idx+1}/{len(examples)}] geometry", flush=True)

    return results


@torch.no_grad()
def compute_accuracy(
    model: torch.nn.Module,
    tokenizer,
    examples: List[Dict],
    seq_data: List[Dict],
    *,
    base_model_name: str,
    max_length: int,
    max_new_tokens: int,
) -> None:
    """Add accuracy measurements to seq_data (in-place)."""
    eval_stem = "gsm8k"
    for entry in seq_data:
        idx = entry["index"]
        messages = examples[idx]["messages"]
        try:
            full = eval_mod.get_messages(base_model_name, messages)
            prompt = eval_mod.format_conversation(full, tokenizer, include_assistant=False, plain=False)
            gen_cfg = type("GenCfg", (), {"max_length": max_length})()
            generated = eval_mod.generate_response(model, tokenizer, prompt, gen_cfg, max_new_tokens)
            entry["Acc"] = float(prediction_matches_gold(generated, messages, eval_stem))
        except Exception:
            entry["Acc"] = float("nan")

        if (entry.get("_acc_count", 0) + 1) % 10 == 0:
            print(f"  [{entry.get('_acc_count', 0)+1}] accuracy", flush=True)
        entry["_acc_count"] = entry.get("_acc_count", 0) + 1


def position_satisfaction_analysis(
    seq_data: List[Dict],
    gamma: float,
    tau: float,
    delta: float,
) -> Dict:
    """For a given δ, at each offset k from a_s, measure fraction of sequences with V_k ≤ threshold."""
    v_star = v_star_steady(gamma, tau)
    threshold = (1.0 + delta) * v_star
    v_ts_mean = float(np.mean([r["V_ts"] for r in seq_data]))
    tmin_float = t_min_tokens(v_ts_mean, gamma, tau, delta)
    tmin = max(1, int(math.ceil(tmin_float)))

    max_assist = max(r["assist_len"] for r in seq_data)
    sat_by_offset = []
    n_by_offset = []

    for k in range(max_assist):
        vals = [r["V_assistant"][k] for r in seq_data if len(r["V_assistant"]) > k]
        n_by_offset.append(len(vals))
        if vals:
            sat_by_offset.append(float(np.mean([v <= threshold for v in vals])))
        else:
            sat_by_offset.append(float("nan"))

    return {
        "delta": delta,
        "threshold": threshold,
        "Tmin": tmin,
        "Tmin_float": tmin_float,
        "sat_by_offset": sat_by_offset,
        "n_by_offset": n_by_offset,
    }


def plot_comprehensive(
    seq_data: List[Dict],
    gamma: float,
    tau: float,
    delta_values: List[float],
    out_path: Path,
) -> None:
    """Produce the comprehensive multi-panel visualization."""
    v_star = v_star_steady(gamma, tau)
    v_ts_mean = float(np.mean([r["V_ts"] for r in seq_data]))

    fig = plt.figure(figsize=(16, 12), dpi=150)
    gs = fig.add_gridspec(3, 2, hspace=0.42, wspace=0.30)

    # =========================================================================
    # Panel A: Theory illustration — envelope, threshold, T_min zones
    # =========================================================================
    ax_a = fig.add_subplot(gs[0, 0])
    k_range = np.arange(0, 60, dtype=float)
    envelope = np.array([v_pred_at_offset(k, v_ts_mean, gamma, tau) for k in k_range])
    ax_a.plot(k_range, np.sqrt(envelope), color="#1f77b4", lw=2.5,
              label=r"Envelope $\sqrt{V^* + \gamma^k V_{ts}}$")
    ax_a.axhline(math.sqrt(v_star), color="red", ls="--", lw=1.5,
                 label=f"$\\sqrt{{V^*}} = {math.sqrt(v_star):.4f}$")

    colors_delta = ["#2ca02c", "#ff7f0e", "#9467bd", "#d62728"]
    for i, delta in enumerate([0.5, 2.0, 5.0, 15.0]):
        thr = (1 + delta) * v_star
        tmin_f = t_min_tokens(v_ts_mean, gamma, tau, delta)
        tmin_i = max(1, int(math.ceil(tmin_f)))
        c = colors_delta[i % len(colors_delta)]
        ax_a.axhline(math.sqrt(thr), color=c, ls=":", lw=1.2, alpha=0.8)
        ax_a.axvline(tmin_i, color=c, ls=":", lw=1.2, alpha=0.8)
        ax_a.annotate(f"δ={delta}, $T_{{min}}$={tmin_i}",
                      xy=(tmin_i, math.sqrt(thr)), fontsize=7, color=c,
                      xytext=(tmin_i + 2, math.sqrt(thr) + 0.01), ha="left")

    ax_a.fill_betweenx([0, 0.4], 0, 15, alpha=0.06, color="red", label="Transient zone (T<T_min)")
    ax_a.set_xlabel("Offset $k$ (from $t_s$)")
    ax_a.set_ylabel("$\\sqrt{V}$")
    ax_a.set_title("(A) Theoretical prediction: envelope & T_min zones")
    ax_a.legend(fontsize=7, loc="upper right")
    ax_a.set_xlim(0, 55)
    ax_a.set_ylim(0, 0.38)
    ax_a.grid(True, alpha=0.2)

    # =========================================================================
    # Panel B: Empirical V_t trajectories (assistant span, from a_s)
    # =========================================================================
    ax_b = fig.add_subplot(gs[0, 1])
    max_assist = max(r["assist_len"] for r in seq_data)

    # Plot individual trajectories (thin, transparent)
    for r in seq_data[:40]:
        v_a = r["V_assistant"]
        ax_b.plot(range(len(v_a)), [math.sqrt(max(v, 1e-30)) for v in v_a],
                  color="#1f77b4", alpha=0.12, lw=0.5)

    # Mean curve
    mean_sqrt = []
    for k in range(max_assist):
        vals = [math.sqrt(max(r["V_assistant"][k], 1e-30)) for r in seq_data if len(r["V_assistant"]) > k]
        mean_sqrt.append(float(np.mean(vals)) if vals else float("nan"))
    ax_b.plot(range(len(mean_sqrt)), mean_sqrt, color="#d62728", lw=2.5, label="Empirical mean $\\sqrt{V_t}$")

    # Overlay envelope
    k_plot = np.arange(0, min(max_assist, 200), dtype=float)
    env_plot = np.sqrt(np.array([v_pred_at_offset(k, v_ts_mean, gamma, tau) for k in k_plot]))
    ax_b.plot(k_plot, env_plot, color="#ff7f0e", lw=2, ls="--", label="Theoretical envelope")
    ax_b.axhline(math.sqrt(v_star), color="red", ls="--", lw=1.2, alpha=0.7, label=f"$\\sqrt{{V^*}}$")

    ax_b.set_xlabel("Offset $k$ from assistant start ($a_s$)")
    ax_b.set_ylabel("$\\sqrt{V_t}$")
    ax_b.set_title("(B) Empirical V_t in assistant span")
    ax_b.legend(fontsize=7)
    ax_b.set_xlim(0, min(max_assist, 200))
    ax_b.grid(True, alpha=0.2)

    # =========================================================================
    # Panel C: Position-based satisfaction rate for different δ
    # =========================================================================
    ax_c = fig.add_subplot(gs[1, 0])
    colors_sweep = plt.cm.viridis(np.linspace(0.2, 0.9, len(delta_values)))

    for i, delta in enumerate(delta_values):
        result = position_satisfaction_analysis(seq_data, gamma, tau, delta)
        offsets = range(len(result["sat_by_offset"]))
        ax_c.plot(offsets, result["sat_by_offset"], color=colors_sweep[i], lw=1.5,
                  label=f"δ={delta:.1f}, $T_{{min}}$={result['Tmin']}, thr={result['threshold']:.4f}")
        ax_c.axvline(result["Tmin"], color=colors_sweep[i], ls=":", alpha=0.5, lw=0.8)

    ax_c.set_xlabel("Offset $k$ from assistant start")
    ax_c.set_ylabel("Fraction with $V_k \\leq (1+\\delta)V^*$")
    ax_c.set_title("(C) Position satisfaction rate by δ")
    ax_c.legend(fontsize=6, loc="lower right", ncol=1)
    ax_c.set_xlim(0, min(max_assist, 150))
    ax_c.set_ylim(-0.05, 1.05)
    ax_c.grid(True, alpha=0.2)

    # =========================================================================
    # Panel D: δ sweep — V_final satisfaction split by G1 vs G2
    # (Using ASSISTANT LENGTH as T for splitting)
    # =========================================================================
    ax_d = fig.add_subplot(gs[1, 1])
    delta_fine = np.geomspace(0.01, 100, 50)
    sat_g1_all, sat_g2_all = [], []
    n_g1_all, n_g2_all = [], []

    for delta in delta_fine:
        threshold = (1 + delta) * v_star
        tmin_f = t_min_tokens(v_ts_mean, gamma, tau, delta)
        tmin_i = max(1, int(math.ceil(tmin_f)))
        g1 = [r for r in seq_data if r["assist_len"] < tmin_i]
        g2 = [r for r in seq_data if r["assist_len"] >= tmin_i]
        sat_g1 = float(np.mean([r["V_final"] <= threshold for r in g1])) if g1 else float("nan")
        sat_g2 = float(np.mean([r["V_final"] <= threshold for r in g2])) if g2 else float("nan")
        sat_g1_all.append(sat_g1)
        sat_g2_all.append(sat_g2)
        n_g1_all.append(len(g1))
        n_g2_all.append(len(g2))

    ax_d.semilogx(delta_fine, sat_g2_all, "o-", color="#1f77b4", lw=2, markersize=3,
                  label="G2: $T_{assist} \\geq T_{min}$")
    ax_d.semilogx(delta_fine, sat_g1_all, "s--", color="#ff7f0e", lw=2, markersize=3,
                  label="G1: $T_{assist} < T_{min}$")
    ax_d.axhline(0, color="gray", lw=0.5)
    ax_d.axhline(1, color="gray", lw=0.5)
    ax_d.set_xlabel("$\\delta$ (log scale)")
    ax_d.set_ylabel("Tube satisfaction rate")
    ax_d.set_title("(D) δ sweep: V_final ≤ (1+δ)V* by group")
    ax_d.legend(fontsize=8)
    ax_d.set_ylim(-0.05, 1.05)
    ax_d.grid(True, alpha=0.2)

    # =========================================================================
    # Panel E: V_final vs Assistant Length (scatter + regression)
    # =========================================================================
    ax_e = fig.add_subplot(gs[2, 0])
    assist_lens = [r["assist_len"] for r in seq_data]
    v_finals = [r["V_final"] for r in seq_data]
    ax_e.scatter(assist_lens, v_finals, alpha=0.6, s=20, c="#1f77b4", edgecolors="none")

    # Linear fit
    if len(assist_lens) > 5:
        z = np.polyfit(assist_lens, v_finals, 1)
        p = np.poly1d(z)
        x_fit = np.linspace(min(assist_lens), max(assist_lens), 100)
        ax_e.plot(x_fit, p(x_fit), color="#d62728", lw=2, ls="--",
                  label=f"Linear fit (slope={z[0]:.5f})")
        corr = np.corrcoef(assist_lens, v_finals)[0, 1]
        ax_e.set_title(f"(E) V_final vs assistant length (r={corr:.3f})")
    else:
        ax_e.set_title("(E) V_final vs assistant length")

    ax_e.axhline(v_star, color="red", ls="--", lw=1, label=f"$V^*={v_star:.4f}$")
    ax_e.set_xlabel("Assistant span length (tokens)")
    ax_e.set_ylabel("$V_{final}$")
    ax_e.legend(fontsize=8)
    ax_e.grid(True, alpha=0.2)

    # =========================================================================
    # Panel F: Accuracy split by tube satisfaction (if accuracy available)
    # =========================================================================
    ax_f = fig.add_subplot(gs[2, 1])
    has_acc = any("Acc" in r and not math.isnan(r.get("Acc", float("nan"))) for r in seq_data)

    if has_acc:
        # Use a medium δ for the split
        delta_for_split = 30.0
        thr_split = (1 + delta_for_split) * v_star
        satisfied = [r for r in seq_data if r["V_final"] <= thr_split and not math.isnan(r.get("Acc", float("nan")))]
        not_satisfied = [r for r in seq_data if r["V_final"] > thr_split and not math.isnan(r.get("Acc", float("nan")))]

        acc_sat = [r["Acc"] for r in satisfied]
        acc_not = [r["Acc"] for r in not_satisfied]

        bars = []
        labels = []
        if acc_sat:
            bars.append(np.mean(acc_sat) * 100)
            labels.append(f"V_final ≤ thr\n(n={len(acc_sat)})")
        if acc_not:
            bars.append(np.mean(acc_not) * 100)
            labels.append(f"V_final > thr\n(n={len(acc_not)})")

        if bars:
            bar_colors = ["#2ca02c", "#d62728"][:len(bars)]
            ax_f.bar(range(len(bars)), bars, color=bar_colors, alpha=0.8)
            ax_f.set_xticks(range(len(bars)))
            ax_f.set_xticklabels(labels)
            ax_f.set_ylabel("Accuracy (%)")
            ax_f.set_title(f"(F) Accuracy: tube satisfied vs not (δ={delta_for_split:.0f})")
        else:
            ax_f.text(0.5, 0.5, "No accuracy data", ha="center", va="center", transform=ax_f.transAxes)
            ax_f.set_title("(F) Accuracy comparison")
    else:
        # Show V_final distribution instead
        ax_f.hist(v_finals, bins=20, color="#1f77b4", alpha=0.7, edgecolor="black", lw=0.5)
        ax_f.axvline(v_star, color="red", ls="--", lw=2, label=f"$V^*={v_star:.5f}$")
        for delta_mark in [10, 20, 30]:
            thr = (1 + delta_mark) * v_star
            ax_f.axvline(thr, color="gray", ls=":", lw=1, alpha=0.7)
            ax_f.text(thr, ax_f.get_ylim()[1] * 0.9, f"δ={delta_mark}", fontsize=7,
                      ha="center", rotation=90)
        ax_f.set_xlabel("$V_{final}$")
        ax_f.set_ylabel("Count")
        ax_f.set_title("(F) Distribution of V_final (with threshold markers)")
        ax_f.legend(fontsize=8)

    fig.suptitle(
        f"Tmin δ-Sweep Experiment: γ={gamma}, τ={tau}, V*={v_star:.5f}, "
        f"$\\bar{{V}}_{{ts}}$={v_ts_mean:.4f}, n={len(seq_data)}",
        fontsize=11, y=0.995,
    )
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  → Saved figure: {out_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Tmin δ-sweep experiment")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--eval_jsonl", type=str, default="datasets/gsm8k_test.jsonl")
    p.add_argument("--base_model_name", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    p.add_argument("--gamma", type=float, required=True)
    p.add_argument("--tau", type=float, required=True)
    p.add_argument("--max_examples", type=int, default=128)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--layer", type=int, default=-1)
    p.add_argument("--compute_accuracy", action="store_true")
    p.add_argument("--delta_plot", type=str, default="0.5,2,5,15,30",
                   help="Comma-separated δ values for position satisfaction plot")
    p.add_argument("--out_json", type=str, default="tmin_delta_sweep.json")
    p.add_argument("--out_plot", type=str, default="tmin_delta_sweep.png")
    return p.parse_args()


def main():
    args = parse_args()
    eval_path = Path(args.eval_jsonl)
    max_examples = None if args.max_examples <= 0 else int(args.max_examples)
    examples = load_examples(eval_path, max_examples=max_examples)
    if not examples:
        raise SystemExit(f"No examples in {eval_path}")

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

    # Phase 1: Extract geometry
    print("=" * 60)
    print("Phase 1: Extracting per-position geometry")
    print("=" * 60)
    seq_data = extract_full_geometry(
        model, tokenizer, examples,
        max_length=args.max_length, layer=args.layer,
    )
    if not seq_data:
        raise SystemExit("No valid sequences.")

    v_ts_mean = float(np.mean([r["V_ts"] for r in seq_data]))
    v_star = v_star_steady(args.gamma, args.tau)
    v_finals = [r["V_final"] for r in seq_data]

    print(f"\n  N sequences: {len(seq_data)}")
    print(f"  V* (theoretical) = {v_star:.6f}")
    print(f"  V_ts (mean @ a_s) = {v_ts_mean:.6f}")
    print(f"  V_final: min={min(v_finals):.4f}, med={np.median(v_finals):.4f}, max={max(v_finals):.4f}")
    assist_lens = [r["assist_len"] for r in seq_data]
    print(f"  Assistant lengths: min={min(assist_lens)}, med={int(np.median(assist_lens))}, max={max(assist_lens)}")

    # Phase 2: Accuracy (optional)
    if args.compute_accuracy:
        print("\n" + "=" * 60)
        print("Phase 2: Computing accuracy")
        print("=" * 60)
        compute_accuracy(
            model, tokenizer, examples, seq_data,
            base_model_name=args.base_model_name,
            max_length=args.max_length,
            max_new_tokens=args.max_new_tokens,
        )
        accs = [r["Acc"] for r in seq_data if not math.isnan(r.get("Acc", float("nan")))]
        if accs:
            print(f"  Accuracy: {np.mean(accs)*100:.1f}% ({sum(a == 1.0 for a in accs)}/{len(accs)})")

    # Phase 3: Analysis
    print("\n" + "=" * 60)
    print("Phase 3: δ sweep analysis")
    print("=" * 60)

    # Key δ values for detailed reporting
    key_deltas = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0]
    print(f"\n{'δ':>8} {'Tmin':>6} {'Threshold':>10} {'nG1':>5} {'nG2':>5} {'Sat_G1':>7} {'Sat_G2':>7} {'ΔSat':>7}")
    print("-" * 72)
    for delta in key_deltas:
        threshold = (1 + delta) * v_star
        tmin_f = t_min_tokens(v_ts_mean, args.gamma, args.tau, delta)
        tmin_i = max(1, int(math.ceil(tmin_f)))
        g1 = [r for r in seq_data if r["assist_len"] < tmin_i]
        g2 = [r for r in seq_data if r["assist_len"] >= tmin_i]
        sat_g1 = float(np.mean([r["V_final"] <= threshold for r in g1])) if g1 else float("nan")
        sat_g2 = float(np.mean([r["V_final"] <= threshold for r in g2])) if g2 else float("nan")
        gap = sat_g2 - sat_g1 if not (math.isnan(sat_g1) or math.isnan(sat_g2)) else float("nan")
        g1_str = f"{sat_g1:.3f}" if not math.isnan(sat_g1) else "  N/A"
        g2_str = f"{sat_g2:.3f}" if not math.isnan(sat_g2) else "  N/A"
        gap_str = f"{gap:+.3f}" if not math.isnan(gap) else "  N/A"
        print(f"{delta:8.2f} {tmin_i:6d} {threshold:10.6f} {len(g1):5d} {len(g2):5d} {g1_str:>7} {g2_str:>7} {gap_str:>7}")

    # Phase 4: Interpretation
    print("\n" + "=" * 60)
    print("Phase 4: Interpretation")
    print("=" * 60)
    # Find the δ where threshold matches median V_final
    median_vf = float(np.median(v_finals))
    delta_median = (median_vf / v_star) - 1
    tmin_at_median = t_min_tokens(v_ts_mean, args.gamma, args.tau, delta_median)
    print(f"\n  To reach median V_final ({median_vf:.4f}) as threshold:")
    print(f"    → Need δ = {delta_median:.1f}")
    print(f"    → T_min(δ) = {tmin_at_median:.1f} tokens")
    print(f"    → All sequences have assist_len ≥ {min(assist_lens)}, so ALL are in G2")

    # Correlation between V_final and assistant length
    corr = np.corrcoef(assist_lens, v_finals)[0, 1]
    print(f"\n  Correlation(assist_len, V_final) = {corr:.4f}")
    if corr < -0.1:
        print("    → Negative correlation: LONGER sequences have LOWER V_final (supports theory!)")
    elif corr > 0.1:
        print("    → Positive correlation: LONGER sequences have HIGHER V_final (opposite of theory)")
    else:
        print("    → No significant correlation between length and V_final")

    # Phase 5: Visualization
    print("\n" + "=" * 60)
    print("Phase 5: Generating visualization")
    print("=" * 60)
    delta_plot_list = [float(x) for x in args.delta_plot.split(",")]
    plot_comprehensive(seq_data, args.gamma, args.tau, delta_plot_list, Path(args.out_plot))

    # Save JSON results
    out_data = {
        "config": {
            "gamma": args.gamma,
            "tau": args.tau,
            "V_star": v_star,
            "V_ts_mean": v_ts_mean,
            "checkpoint": args.checkpoint,
            "n_sequences": len(seq_data),
        },
        "statistics": {
            "V_final_min": float(min(v_finals)),
            "V_final_median": float(np.median(v_finals)),
            "V_final_max": float(max(v_finals)),
            "assist_len_min": int(min(assist_lens)),
            "assist_len_max": int(max(assist_lens)),
            "corr_len_vfinal": float(corr),
            "delta_for_median_vfinal": float(delta_median),
            "Tmin_for_median_vfinal": float(tmin_at_median),
        },
    }
    Path(args.out_json).write_text(json.dumps(out_data, indent=2))
    print(f"  → Saved JSON: {args.out_json}")


if __name__ == "__main__":
    main()
