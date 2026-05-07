#!/usr/bin/env python3
"""Transversal stability validation for Theorem 1.1.

This script reports both:
  1) The original theorem check using l_max = max softplus(violation),
  2) A stricter check using raw positive violation max(violation, 0).

Per sequence it computes:
  sigma^2 = Var_t[V_t], where V_t is measured on assistant span (excluding a_e),
  violation_t = V_{t+1} - gamma V_t - tau,
  l_max_softplus = max softplus(violation_t),
  l_max_raw      = max(violation_t, 0).

Bounds:
  bound_softplus = (tau + l_max_softplus)^2 / ((1-gamma)^2 (1+gamma)),
  bound_raw      = (tau + l_max_raw)^2 / ((1-gamma)^2 (1+gamma)).

The script exports:
  - per-sequence JSON,
  - validation figure,
  - markdown table with CIs for paper reporting.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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

from scripts.tmin_experiment import compute_V_all_vgeo, find_start_end, load_examples


@dataclass
class SeqStabilityResult:
    index: int
    assist_len: int
    v_ts: float
    v_final: float
    sigma2: float
    lyap_seq: float
    l_max_softplus: float
    l_max_raw: float
    bound_softplus: float
    bound_raw: float
    ratio_softplus: float
    ratio_raw: float
    sat_softplus: bool
    sat_raw: bool


def theorem_bound(gamma: float, tau: float, l_max: float) -> float:
    denom = ((1.0 - gamma) ** 2) * (1.0 + gamma)
    return ((tau + l_max) ** 2) / max(denom, 1e-30)


def bootstrap_ci_mean(
    x: np.ndarray,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Tuple[float, float]:
    if x.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    n = x.size
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = x[idx].mean(axis=1)
    lo = float(np.quantile(boots, alpha / 2.0))
    hi = float(np.quantile(boots, 1.0 - alpha / 2.0))
    return lo, hi


def bootstrap_ci_diff_means(
    x: np.ndarray,
    y: np.ndarray,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Tuple[float, float, float]:
    if x.size == 0 or y.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    nx = x.size
    ny = y.size
    ix = rng.integers(0, nx, size=(n_boot, nx))
    iy = rng.integers(0, ny, size=(n_boot, ny))
    boots = x[ix].mean(axis=1) - y[iy].mean(axis=1)
    est = float(np.mean(x) - np.mean(y))
    lo = float(np.quantile(boots, alpha / 2.0))
    hi = float(np.quantile(boots, 1.0 - alpha / 2.0))
    return est, lo, hi


@torch.no_grad()
def compute_sequence_stability(
    model: torch.nn.Module,
    tokenizer,
    messages: List[Dict],
    *,
    gamma: float,
    tau: float,
    max_length: int,
    layer: int,
) -> Optional[SeqStabilityResult]:
    if len(messages) < 3:
        return None

    full_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    tok = tokenizer(
        full_text,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_tensors=None,
        add_special_tokens=True,
    )
    input_ids: List[int] = tok["input_ids"]
    attention_mask: List[int] = [1] * len(input_ids)

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
    # Need at least 3 assistant positions to form variance and transitions.
    if a_e - a_s < 3:
        return None

    device = next(model.parameters()).device
    iid = torch.tensor([input_ids], dtype=torch.long, device=device)
    am = torch.tensor([attention_mask], dtype=torch.long, device=device)
    out = model(input_ids=iid, attention_mask=am, output_hidden_states=True, use_cache=False)
    h = out.hidden_states[layer][0].float()

    V_all = compute_V_all_vgeo(h, u_s, u_e, a_s, a_e)
    # Exclude endpoint a_e because V[a_e]=0 is tautological by construction.
    V = V_all[a_s:a_e]
    if V.numel() < 3:
        return None

    assist_len = int(V.numel())
    v_ts = float(V[0].item())
    v_final = float(V[-1].item())

    sigma2 = float(torch.var(V, unbiased=True).item())
    violation = V[1:] - gamma * V[:-1] - tau
    surrogate = F.softplus(violation)
    lyap_seq = float(surrogate.mean().item())
    raw_pos = violation.clamp(min=0.0)
    l_max_softplus = float(surrogate.max().item())
    l_max_raw = float(raw_pos.max().item())
    bound_softplus = float(theorem_bound(gamma, tau, l_max_softplus))
    bound_raw = float(theorem_bound(gamma, tau, l_max_raw))
    ratio_softplus = sigma2 / max(bound_softplus, 1e-30)
    ratio_raw = sigma2 / max(bound_raw, 1e-30)

    return SeqStabilityResult(
        index=-1,
        assist_len=assist_len,
        v_ts=v_ts,
        v_final=v_final,
        sigma2=sigma2,
        lyap_seq=lyap_seq,
        l_max_softplus=l_max_softplus,
        l_max_raw=l_max_raw,
        bound_softplus=bound_softplus,
        bound_raw=bound_raw,
        ratio_softplus=ratio_softplus,
        ratio_raw=ratio_raw,
        sat_softplus=(sigma2 <= bound_softplus),
        sat_raw=(sigma2 <= bound_raw),
    )


def summarize(results: List[SeqStabilityResult]) -> Dict:
    if not results:
        return {
            "n": 0,
            "satisfied_rate": float("nan"),
        }

    sigma2 = np.array([r.sigma2 for r in results], dtype=np.float64)
    bound_soft = np.array([r.bound_softplus for r in results], dtype=np.float64)
    bound_raw = np.array([r.bound_raw for r in results], dtype=np.float64)
    ratio_soft = np.array([r.ratio_softplus for r in results], dtype=np.float64)
    ratio_raw = np.array([r.ratio_raw for r in results], dtype=np.float64)
    sat_soft = np.array([r.sat_softplus for r in results], dtype=np.float64)
    sat_raw = np.array([r.sat_raw for r in results], dtype=np.float64)
    lmax_soft = np.array([r.l_max_softplus for r in results], dtype=np.float64)
    lmax_raw = np.array([r.l_max_raw for r in results], dtype=np.float64)
    lyap = np.array([r.lyap_seq for r in results], dtype=np.float64)
    lengths = np.array([r.assist_len for r in results], dtype=np.float64)
    log_ratio_soft = np.log10(np.clip(ratio_soft, 1e-30, None))
    log_ratio_raw = np.log10(np.clip(ratio_raw, 1e-30, None))
    lo_soft, hi_soft = bootstrap_ci_mean(log_ratio_soft)
    lo_raw, hi_raw = bootstrap_ci_mean(log_ratio_raw, seed=1)
    lo_sat_raw, hi_sat_raw = bootstrap_ci_mean(sat_raw, seed=2)
    lo_sat_soft, hi_sat_soft = bootstrap_ci_mean(sat_soft, seed=3)

    return {
        "n": int(len(results)),
        "satisfied_rate_softplus": float(np.mean(sat_soft)),
        "satisfied_rate_raw": float(np.mean(sat_raw)),
        "satisfied_rate_softplus_ci95": [lo_sat_soft, hi_sat_soft],
        "satisfied_rate_raw_ci95": [lo_sat_raw, hi_sat_raw],
        "mean_sigma2": float(np.mean(sigma2)),
        "median_sigma2": float(np.median(sigma2)),
        "mean_bound_softplus": float(np.mean(bound_soft)),
        "median_bound_softplus": float(np.median(bound_soft)),
        "mean_bound_raw": float(np.mean(bound_raw)),
        "median_bound_raw": float(np.median(bound_raw)),
        "mean_ratio_softplus": float(np.mean(ratio_soft)),
        "median_ratio_softplus": float(np.median(ratio_soft)),
        "mean_ratio_raw": float(np.mean(ratio_raw)),
        "median_ratio_raw": float(np.median(ratio_raw)),
        "mean_log10_ratio_softplus": float(np.mean(log_ratio_soft)),
        "mean_log10_ratio_softplus_ci95": [lo_soft, hi_soft],
        "mean_log10_ratio_raw": float(np.mean(log_ratio_raw)),
        "mean_log10_ratio_raw_ci95": [lo_raw, hi_raw],
        "mean_lmax_softplus": float(np.mean(lmax_soft)),
        "mean_lmax_raw": float(np.mean(lmax_raw)),
        "mean_lyap_seq": float(np.mean(lyap)),
        "mean_assist_len": float(np.mean(lengths)),
        "corr_len_sigma2": float(np.corrcoef(lengths, sigma2)[0, 1]) if len(results) > 2 else float("nan"),
    }


def plot_validation(
    results_a: List[SeqStabilityResult],
    summary_a: Dict,
    *,
    label_a: str,
    out_plot: Path,
    results_b: Optional[List[SeqStabilityResult]] = None,
    summary_b: Optional[Dict] = None,
    label_b: Optional[str] = None,
) -> None:
    has_compare = results_b is not None and summary_b is not None and label_b is not None
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), dpi=150)
    ax1, ax2, ax3, ax4 = axes.flatten()

    def unpack(results: List[SeqStabilityResult]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        sigma2 = np.array([r.sigma2 for r in results], dtype=np.float64)
        bounds_soft = np.array([r.bound_softplus for r in results], dtype=np.float64)
        ratios_soft = np.array([r.ratio_softplus for r in results], dtype=np.float64)
        ratios_raw = np.array([r.ratio_raw for r in results], dtype=np.float64)
        sat_raw = np.array([r.sat_raw for r in results], dtype=np.float64)
        return sigma2, bounds_soft, ratios_soft, ratios_raw, sat_raw

    sigma2_a, bounds_a, ratios_soft_a, ratios_raw_a, sat_raw_a = unpack(results_a)
    ax1.scatter(bounds_a, sigma2_a, s=18, alpha=0.6, label=label_a, color="#1f77b4")
    if has_compare:
        sigma2_b, bounds_b, ratios_soft_b, ratios_raw_b, sat_raw_b = unpack(results_b)
        ax1.scatter(bounds_b, sigma2_b, s=18, alpha=0.5, label=label_b, color="#ff7f0e")
    lim_hi = max(float(np.max(bounds_a)), float(np.max(sigma2_a)))
    if has_compare:
        lim_hi = max(lim_hi, float(np.max(bounds_b)), float(np.max(sigma2_b)))
    lim_hi = max(lim_hi, 1e-8)
    ax1.plot([1e-12, lim_hi], [1e-12, lim_hi], "k--", lw=1.2, label="sigma^2 = bound")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("Theorem bound")
    ax1.set_ylabel("Empirical sigma^2")
    ax1.set_title("Per-sequence theorem check")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.2)

    logs_soft_a = np.log10(np.clip(ratios_soft_a, 1e-30, None))
    logs_raw_a = np.log10(np.clip(ratios_raw_a, 1e-30, None))
    bins = np.linspace(min(-8.0, float(np.min(logs_raw_a)) - 0.2), 0.5, 45)
    ax2.hist(logs_soft_a, bins=bins, alpha=0.6, color="#1f77b4", label=f"{label_a} (softplus)")
    ax2.hist(logs_raw_a, bins=bins, alpha=0.4, color="#2ca02c", label=f"{label_a} (raw)")
    if has_compare:
        logs_soft_b = np.log10(np.clip(ratios_soft_b, 1e-30, None))
        logs_raw_b = np.log10(np.clip(ratios_raw_b, 1e-30, None))
        ax2.hist(logs_soft_b, bins=bins, alpha=0.35, color="#ff7f0e", label=f"{label_b} (softplus)")
        ax2.hist(logs_raw_b, bins=bins, alpha=0.25, color="#d62728", label=f"{label_b} (raw)")
    ax2.axvline(0.0, color="red", ls="--", lw=1.5, label="log10(ratio)=0")
    ax2.set_xlabel("log10(sigma^2 / bound)")
    ax2.set_ylabel("Count")
    ax2.set_title("Bound-tightness distribution (lower is tighter)")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.2)

    bar_labels = [f"{label_a}\nsoft", f"{label_a}\nraw"]
    bar_vals = [
        100.0 * summary_a["satisfied_rate_softplus"],
        100.0 * summary_a["satisfied_rate_raw"],
    ]
    colors = ["#1f77b4", "#2ca02c"]
    if has_compare:
        bar_labels += [f"{label_b}\nsoft", f"{label_b}\nraw"]
        bar_vals += [
            100.0 * summary_b["satisfied_rate_softplus"],
            100.0 * summary_b["satisfied_rate_raw"],
        ]
        colors += ["#ff7f0e", "#d62728"]
    ax3.bar(np.arange(len(bar_vals)), bar_vals, color=colors, alpha=0.85)
    ax3.set_xticks(np.arange(len(bar_vals)))
    ax3.set_xticklabels(bar_labels)
    ax3.set_ylim(0, 105)
    ax3.set_ylabel("Validation rate (%)")
    ax3.set_title("Validation rate: theorem vs strict bound")
    ax3.grid(True, axis="y", alpha=0.2)

    text_lines = [
        f"{label_a}: n={summary_a['n']}, soft_sat={summary_a['satisfied_rate_softplus']*100:.1f}%, raw_sat={summary_a['satisfied_rate_raw']*100:.1f}%",
        f"  mean sigma^2={summary_a['mean_sigma2']:.4e}",
        f"  mean bound soft/raw={summary_a['mean_bound_softplus']:.4e} / {summary_a['mean_bound_raw']:.4e}",
        f"  mean log10 ratio soft/raw={summary_a['mean_log10_ratio_softplus']:.3f} / {summary_a['mean_log10_ratio_raw']:.3f}",
    ]
    if has_compare:
        text_lines += [
            "",
            f"{label_b}: n={summary_b['n']}, soft_sat={summary_b['satisfied_rate_softplus']*100:.1f}%, raw_sat={summary_b['satisfied_rate_raw']*100:.1f}%",
            f"  mean sigma^2={summary_b['mean_sigma2']:.4e}",
            f"  mean bound soft/raw={summary_b['mean_bound_softplus']:.4e} / {summary_b['mean_bound_raw']:.4e}",
            f"  mean log10 ratio soft/raw={summary_b['mean_log10_ratio_softplus']:.3f} / {summary_b['mean_log10_ratio_raw']:.3f}",
        ]
    ax4.axis("off")
    ax4.text(
        0.02,
        0.95,
        "\n".join(text_lines),
        va="top",
        ha="left",
        family="monospace",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="whitesmoke", alpha=0.9),
    )
    ax4.set_title("Summary")

    fig.suptitle("Transversal Stability Validation (Theorem 1.1)", fontsize=13, y=0.99)
    fig.tight_layout()
    fig.savefig(out_plot, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def run_for_model(
    model,
    tokenizer,
    examples: List[Dict],
    *,
    gamma: float,
    tau: float,
    max_length: int,
    layer: int,
) -> List[SeqStabilityResult]:
    results: List[SeqStabilityResult] = []
    for i, ex in enumerate(examples):
        seq = compute_sequence_stability(
            model,
            tokenizer,
            ex["messages"],
            gamma=gamma,
            tau=tau,
            max_length=max_length,
            layer=layer,
        )
        if seq is None:
            continue
        seq.index = i
        results.append(seq)
    return results


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Transversal stability validation experiment")
    p.add_argument("--checkpoint", type=str, required=True, help="Trained checkpoint path")
    p.add_argument("--eval_jsonl", type=str, default="datasets/synth_test.jsonl")
    p.add_argument("--gamma", type=float, required=True)
    p.add_argument("--tau", type=float, required=True)
    p.add_argument("--max_examples", type=int, default=200)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--layer", type=int, default=-1)
    p.add_argument("--base_model", type=str, default="", help="Optional base model for comparison")
    p.add_argument("--out_json", type=str, default="transversal_stability_report.json")
    p.add_argument("--out_plot", type=str, default="transversal_stability_validation.png")
    p.add_argument("--out_table", type=str, default="transversal_stability_table.md")
    return p.parse_args()


def build_table(
    summary_a: Dict,
    *,
    label_a: str,
    summary_b: Optional[Dict] = None,
    label_b: Optional[str] = None,
) -> str:
    header = (
        "| Model | n | Soft Sat (%) | Raw Sat (%) | mean sigma^2 | "
        "mean bound soft | mean bound raw | mean log10 ratio soft | mean log10 ratio raw |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    )
    rows = []
    rows.append(
        f"| {label_a} | {summary_a['n']} | {100*summary_a['satisfied_rate_softplus']:.2f} | "
        f"{100*summary_a['satisfied_rate_raw']:.2f} | {summary_a['mean_sigma2']:.4e} | "
        f"{summary_a['mean_bound_softplus']:.4e} | {summary_a['mean_bound_raw']:.4e} | "
        f"{summary_a['mean_log10_ratio_softplus']:.3f} | {summary_a['mean_log10_ratio_raw']:.3f} |"
    )
    if summary_b is not None and label_b is not None:
        rows.append(
            f"| {label_b} | {summary_b['n']} | {100*summary_b['satisfied_rate_softplus']:.2f} | "
            f"{100*summary_b['satisfied_rate_raw']:.2f} | {summary_b['mean_sigma2']:.4e} | "
            f"{summary_b['mean_bound_softplus']:.4e} | {summary_b['mean_bound_raw']:.4e} | "
            f"{summary_b['mean_log10_ratio_softplus']:.3f} | {summary_b['mean_log10_ratio_raw']:.3f} |"
        )
    return header + "\n".join(rows) + "\n"


def main() -> None:
    args = parse_args()
    examples = load_examples(Path(args.eval_jsonl), max_examples=args.max_examples)
    if not examples:
        raise SystemExit(f"No examples found in {args.eval_jsonl}")

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_cuda = torch.cuda.is_available()
    dtype = torch.bfloat16 if use_cuda else torch.float32

    print(f"Loading trained model: {args.checkpoint}")
    model_a = AutoModelForCausalLM.from_pretrained(
        args.checkpoint,
        torch_dtype=dtype,
        device_map="auto" if use_cuda else None,
        trust_remote_code=True,
    )
    model_a.eval()

    print("Running theorem validation on trained model...")
    results_a = run_for_model(
        model_a,
        tokenizer,
        examples,
        gamma=args.gamma,
        tau=args.tau,
        max_length=args.max_length,
        layer=args.layer,
    )
    if not results_a:
        raise SystemExit("No valid sequences processed for trained model.")
    summary_a = summarize(results_a)
    ratio_soft_a = np.array([r.ratio_softplus for r in results_a], dtype=np.float64)
    ratio_raw_a = np.array([r.ratio_raw for r in results_a], dtype=np.float64)
    print(
        f"[trained] n={summary_a['n']} | soft_sat={summary_a['satisfied_rate_softplus']*100:.1f}% | "
        f"raw_sat={summary_a['satisfied_rate_raw']*100:.1f}% | "
        f"mean sigma2={summary_a['mean_sigma2']:.4e}"
    )

    results_b: Optional[List[SeqStabilityResult]] = None
    summary_b: Optional[Dict] = None
    if args.base_model:
        print(f"Loading base model: {args.base_model}")
        model_b = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=dtype,
            device_map="auto" if use_cuda else None,
            trust_remote_code=True,
        )
        model_b.eval()
        print("Running theorem validation on base model...")
        results_b = run_for_model(
            model_b,
            tokenizer,
            examples,
            gamma=args.gamma,
            tau=args.tau,
            max_length=args.max_length,
            layer=args.layer,
        )
        summary_b = summarize(results_b)
        ratio_soft_b = np.array([r.ratio_softplus for r in results_b], dtype=np.float64)
        ratio_raw_b = np.array([r.ratio_raw for r in results_b], dtype=np.float64)
        d_soft, d_soft_lo, d_soft_hi = bootstrap_ci_diff_means(
            np.log10(np.clip(ratio_soft_a, 1e-30, None)),
            np.log10(np.clip(ratio_soft_b, 1e-30, None)),
            seed=42,
        )
        d_raw, d_raw_lo, d_raw_hi = bootstrap_ci_diff_means(
            np.log10(np.clip(ratio_raw_a, 1e-30, None)),
            np.log10(np.clip(ratio_raw_b, 1e-30, None)),
            seed=43,
        )
        summary_a["delta_vs_base_log10_ratio_soft"] = [d_soft, d_soft_lo, d_soft_hi]
        summary_a["delta_vs_base_log10_ratio_raw"] = [d_raw, d_raw_lo, d_raw_hi]
        print(
            f"[base] n={summary_b['n']} | soft_sat={summary_b['satisfied_rate_softplus']*100:.1f}% | "
            f"raw_sat={summary_b['satisfied_rate_raw']*100:.1f}% | "
            f"mean sigma2={summary_b['mean_sigma2']:.4e}"
        )
        print(
            f"[delta trained-base] log10 ratio soft={d_soft:.3f} (95% CI {d_soft_lo:.3f},{d_soft_hi:.3f}), "
            f"raw={d_raw:.3f} (95% CI {d_raw_lo:.3f},{d_raw_hi:.3f})"
        )

    plot_validation(
        results_a,
        summary_a,
        label_a="trained",
        out_plot=Path(args.out_plot),
        results_b=results_b,
        summary_b=summary_b,
        label_b="base" if results_b is not None else None,
    )

    report = {
        "config": {
            "checkpoint": args.checkpoint,
            "base_model": args.base_model,
            "eval_jsonl": args.eval_jsonl,
            "gamma": args.gamma,
            "tau": args.tau,
            "max_examples": args.max_examples,
            "max_length": args.max_length,
            "layer": args.layer,
        },
        "trained_summary": summary_a,
        "base_summary": summary_b,
        "trained_sequences": [asdict(x) for x in results_a],
        "base_sequences": [asdict(x) for x in results_b] if results_b is not None else None,
    }
    Path(args.out_json).write_text(json.dumps(report, indent=2))
    Path(args.out_table).write_text(build_table(summary_a, label_a="trained", summary_b=summary_b, label_b="base" if summary_b else None))
    print(f"Wrote: {args.out_json}")
    print(f"Wrote: {args.out_plot}")
    print(f"Wrote: {args.out_table}")


if __name__ == "__main__":
    main()
