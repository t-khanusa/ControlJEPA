#!/usr/bin/env python3
"""
Minimum-context (Tmin) validation experiment from the "Tmin experiment.pdf" note.

Implements the six phases in one runnable script:
  1) derive theoretical Tmin / V* from (gamma, tau, delta, V_ts),
  2) load a trained checkpoint,
  3) split test examples by sequence length (< Tmin vs >= Tmin),
  4) measure per-sequence geometry metrics M1..M5,
  5) report group tables,
  6) export the two-panel figure.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import evaluate as eval_mod
from eval_testset import prediction_matches_gold


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
    """Locate content token span inside a tokenized full conversation.

    Returns (start_minus_1, end), matching the convention used in `stp.py`.
    """
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

    raise RuntimeError("Failed to locate message content span in tokenized sequence.")


def _model_device(model: torch.nn.Module) -> torch.device:
    if hasattr(model, "device"):
        return model.device
    return next(model.parameters()).device


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
    """Compute V-path and per-sequence geometry metrics for one chat example."""
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

    device = _model_device(model)
    iid = torch.tensor([input_ids], dtype=torch.long, device=device)
    am = torch.tensor([attention_mask], dtype=torch.long, device=device)
    out = model(input_ids=iid, attention_mask=am, output_hidden_states=True, use_cache=False)
    h = out.hidden_states[layer][0]  # (T, D)

    v_user = h[u_e] - h[u_s]
    v_geo = v_user + (h[a_e] - h[a_s])
    v2 = float((v_geo * v_geo).sum().item())
    if v2 <= eps:
        return None
    v_hat = v_geo / math.sqrt(v2)

    valid_idx = torch.cat(
        [
            torch.arange(u_s, u_e + 1, device=device, dtype=torch.long),
            torch.arange(a_s, a_e + 1, device=device, dtype=torch.long),
        ]
    )
    d = torch.zeros(valid_idx.numel(), h.shape[-1], dtype=h.dtype, device=device)
    in_user = valid_idx <= u_e
    if in_user.any():
        d[in_user] = h[valid_idx[in_user]] - h[u_s]
    if (~in_user).any():
        d[~in_user] = v_user + (h[valid_idx[~in_user]] - h[a_s])

    proj = (d @ v_hat).unsqueeze(-1) * v_hat.unsqueeze(0)
    e = d - proj
    e2 = (e * e).sum(dim=-1)
    v_path = (e2 / max(v2, eps)).detach().float().cpu().numpy().tolist()

    # Skip anchor point at u_s (typically exactly zero displacement).
    if len(v_path) <= 1:
        return None
    path = [float(x) for x in v_path[1:]]
    if not path:
        return None

    return {
        "V_path": path,
        "V_ts": float(path[0]),
        "V_final": float(path[-1]),
        "T": int(len(path)),
    }


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


def mean_curve(paths: Sequence[Sequence[float]]) -> np.ndarray:
    if not paths:
        return np.zeros((0,), dtype=np.float64)
    max_len = max(len(p) for p in paths)
    vals = []
    for t in range(max_len):
        tok_vals = [p[t] for p in paths if len(p) > t]
        vals.append(float(np.mean(tok_vals)) if tok_vals else np.nan)
    return np.asarray(vals, dtype=np.float64)


def compute_tmin(gamma: float, tau: float, delta: float, v_ts: float) -> int:
    if not (0.0 < gamma < 1.0):
        raise ValueError("gamma must be in (0, 1).")
    if tau <= 0:
        raise ValueError("tau must be > 0.")
    if not (0.0 < delta < 1.0):
        raise ValueError("delta must be in (0, 1).")
    if v_ts <= 0:
        raise ValueError("V_ts must be > 0.")

    arg = (v_ts * (1.0 - gamma)) / (delta * tau)
    if arg <= 1.0:
        return 0
    val = math.log(arg) / math.log(1.0 / gamma)
    return max(0, int(math.ceil(val)))


def plot_main_figure(
    *,
    out_path: Path,
    gamma: float,
    tmin: int,
    v_star: float,
    v_ts_theory: float,
    g2_paths: Sequence[Sequence[float]],
    matched_paths: Sequence[Sequence[float]],
    g1_summary: Dict[str, float],
    g2_summary: Dict[str, float],
) -> None:
    g2_curve = mean_curve(g2_paths)
    matched_curve = mean_curve(matched_paths)
    max_len = int(max(len(g2_curve), len(matched_curve), max(1, tmin + 1)))

    t = np.arange(1, max_len + 1, dtype=np.float64)
    bound = np.sqrt(np.maximum(v_star + np.power(gamma, t) * v_ts_theory, 0.0))

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    ax = axes[0]
    if len(g2_curve) > 0:
        ax.plot(np.arange(1, len(g2_curve) + 1), np.sqrt(np.maximum(g2_curve, 0.0)), color="tab:blue", linewidth=2.0, label="Empirical G2 (T >= Tmin)")
    if len(matched_curve) > 0:
        ax.plot(
            np.arange(1, len(matched_curve) + 1),
            np.sqrt(np.maximum(matched_curve, 0.0)),
            color="tab:blue",
            linestyle="--",
            linewidth=2.0,
            label="Matched G2 truncated (< Tmin)",
        )
    ax.plot(t, bound, color="tab:orange", linestyle="--", linewidth=2.0, label="Theoretical bound sqrt(V* + gamma^t V_ts)")
    ax.axhline(math.sqrt(max(v_star, 0.0)), color="tab:red", linewidth=1.8, label="Steady-state threshold sqrt(V*)")
    ax.axvline(max(1, tmin), color="black", linestyle="--", linewidth=1.2, label="t = Tmin")
    ax.set_xlabel("Token position t - ts")
    ax.set_ylabel("Mean sqrt(V_t)")
    ax.set_title("Geometric validation")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax2 = axes[1]
    labels = ["G1 (T < Tmin)", "G2 (T >= Tmin)"]
    x = np.arange(len(labels))
    acc_vals = [g1_summary["accuracy"], g2_summary["accuracy"]]
    if all(not math.isnan(v) for v in acc_vals):
        ax2.bar(x, acc_vals, width=0.55, color=["#a6cee3", "#1f78b4"], label="Accuracy")
        ax2.set_ylim(0.0, 1.05)
        ax2.set_ylabel("Accuracy")
    else:
        ax2.bar(x, [0.0, 0.0], width=0.55, color=["#d9d9d9", "#969696"], label="Accuracy (not computed)")
        ax2.set_ylim(0.0, 1.05)
        ax2.set_ylabel("Accuracy")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_title("Performance vs geometry")
    ax2.grid(True, axis="y", alpha=0.25)

    ax3 = ax2.twinx()
    v_vals = [g1_summary["mean_V_final"], g2_summary["mean_V_final"]]
    if all(not math.isnan(v) for v in v_vals):
        ax3.plot(x, v_vals, color="tab:red", marker="o", linewidth=2, label="Mean V_final")
        ax3.set_ylabel("Mean V_final")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def auto_matched_short_len(tmin: int, g1_rows: Sequence[Dict]) -> int:
    if tmin <= 1:
        return 1
    cap = tmin - 1
    if g1_rows:
        cap = min(cap, max(int(r["T"]) for r in g1_rows))
    return max(1, cap)


@torch.no_grad()
def estimate_vts_from_data(
    model: torch.nn.Module,
    tokenizer,
    examples: Sequence[Dict],
    *,
    max_length: int,
    layer: int,
    n_examples: int,
) -> float:
    vals: List[float] = []
    for ex in examples:
        g = geometry_for_example(
            model,
            tokenizer,
            ex["messages"],
            max_length=max_length,
            layer=layer,
        )
        if g is None:
            continue
        vals.append(float(g["V_ts"]))
        if len(vals) >= n_examples:
            break
    if not vals:
        raise RuntimeError("Could not estimate V_ts: no valid examples found.")
    return float(np.mean(np.asarray(vals, dtype=np.float64)))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate the Tmin minimum-context corollary.")
    p.add_argument("--checkpoint", type=str, required=True, help="HF checkpoint directory.")
    p.add_argument("--eval_jsonl", type=str, default="datasets/synth_test.jsonl")
    p.add_argument("--base_model_name", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    p.add_argument("--gamma", type=float, required=True, help="Contraction rate in (0, 1).")
    p.add_argument("--tau", type=float, required=True, help="ISS slack > 0.")
    p.add_argument("--delta", type=float, default=0.01, help="Transient tolerance in (0, 1).")
    p.add_argument("--vts_override", type=float, default=-1.0, help="If > 0, use this V_ts instead of estimating.")
    p.add_argument("--estimate_vts_examples", type=int, default=16, help="How many examples to estimate V_ts from.")
    p.add_argument("--max_examples", type=int, default=128, help="Max eval examples (None for all).")
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--layer", type=int, default=-1, help="Hidden-state layer index.")
    p.add_argument(
        "--accuracy_mode",
        choices=("exact_match", "none"),
        default="exact_match",
        help="Per-example Acc(i).",
    )
    p.add_argument(
        "--matched_short_len",
        type=int,
        default=0,
        help="Control truncation length (< Tmin). 0 = auto: min(Tmin-1, max G1 length).",
    )
    p.add_argument("--out_json", type=str, default="tmin_experiment_report.json")
    p.add_argument("--out_plot", type=str, default="tmin_experiment_main_figure.png")
    p.add_argument("--save_per_sequence", action="store_true", help="Include per-sequence metrics in JSON.")
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

    if args.vts_override > 0:
        v_ts_theory = float(args.vts_override)
    else:
        v_ts_theory = estimate_vts_from_data(
            model,
            tokenizer,
            examples,
            max_length=args.max_length,
            layer=args.layer,
            n_examples=max(1, args.estimate_vts_examples),
        )
    tmin = compute_tmin(args.gamma, args.tau, args.delta, v_ts_theory)
    v_star = args.tau / (1.0 - args.gamma)

    rows: List[Dict] = []
    eval_stem = eval_path.stem
    for idx, ex in enumerate(examples):
        geom = geometry_for_example(
            model,
            tokenizer,
            ex["messages"],
            max_length=args.max_length,
            layer=args.layer,
        )
        if geom is None:
            continue
        t_i = int(geom["T"])
        v_pred = v_star + (args.gamma ** t_i) * float(geom["V_ts"])
        sat = float(float(geom["V_final"]) <= (1.0 + args.delta) * v_star)
        acc = float("nan")
        if args.accuracy_mode == "exact_match":
            acc = exact_match_for_example(
                model,
                tokenizer,
                messages=ex["messages"],
                base_model_name=args.base_model_name,
                max_length=args.max_length,
                max_new_tokens=args.max_new_tokens,
                eval_stem=eval_stem,
            )
        rows.append(
            {
                "index": idx,
                "group": "G1" if t_i < tmin else "G2",
                "T": t_i,
                "V_ts": float(geom["V_ts"]),
                "V_final": float(geom["V_final"]),
                "V_pred": float(v_pred),
                "Sat": sat,
                "Acc": acc,
                "V_path": geom["V_path"],
            }
        )

    if not rows:
        raise SystemExit("No valid examples after span detection and geometry extraction.")

    g1_rows = [r for r in rows if r["group"] == "G1"]
    g2_rows = [r for r in rows if r["group"] == "G2"]
    g1_summary = summarize_group(g1_rows)
    g2_summary = summarize_group(g2_rows)

    if args.matched_short_len > 0:
        short_len = int(args.matched_short_len)
        if short_len >= tmin and tmin > 0:
            short_len = max(1, tmin - 1)
    else:
        short_len = auto_matched_short_len(tmin, g1_rows)

    matched_rows: List[Dict] = []
    for r in g2_rows:
        if r["T"] < short_len:
            continue
        vpath = r["V_path"][:short_len]
        v_final_short = float(vpath[-1])
        matched_rows.append(
            {
                "T_short": short_len,
                "V_ts": float(vpath[0]),
                "V_final": v_final_short,
                "V_pred": float(v_star + (args.gamma ** short_len) * float(vpath[0])),
                "Sat": float(v_final_short <= (1.0 + args.delta) * v_star),
                "Acc": r["Acc"],
                "V_path": vpath,
            }
        )
    matched_summary = summarize_group(matched_rows)

    out = {
        "phase1_theory": {
            "gamma": args.gamma,
            "tau": args.tau,
            "delta": args.delta,
            "V_ts_theory": v_ts_theory,
            "Tmin": tmin,
            "V_star": v_star,
            "ideal_tube_width_over_vgeo_norm": math.sqrt(max(v_star, 0.0)),
        },
        "data": {
            "checkpoint": args.checkpoint,
            "eval_jsonl": str(eval_path),
            "evaluated_examples": float(len(rows)),
            "invalid_or_skipped_examples": float(len(examples) - len(rows)),
            "accuracy_mode": args.accuracy_mode,
        },
        "groups": {
            "G1_T_less_Tmin": g1_summary,
            "G2_T_ge_Tmin": g2_summary,
        },
        "matched_control_from_G2_truncated": {
            "short_len": float(short_len),
            **matched_summary,
        },
    }
    if args.save_per_sequence:
        out["per_sequence"] = [
            {k: v for k, v in r.items() if k != "V_path"} for r in rows
        ]

    out_json = Path(args.out_json)
    out_json.write_text(json.dumps(out, indent=2))

    plot_main_figure(
        out_path=Path(args.out_plot),
        gamma=args.gamma,
        tmin=tmin,
        v_star=v_star,
        v_ts_theory=v_ts_theory,
        g2_paths=[r["V_path"] for r in g2_rows],
        matched_paths=[r["V_path"] for r in matched_rows],
        g1_summary=g1_summary,
        g2_summary=g2_summary,
    )

    print("=== Tmin Experiment Report ===")
    print(f"checkpoint           : {args.checkpoint}")
    print(f"eval_jsonl           : {eval_path}")
    print(f"V_ts(theory)         : {v_ts_theory:.6g}")
    print(f"Tmin                 : {tmin}")
    print(f"V* = tau/(1-gamma)   : {v_star:.6g}")
    print("--- Group summaries ---")
    print("G1 (T < Tmin):", g1_summary)
    print("G2 (T >= Tmin):", g2_summary)
    print(f"Matched control short length: {short_len}")
    print("Matched G2->short:", matched_summary)
    print(f"Wrote JSON: {out_json}")
    print(f"Wrote plot: {args.out_plot}")


if __name__ == "__main__":
    main()
