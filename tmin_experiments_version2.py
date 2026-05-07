"""
Minimum context length (T_min) experiment — Lyapunov tube corollary validation.

Implements `Tmin experiment.pdf` end-to-end:

- **Phase 1:** Derived ``T_min``, ``V^*`` from ``(gamma, tau, delta)`` plus ``V_ts``
  calibrated on a small forward-pass sample (assistant start anchor; pivot ``u_s`` gives
  degenerate ``V``).
- **Phase 3:** Split by sequence length ``T^{(i)} = t_e^{(i)} - t_s^{(i)}`` as
  ``a_e - u_s`` aligned with axis ``t - t_s``. If ``max(T) < ceil(T_min)``, an
  empirical split can populate both groups (**not** strict PDF); use
  ``--strict-theory-split`` for Phase-3 purity (G2 may be empty).
  **Matched control:** long sequences are truncated along the same trajectory so
  final offset ``< ceil(T_min)``.
  **Data:** Short synthetic shards rarely reach ``T_min``; **GSM8K** (long solutions)
  with ``--max-length`` large enough (e.g. 2048) is better for testing the PDF split.
  Preflight without GPU: ``--data-stats-only`` prints horizon quantiles vs ``T_min``.
- **Phase 4:** Per-seq ``V_final``, ``Sat``, ``Acc``; ``V_pred`` at the final token
  uses **per-sequence** ``V_ts^{(i)}`` at assistant start ``a_s``.
- **Phase 5–6:** Group table + two-panel figure. **Phase 6 left panel:** mean
  ``sqrt(V_t)`` (inside the ensemble mean), envelope ``sqrt(V_pred(k))``, line
  ``sqrt(V^*)`` — toggle ``--legacy-v-geometry-plot`` for the older log-scale
  ``mean V_t`` view.

Requires ``llm-jepa/dynamics_tube_loss.compute_V_all_stp`` on ``PYTHONPATH``

Run::

    python tmin_context_experiment.py --checkpoint DIR --eval-jsonl PATH \\
        --base-model NAME --gamma 0.95 --tau 1e-4 --delta 0.01

GSM8K example (longer horizons; raise ``--max-length`` if truncated)::

    python tmin_context_experiment.py ... \\
        --eval-jsonl ../../llm-jepa/datasets/gsm8k_test.jsonl --max-length 2048

Horizon check only (tokenizer + dataset, no checkpoint)::

    python tmin_context_experiment.py --data-stats-only \\
        --eval-jsonl ../../llm-jepa/datasets/gsm8k_test.jsonl \\
        --base-model NAME --max-length 2048 --v-ts 0.11
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# Repo imports: `test/llm-jepa` must precede `llm-jepa` on sys.path so
# `import evaluate` loads our `evaluate.py`, not HuggingFace's `evaluate` package.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_THIS_DIR = Path(__file__).resolve().parent
_LLM_JEPA = _REPO_ROOT / "llm-jepa"
sys.path.insert(0, str(_LLM_JEPA))
sys.path.insert(0, str(_THIS_DIR))

import evaluate as eval_mod  # noqa: E402
from dynamics_tube_loss import compute_V_all_stp  # noqa: E402


# ---------------------------------------------------------------------------
# Phase 1 — theory (hyperparameters only; no test leakage beyond calibration V_ts)
# ---------------------------------------------------------------------------


def v_star_steady(gamma: float, tau: float) -> float:
    """Target steady-state tube level V* = τ / (1 − γ)."""
    return tau / (1.0 - gamma)


def t_min_tokens(v_ts: float, gamma: float, tau: float, delta: float) -> float:
    """
    Minimum token-scale horizon from the research note:
    T_min = log(V_ts (1−γ) / (δ τ)) / log(1/γ).
    """
    if gamma <= 0.0 or gamma >= 1.0:
        raise ValueError("gamma must lie in (0, 1)")
    if tau <= 0.0 or delta <= 0.0:
        raise ValueError("tau and delta must be positive")
    num = v_ts * (1.0 - gamma) / (delta * tau)
    if num <= 1.0:
        return 0.0
    return math.log(num) / math.log(1.0 / gamma)


def v_pred_at_offset(
    k_offset: float, v_ts: float, gamma: float, tau: float
) -> float:
    """
    Discrete ISS-style envelope at offset k from anchor t_s (same k as x-axis):
    V_pred(k) = γ^k V_ts + V*(1 − γ^k).
    """
    if k_offset < 0:
        return float("nan")
    g = gamma ** k_offset
    v_star = v_star_steady(gamma, tau)
    return g * v_ts + v_star * (1.0 - g)


def tube_satisfied(
    v_final: float, delta: float, gamma: float, tau: float
) -> bool:
    """Sat = 1[ V_final ≤ (1+δ) V* ]."""
    v_star = v_star_steady(gamma, tau)
    return v_final <= (1.0 + delta) * v_star


def _collate_bounds(batch: List[dict]) -> dict:
    out: Dict[str, Any] = {}
    for k in batch[0].keys():
        vals = [b[k] for b in batch]
        if k in ("user_start_end", "assistant_start_end"):
            out[k] = torch.tensor(vals, dtype=torch.long)
        elif k == "input_ids":
            out[k] = torch.tensor(vals, dtype=torch.long)
        elif k == "labels":
            out[k] = torch.tensor(vals, dtype=torch.long)
        elif k == "attention_mask":
            out[k] = torch.tensor(vals, dtype=torch.long)
        else:
            out[k] = torch.tensor(vals)
    return out


def _horizon_a_e_minus_u_s(u_s: int, a_e: int) -> int:
    """
    Token offset from pivot ``u_s`` to last assistant index ``a_e`` (same units as
    x-axis ``k = t - t_s`` and ``k_fin`` in the theory envelope).
    """
    return a_e - u_s


def _collect_horizons_from_loader(
    loader: DataLoader,
    max_batches: Optional[int],
) -> List[int]:
    """Bounds-only scan: horizons in evaluation order (matches forward pass)."""
    horizons: List[int] = []
    n_batches = 0
    for batch in loader:
        usb = batch["user_start_end"]
        asb = batch["assistant_start_end"]
        bsz = usb.shape[0]
        for i in range(bsz):
            u_s = int(usb[i, 0].item()) + 1
            a_e = int(asb[i, 1].item())
            horizons.append(_horizon_a_e_minus_u_s(u_s, a_e))
        n_batches += 1
        if max_batches is not None and n_batches >= max_batches:
            break
    return horizons


def _choose_group_threshold(
    horizons: List[int],
    t_min_ceil: int,
    *,
    strict_theory: bool,
) -> Tuple[int, str, Dict[str, Any]]:
    """
    G1 / G2 split: **G1** if horizon ``< T``, **G2** if ``>= T``.

    Uses ``t_min_ceil`` when any sequence reaches that horizon; otherwise (typical
    for capped ``max_length`` eval sets) theory ``T_min`` exceeds all examples and
    we pick a data-driven threshold among **distinct** horizon lengths so both
    groups are populated whenever the horizon is not constant.
    """
    L = np.asarray(horizons, dtype=np.int64)
    mn = int(L.min())
    mx = int(L.max())
    meta: Dict[str, Any] = {
        "horizon_L_min": mn,
        "horizon_L_max": mx,
        "horizon_n": int(L.size),
    }
    if mx >= t_min_ceil:
        return t_min_ceil, "theory", meta
    if strict_theory:
        return t_min_ceil, "theory_data_shorter_than_T_min", {
            **meta,
            "note": (
                "No sequence reaches T_min; all examples are in G1 unless you "
                "disable strict_theory_split."
            ),
        }

    u = np.unique(L)
    if len(u) == 1:
        thr = int(u[0])
        # All sequences share one horizon H=thr: H < thr is impossible, so all are G2.
        return thr, "unreachable_tmin_constant_all_g2", {
            **meta,
            "note": (
                "Single horizon value; all examples assigned to G2 as maximal-length "
                "context in this eval set (G1 empty)."
            ),
        }

    # Pick a split between observed distinct lengths (median distinct value).
    j = max(1, min(len(u) // 2, len(u) - 1))
    thr = int(u[j])
    n_short = int((L < thr).sum())
    n_long = int((L >= thr).sum())
    if n_short == 0 or n_long == 0:
        thr = int(u[1])
        n_short = int((L < thr).sum())
        n_long = int((L >= thr).sum())
    if n_short == 0 or n_long == 0:
        thr = int(u[-2])
        n_short = int((L < thr).sum())
        n_long = int((L >= thr).sum())

    meta["split_threshold_candidates"] = u.tolist()
    meta["empirical_split_preview_g1"] = n_short
    meta["empirical_split_preview_g2"] = n_long
    return thr, "empirical_distinct_median", meta


def _sanitize_json_tree(obj: Any) -> Any:
    """Replace NaN/inf with None for strict JSON (and readable null in exports)."""
    if isinstance(obj, dict):
        return {k: _sanitize_json_tree(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_json_tree(v) for v in obj]
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
    return obj


_SQRT_CLIP = 1e-30


def _mean_curve(raw: DefaultDict[int, List[float]]) -> Tuple[np.ndarray, np.ndarray]:
    if not raw:
        return np.array([]), np.array([])
    ks = sorted(raw.keys())
    means = [float(np.nanmean(raw[k])) for k in ks]
    return np.array(ks, dtype=float), np.array(means, dtype=float)


def _mean_curve_sqrt_inside(raw: DefaultDict[int, List[float]]) -> Tuple[np.ndarray, np.ndarray]:
    """Phase 6 PDF: ensemble mean of ``sqrt(V)`` at fixed offset ``k`` (not ``sqrt(mean V)``)."""
    if not raw:
        return np.array([]), np.array([])
    ks = sorted(raw.keys())
    means = [
        float(
            np.nanmean(
                [math.sqrt(max(v, _SQRT_CLIP)) for v in raw[k]],
            ),
        )
        for k in ks
    ]
    return np.array(ks, dtype=float), np.array(means, dtype=float)


def _teacher_forced_accuracy_row(
    logits: torch.Tensor,
    labels: torch.Tensor,
    a_s: int,
    a_e: int,
) -> float:
    """Fraction of correct next-token preds on assistant span [a_s, a_e]."""
    # logits[t] predicts token at t+1; compare to labels[t+1]
    shift_logits = logits[0, :-1]
    shift_labels = labels[0, 1:]
    mask = torch.zeros_like(shift_labels, dtype=torch.bool)
    lo = max(0, a_s - 1)
    hi = min(shift_labels.numel() - 1, a_e - 1)
    if hi >= lo:
        mask[lo : hi + 1] = True
    mask &= shift_labels != -100
    if not mask.any():
        return float("nan")
    pred = shift_logits.argmax(dim=-1)
    ok = (pred == shift_labels) & mask
    return float(ok.sum().item() / mask.sum().item())


def run_tmin_experiment(
    checkpoint_dir: str,
    eval_jsonl: str,
    base_model_name: str,
    *,
    gamma: float,
    tau: float,
    delta: float,
    dataset_linear: str = "dynamics",
    max_length: int = 512,
    batch_size: int = 2,
    embedding_layer: int = -1,
    max_batches: Optional[int] = None,
    predictors: int = 0,
    unmask_assistant_special_tokens: bool = False,
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
    device_map: str = "auto",
    calibration_max_examples: int = 256,
    v_ts_override: Optional[float] = None,
    strict_theory_split: bool = False,
    legacy_v_geometry_plot: bool = False,
) -> Dict[str, Any]:
    """
    Phase 3–5: forward passes, group by horizon vs. split threshold, fill metrics.

    **Grouping:** Compared to ``split_threshold`` (usually ``ceil(T_min)``). Horizon is
    ``a_e - u_s`` — same units as the plot axis ``k = t - t_s``. If no sequence reaches
    ``T_min`` (common with ``max_length`` truncation), we split by an empirical
    threshold among distinct horizons unless ``strict_theory_split`` is True.

    **V_ts calibration:** V at the user pivot ``u_s`` is identically 0 in STP geometry
    (``d_{u_s}=0``), so it cannot estimate an initial tube radius. Following the note,
    we estimate ``V_ts`` as the mean of ``V`` at the **first assistant token** ``a_s``
    (offset ``k = a_s - u_s``), where ``d`` is already aligned with the assistant span.
    Pass ``v_ts_override`` to use a fixed value (e.g. a published calibration).
    """
    model, tokenizer = eval_mod.load_model_and_tokenizer(
        checkpoint_dir,
        base_model_name,
        load_in_8bit=load_in_8bit,
        load_in_4bit=load_in_4bit,
        device_map=device_map,
    )

    ds = eval_mod.prepare_jepa_eval_dataset(
        eval_jsonl,
        tokenizer,
        base_model_name,
        max_length,
        dataset_linear=dataset_linear,
        predictors=predictors,
        unmask_assistant_special_tokens=unmask_assistant_special_tokens,
    )
    keep = ["input_ids", "labels", "attention_mask", "user_start_end", "assistant_start_end"]
    ds = ds.remove_columns([c for c in ds.column_names if c not in keep])
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=_collate_bounds)

    device = next(model.parameters()).device
    model.eval()

    # Pass 1: calibrate V_ts at first assistant token (V at u_s is degenerate)
    v_ts_samples: List[float] = []
    n_cal = 0
    if v_ts_override is None:
        with torch.no_grad():
            for batch in loader:
                batch = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                out = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    output_hidden_states=True,
                )
                layer_ix = (
                    embedding_layer
                    if embedding_layer >= 0
                    else len(out.hidden_states) + embedding_layer
                )
                h = out.hidden_states[layer_ix].float()
                bsz = h.shape[0]
                usb = batch["user_start_end"]
                asb = batch["assistant_start_end"]
                for i in range(bsz):
                    u_s = int(usb[i, 0].item()) + 1
                    u_e = int(usb[i, 1].item())
                    a_s = int(asb[i, 0].item()) + 1
                    a_e = int(asb[i, 1].item())
                    if a_e < a_s:
                        continue
                    ub = [(u_s, u_e)]
                    ab = [(a_s, a_e)]
                    V_all, _ = compute_V_all_stp(h[i : i + 1], ub, ab)
                    Tseq = int(V_all.shape[1])
                    if 0 <= a_s < Tseq:
                        v_ts_samples.append(float(V_all[0, a_s].item()))
                    n_cal += 1
                    if n_cal >= calibration_max_examples:
                        break
                if n_cal >= calibration_max_examples:
                    break

        if not v_ts_samples:
            raise RuntimeError(
                "No calibration V_ts samples; empty dataset or no valid assistant spans?"
            )

        v_ts_mean = float(np.mean(v_ts_samples))
    else:
        v_ts_mean = float(v_ts_override)
    t_min_float = t_min_tokens(v_ts_mean, gamma, tau, delta)
    t_min_thr = max(1, int(math.ceil(t_min_float)))
    v_star = v_star_steady(gamma, tau)

    horizons = _collect_horizons_from_loader(loader, max_batches)
    if not horizons:
        raise RuntimeError("No examples when scanning horizons; check dataset or max_batches.")
    split_thr, split_mode, split_diag = _choose_group_threshold(
        horizons, t_min_thr, strict_theory=strict_theory_split
    )

    theory = {
        "gamma": gamma,
        "tau": tau,
        "delta": delta,
        "v_ts_calibrated": v_ts_mean,
        "v_ts_anchor": "assistant_start_a_s",
        "v_ts_override": v_ts_override,
        "t_min": t_min_float,
        "t_min_ceil": t_min_thr,
        "split_threshold": split_thr,
        "split_mode": split_mode,
        "horizon_metric": "a_e_minus_u_s",
        "split_diagnostics": split_diag,
        "strict_theory_split": strict_theory_split,
        "v_star": v_star,
    }

    # Pass 2: per-sequence metrics + curve bins
    sums: Dict[str, DefaultDict[str, float]] = {
        "G1": defaultdict(float),
        "G2": defaultdict(float),
    }
    counts: Dict[str, DefaultDict[str, int]] = {
        "G1": defaultdict(int),
        "G2": defaultdict(int),
    }
    # offset k -> list of V values
    curve_g2: DefaultDict[int, List[float]] = defaultdict(list)
    curve_g1_natural: DefaultDict[int, List[float]] = defaultdict(list)
    curve_g1_trunc: DefaultDict[int, List[float]] = defaultdict(list)

    n_seq = {"G1": 0, "G2": 0}

    n_batches = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="t_min_eval"):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                output_hidden_states=True,
            )
            logits = out.logits
            layer_ix = (
                embedding_layer
                if embedding_layer >= 0
                else len(out.hidden_states) + embedding_layer
            )
            h = out.hidden_states[layer_ix].float()
            bsz = h.shape[0]
            usb = batch["user_start_end"]
            asb = batch["assistant_start_end"]
            labels = batch["labels"]

            for i in range(bsz):
                u_s = int(usb[i, 0].item()) + 1
                u_e = int(usb[i, 1].item())
                a_s = int(asb[i, 0].item()) + 1
                a_e = int(asb[i, 1].item())
                horizon = _horizon_a_e_minus_u_s(u_s, a_e)
                group = "G1" if horizon < split_thr else "G2"

                ub = [(u_s, u_e)]
                ab = [(a_s, a_e)]
                V_all, _ = compute_V_all_stp(h[i : i + 1], ub, ab)
                vac = V_all[0]

                v_fin = float(vac[a_e].item())
                v_ts_i = float(vac[a_s].item())
                k_fin = float(a_e - u_s)
                v_pred_f = v_pred_at_offset(k_fin, v_ts_i, gamma, tau)
                err = abs(v_pred_f - v_fin)
                sat = tube_satisfied(v_fin, delta, gamma, tau)
                acc = _teacher_forced_accuracy_row(
                    logits[i : i + 1], labels[i : i + 1], a_s, a_e
                )

                g = group
                n_seq[g] += 1
                sums[g]["v_final"] += v_fin
                sums[g]["v_pred_err"] += err
                sums[g]["sat"] += 1.0 if sat else 0.0
                if not math.isnan(acc):
                    sums[g]["acc"] += acc
                    counts[g]["acc"] += 1

                # Curves: offset k = t - u_s along user ∪ assistant (STP/Lyapunov span)
                Tseq = int(vac.shape[0])
                for t in range(u_s, min(Tseq, a_e + 1)):
                    k = t - u_s
                    vk = float(vac[t].item())
                    if group == "G2":
                        curve_g2[k].append(vk)
                    else:
                        curve_g1_natural[k].append(vk)

                # Phase 3 matched control (PDF): truncate long G2 arcs to horizon < ceil(T_min)
                if group == "G2" and t_min_thr >= 2:
                    end_trunc_ctrl = min(a_e, u_s + t_min_thr - 1)
                    for t in range(u_s, min(Tseq, end_trunc_ctrl + 1)):
                        k = t - u_s
                        curve_g1_trunc[k].append(float(vac[t].item()))

            n_batches += 1
            if max_batches is not None and n_batches >= max_batches:
                break

    k_g2, v_g2 = _mean_curve(curve_g2)
    # dashed = natural short + truncated long (pool note "short, truncated")
    merged_dashed: DefaultDict[int, List[float]] = defaultdict(list)
    for d in (curve_g1_natural, curve_g1_trunc):
        for k, vs in d.items():
            merged_dashed[k].extend(vs)
    k_g1, v_g1 = _mean_curve(merged_dashed)

    k_sqrt_g2, sqrt_g2 = _mean_curve_sqrt_inside(curve_g2)
    merged_sqrt: DefaultDict[int, List[float]] = defaultdict(list)
    for d in (curve_g1_natural, curve_g1_trunc):
        for k2, vs in d.items():
            merged_sqrt[k2].extend(vs)
    k_sqrt_g1, sqrt_g1 = _mean_curve_sqrt_inside(merged_sqrt)

    def group_stats(name: str) -> Dict[str, Any]:
        n = n_seq[name]
        if n == 0:
            return {
                "n": 0,
                "mean_V_final": None,
                "mean_abs_vpred_minus_vfinal": None,
                "mean_sat_rate": None,
                "mean_accuracy": None,
            }
        c_acc = counts[name]["acc"]
        return {
            "n": n,
            "mean_V_final": sums[name]["v_final"] / n,
            "mean_abs_vpred_minus_vfinal": sums[name]["v_pred_err"] / n,
            "mean_sat_rate": sums[name]["sat"] / n,
            "mean_accuracy": (sums[name]["acc"] / c_acc) if c_acc else None,
        }

    table = {"G1": group_stats("G1"), "G2": group_stats("G2")}

    # Theory curve for plotting (same k grid as empirical max)
    k_max = int(
        max(
            np.max(k_g2) if len(k_g2) else 0,
            np.max(k_g1) if len(k_g1) else 0,
            split_thr + 2,
            t_min_thr + 2,
        )
    )
    k_theory = np.arange(0, k_max + 1, dtype=float)
    v_theory = np.array([v_pred_at_offset(float(k), v_ts_mean, gamma, tau) for k in k_theory])

    sqrt_v_theory = np.sqrt(np.maximum(v_theory, 0.0))

    protocol: Dict[str, Any] = {
        "pdf_source": "Tmin experiment.pdf",
        "phase3_T_definition": "T(i) = a_e - u_s (token offsets along t_ts = user pivot)",
        "phase4_V_pred_final": (
            "V_pred_at_offset(k_fin, gamma, tau) uses per-sequence V_ts^(i)=V[a_s]"
        ),
        "phase6_left_panel_y": (
            "mean_i sqrt(V^(i)_t) vs sqrt(envelope(k)); sqrt(V*) reference (PDF)"
        ),
        "phase3_truncation_end": (
            "G2 arcs truncated at min(a_e, u_s + ceil(T_min) - 1) for dashed pooled G1"
        ),
    }
    if split_mode != "theory":
        protocol["note_phase3_empirical_split"] = (
            "Sequences never reach ceil(T_min); G1/G2 use data-driven split_threshold "
            "so both groups populate. Strict PDF grouping: rerun with "
            "--strict-theory-split (G2 empty if horizons stay below T_min)."
        )

    result: Dict[str, Any] = {
        "protocol": protocol,
        "theory": theory,
        "table": table,
        "curves": {
            "k_g2": k_g2.tolist(),
            "mean_V_g2": v_g2.tolist(),
            "k_sqrt_g2": k_sqrt_g2.tolist(),
            "mean_sqrt_V_g2": sqrt_g2.tolist(),
            "k_g1_dashed": k_g1.tolist(),
            "mean_V_g1_dashed": v_g1.tolist(),
            "k_sqrt_g1_dashed": k_sqrt_g1.tolist(),
            "mean_sqrt_V_g1_dashed": sqrt_g1.tolist(),
            "k_theory": k_theory.tolist(),
            "v_theory": v_theory.tolist(),
            "sqrt_v_theory": sqrt_v_theory.tolist(),
        },
        "checkpoint": checkpoint_dir,
        "eval_jsonl": eval_jsonl,
    }
    result["plot_options"] = {"legacy_v_geometry_plot": legacy_v_geometry_plot}
    return result


def plot_tmin_main_figure(
    result: Dict[str, Any],
    save_path: Path,
    *,
    acc_override: Optional[Dict[str, float]] = None,
) -> Path:
    """
    Phase 6 main figure: left = geometry, right = grouped accuracy + mean V_final.

    acc_override: optional ``{"G1": x, "G2": y}`` to replace teacher-forced bars
    (e.g. published exact-match numbers).
    """
    theory = result["theory"]
    table = result["table"]
    curves = result["curves"]
    gamma = theory["gamma"]
    tau = theory["tau"]
    v_star = theory["v_star"]
    t_min = theory["t_min"]
    split_thr = theory.get("split_threshold", theory["t_min_ceil"])
    split_mode = theory.get("split_mode", "theory")
    use_emp_labels = split_mode not in ("theory", "theory_data_shorter_than_T_min")

    def _as_float(x: Any) -> float:
        if x is None:
            return float("nan")
        return float(x)

    fig, (ax_l, ax_r) = plt.subplots(
        1,
        2,
        figsize=(12.5, 5.0),
        dpi=300,
        gridspec_kw={"width_ratios": [1.45, 1.0], "wspace": 0.35},
    )

    legacy_v = bool(result.get("plot_options", {}).get("legacy_v_geometry_plot"))
    if not legacy_v and "mean_sqrt_V_g2" not in curves:
        legacy_v = True  # backward compat older JSON exports

    k2 = np.array(
        curves["k_g2" if legacy_v else "k_sqrt_g2"],
        dtype=float,
    )
    y2 = np.array(
        curves["mean_V_g2" if legacy_v else "mean_sqrt_V_g2"],
        dtype=float,
    )
    k1 = np.array(
        curves["k_g1_dashed" if legacy_v else "k_sqrt_g1_dashed"],
        dtype=float,
    )
    y1 = np.array(
        curves["mean_V_g1_dashed" if legacy_v else "mean_sqrt_V_g1_dashed"],
        dtype=float,
    )
    kt = np.array(curves["k_theory"], dtype=float)
    yt = np.array(
        curves["v_theory" if legacy_v else "sqrt_v_theory"],
        dtype=float,
    )

    g2_lab = "Empirical G2 ($T \\geq T_{\\mathrm{split}}$)" if use_emp_labels else "Empirical G2 ($T \\geq T_{\\min}$)"
    g1_lab = "Empirical G1 (short / truncated)" if not use_emp_labels else "Empirical G1 ($T < T_{\\mathrm{split}}$)"
    ax_l.plot(k2, y2, color="#1f77b4", lw=2.0, label=g2_lab)
    ax_l.plot(k1, y1, color="#1f77b4", lw=2.0, ls="--", label=g1_lab)
    thr_line = math.sqrt(max(v_star, 0.0)) if not legacy_v else v_star
    lbl_thr = "$\\sqrt{V^*}$" if not legacy_v else "$V^* = \\tau/(1-\\gamma)$"
    lbl_env = (
        r"$\sqrt{V_{\mathrm{pred}}(k)}$ envelope"
        if not legacy_v
        else r"$V_{\mathrm{pred}}(k)$ envelope"
    )
    ax_l.plot(kt, yt, color="#ff7f0e", lw=2.0, ls="--", label=lbl_env)
    ax_l.axhline(thr_line, color="#d62728", lw=1.8, label=lbl_thr)
    ax_l.axvline(t_min, color="black", ls="--", lw=1.2, alpha=0.7, label="$T_{\\min}$")

    if legacy_v:
        ax_l.set_yscale("log")
        ax_l.set_ylabel("Mean $V_t$ (log scale)")
    else:
        ax_l.set_yscale("linear")
        ax_l.set_ylabel(r"Mean $\sqrt{V_t}$ (PDF Phase 6)")
        vmax = thr_line
        for arr in (y2, y1, yt):
            if getattr(arr, "size", 0):
                vmax = float(max(vmax, float(np.nanmax(arr))))
        ax_l.set_ylim(0.0, min(1.15, vmax * 1.12))

    ax_l.set_xlabel("Token offset $t - t_s$")
    ax_l.set_title("Geometric validation")
    ax_l.grid(True, which="major", alpha=0.25)
    ax_l.legend(loc="upper right", fontsize=8)

    g1a = acc_override.get("G1") if acc_override else table["G1"]["mean_accuracy"]
    g2a = acc_override.get("G2") if acc_override else table["G2"]["mean_accuracy"]
    g1v = _as_float(table["G1"]["mean_V_final"])
    g2v = _as_float(table["G2"]["mean_V_final"])
    n1, n2 = int(table["G1"]["n"]), int(table["G2"]["n"])

    x = np.array([0, 1])
    w = 0.35
    h1 = (
        g1a * 100.0
        if (g1a is not None and g1a == g1a and n1 > 0)
        else 0.0
    )
    h2 = (
        g2a * 100.0
        if (g2a is not None and g2a == g2a and n2 > 0)
        else 0.0
    )
    bars = ax_r.bar(
        x - w / 2,
        [h1, h2],
        width=w,
        color=["#aec7e8", "#1f77b4"],
        edgecolor="black",
        label="Accuracy (%)",
    )
    ax_r.set_xticks([0, 1])
    thr_tex = "T_{\\mathrm{split}}" if use_emp_labels else "T_{\\min}"
    ax_r.set_xticklabels([f"G1 ($T < {thr_tex}$)", f"G2 ($T \\geq {thr_tex}$)"])
    ax_r.set_ylabel("Accuracy (%)")
    ax_r.set_ylim(0.0, 105.0)
    ax_r.set_title("Performance")

    ax_r2 = ax_r.twinx()
    v_line = [g1v if g1v == g1v else float("nan"), g2v if g2v == g2v else float("nan")]
    ax_r2.plot(
        x,
        v_line,
        color="#ff7f0e",
        marker="o",
        lw=1.5,
        markersize=8,
        label="Mean $V_{\\mathrm{final}}$",
    )
    ax_r2.set_ylabel("Mean $V_{\\mathrm{final}}$")
    ax_r2.legend(loc="upper right", fontsize=8)
    labels: List[str] = []
    for name, h_acc, ni, acc_raw in (
        ("G1", h1, n1, g1a),
        ("G2", h2, n2, g2a),
    ):
        if ni == 0:
            labels.append("n=0")
        elif acc_raw is None or (isinstance(acc_raw, float) and acc_raw != acc_raw):
            labels.append("n/a")
        elif acc_override and name in acc_override:
            labels.append(f"{h_acc:.1f}*")
        else:
            labels.append(f"{h_acc:.1f}")
    ax_r.bar_label(bars, labels=labels, padding=2, fontsize=9)

    sup_parts = [f"$\\gamma={gamma}$, $\\tau={tau}$, $T_{{\\min}}={t_min:.1f}$"]
    if use_emp_labels:
        sup_parts.append(f"$T_{{\\mathrm{{split}}}}={split_thr}$ ({split_mode})")
    fig.suptitle(", ".join(sup_parts), fontsize=11, y=0.99)
    # tight_layout() warns (and can mis-layout) with twinx(); use fixed margins.
    fig.subplots_adjust(left=0.07, right=0.98, top=0.88, bottom=0.14, wspace=0.32)
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return save_path


def run_horizon_data_stats(
    eval_jsonl: str,
    base_model_name: str,
    *,
    max_length: int,
    dataset_linear: str,
    predictors: int,
    unmask_assistant_special_tokens: bool,
    max_batches: Optional[int],
    gamma: float,
    tau: float,
    delta: float,
    v_ts_hint: Optional[float],
) -> None:
    """
    No checkpoint required: tokenizes the eval jsonl like the full experiment and
    reports the distribution of ``T^(i) = a_e - u_s`` (PDF Phase 3 length).

    Use this before a long GPU run to see whether ``max_length`` is large enough
    that some sequences can reach ``ceil(T_min)`` (strict PDF split).
    """
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(base_model_name)
    ds = eval_mod.prepare_jepa_eval_dataset(
        eval_jsonl,
        tok,
        base_model_name,
        max_length,
        dataset_linear=dataset_linear,
        predictors=predictors,
        unmask_assistant_special_tokens=unmask_assistant_special_tokens,
    )
    keep = ["input_ids", "labels", "attention_mask", "user_start_end", "assistant_start_end"]
    ds = ds.remove_columns([c for c in ds.column_names if c not in keep])
    loader = DataLoader(ds, batch_size=32, shuffle=False, collate_fn=_collate_bounds)
    horizons = _collect_horizons_from_loader(loader, max_batches)
    if not horizons:
        print("No horizons collected (empty dataset?).")
        return
    L = np.asarray(horizons, dtype=np.int64)
    print(f"Dataset: {eval_jsonl}")
    print(f"Examples counted: {len(L)}  (max_length={max_length})")
    print(f"Horizon T = a_e - u_s  —  min={int(L.min())}  max={int(L.max())}  "
          f"median={int(np.median(L))}  p90={int(np.percentile(L, 90))}  "
          f"p95={int(np.percentile(L, 95))}")
    over = [50, 100, 128, 168, 200, 256, 300, 512]
    print("Fraction with horizon >= k:")
    for k in over:
        print(f"  >= {k:4d}: {100.0 * float((L >= k).mean()):5.1f}%")

    if v_ts_hint is not None:
        tm = t_min_tokens(v_ts_hint, gamma, tau, delta)
        thr = max(1, int(math.ceil(tm)))
        frac = float((L >= thr).mean())
        print(
            f"\nWith hypothetical V_ts={v_ts_hint:g} (same γ,τ,δ as experiment): "
            f"T_min≈{tm:.2f}, ceil(T_min)={thr}"
        )
        print(f"Fraction with horizon >= ceil(T_min): {100.0 * frac:.1f}%")
        if frac < 0.05:
            print(
                "  → Very few sequences reach T_min: increase --max-length and/or pick "
                "long-form data (e.g. GSM8K vs short synth)."
            )
    else:
        print(
            "\nPass --v-ts FLOAT with --data-stats-only to see overlap with ceil(T_min) "
            "(e.g. calibrated value ~0.11 from a prior run)."
        )


def main() -> None:
    p = argparse.ArgumentParser(description="T_min minimum-context Lyapunov tube experiment")
    p.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Fine-tuned checkpoint (not needed with --data-stats-only)",
    )
    p.add_argument("--eval-jsonl", type=str, required=True)
    p.add_argument("--base-model", type=str, required=True)
    p.add_argument("--gamma", type=float, default=0.95)
    p.add_argument("--tau", type=float, default=1e-4)
    p.add_argument("--delta", type=float, default=0.01)
    p.add_argument("--dataset-linear", type=str, default="dynamics")
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--max-batches", type=int, default=None)
    p.add_argument("--embedding-layer", type=int, default=-1)
    p.add_argument("--out-json", type=str, default="tmin_experiment_results.json")
    p.add_argument("--out-figure", type=str, default="tmin_main_figure.pdf")
    p.add_argument("--calibration-n", type=int, default=256)
    p.add_argument(
        "--v-ts",
        type=float,
        default=None,
        dest="v_ts",
        help="Optional fixed V_ts (skip empirical calibration at assistant start)",
    )
    p.add_argument(
        "--strict-theory-split",
        action="store_true",
        help="G1/G2 only use ceil(T_min) (no empirical split when data are shorter)",
    )
    p.add_argument(
        "--legacy-v-geometry-plot",
        action="store_true",
        help="Plot mean V_t log-scale (legacy); default matches PDF Phase 6 (mean sqrt(V_t))",
    )
    p.add_argument("--load-in-8bit", action="store_true")
    p.add_argument("--load-in-4bit", action="store_true")
    p.add_argument(
        "--data-stats-only",
        action="store_true",
        help=(
            "Only tokenize eval-jsonl and print horizon T=a_e-u_s distribution; "
            "no model load. Optional --v-ts prints overlap with ceil(T_min)."
        ),
    )
    args = p.parse_args()

    if args.data_stats_only:
        run_horizon_data_stats(
            args.eval_jsonl,
            args.base_model,
            max_length=args.max_length,
            dataset_linear=args.dataset_linear,
            predictors=0,
            unmask_assistant_special_tokens=False,
            max_batches=args.max_batches,
            gamma=args.gamma,
            tau=args.tau,
            delta=args.delta,
            v_ts_hint=args.v_ts,
        )
        return

    if not args.checkpoint:
        raise SystemExit("error: --checkpoint is required unless --data-stats-only")

    result = run_tmin_experiment(
        args.checkpoint,
        args.eval_jsonl,
        args.base_model,
        gamma=args.gamma,
        tau=args.tau,
        delta=args.delta,
        dataset_linear=args.dataset_linear,
        max_length=args.max_length,
        batch_size=args.batch_size,
        embedding_layer=args.embedding_layer,
        max_batches=args.max_batches,
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit,
        calibration_max_examples=args.calibration_n,
        v_ts_override=args.v_ts,
        strict_theory_split=args.strict_theory_split,
        legacy_v_geometry_plot=args.legacy_v_geometry_plot,
    )

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    safe = _sanitize_json_tree(result)
    with out_json.open("w") as f:
        json.dump(safe, f, indent=2, allow_nan=False)
    print(f"Wrote {out_json}")

    fig_path = plot_tmin_main_figure(result, Path(args.out_figure))
    print(f"Wrote figure {fig_path}")


if __name__ == "__main__":
    main()
