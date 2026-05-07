"""
Plot training-loss curves from one or more HuggingFace Trainer run directories.

Reads ``trainer_state.json`` from each run directory and draws two stacked
panels:

  * Top panel: NTP loss (``lm_loss``), plus the HF-aggregate ``loss`` as a
    dashed reference (this aggregate is scaled by ``gradient_accumulation_steps``
    because of the newer HF Trainer GA bug-fix path, so it is shown only as
    context, not as the minimized quantity).
  * Bottom panel: the auxiliary loss (``jepa_loss``) — its interpretation
    (STP cosine / LyapunovLoss / curvature) is read from the
    ``loss_kind`` tag emitted by ``RepresentationTrainer.compute_loss`` and
    surfaced in the legend / panel title. For ``loss_kind == "lyapunov"``
    the V_t normalization (``control_norm``) is added to the label.

Usage
-----
Single run::

    python plot_training_curves.py <run_dir>
    python plot_training_curves.py <run_dir> --out loss.png

Multiple runs overlaid (useful to compare regular / STP / control_JEPA)::

    python plot_training_curves.py ft-r-2e-5-82 ft-j-2e-5-0.02-0-82 ft-c-d_t-g0.9-t1e-4-2e-5-0.05-0-82 \\
        --out compare.png --title "Llama-3.2-1B-Instruct / synth / seed=82"

The function ``plot_training_curves([run_dir], save_path)`` is the library
entry point and is invoked automatically at the end of ``stp.py`` training.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PathLike = Union[str, os.PathLike]


_LOSS_KIND_LABEL = {
    "lyapunov": "LyapunovLoss",
    "lyapunov_reach": "LyapunovReachabilityLoss",
    "stp_cosine": "STP cosine",
    "curvature": "Curvature",
    "aux": "Auxiliary",
    "none": "Auxiliary (not logged)",
    "other": "Auxiliary",
}


def _load_log_history(run_dir: Path) -> List[Dict[str, Any]]:
    state_path = run_dir / "trainer_state.json"
    if not state_path.is_file():
        raise FileNotFoundError(
            f"trainer_state.json not found in {run_dir} (searched {state_path})"
        )
    with state_path.open("r") as f:
        state = json.load(f)
    history = state.get("log_history", [])
    if not history:
        raise RuntimeError(f"log_history is empty in {state_path}")
    return history


def _series(
    history: Iterable[Dict[str, Any]], key: str
) -> Tuple[List[int], List[float]]:
    xs: List[int] = []
    ys: List[float] = []
    for entry in history:
        if key not in entry:
            continue
        step = entry.get("step")
        if step is None:
            continue
        try:
            y = float(entry[key])
        except (TypeError, ValueError):
            continue
        xs.append(int(step))
        ys.append(y)
    return xs, ys


def _infer_tag(
    history: Iterable[Dict[str, Any]], key: str
) -> Optional[str]:
    # Search most-recent-first; these tags are emitted at every logging step
    # by RepresentationTrainer.compute_loss for linear in {stp, curvature,
    # control_JEPA}.
    history = list(history)
    for entry in reversed(history):
        val = entry.get(key)
        if val is not None:
            return str(val)
    return None


def _infer_lbd_used(history: Iterable[Dict[str, Any]]) -> Optional[float]:
    history = list(history)
    for entry in reversed(history):
        v = entry.get("lbd_used")
        if v is None:
            continue
        try:
            return float(v)
        except (TypeError, ValueError):
            continue
    return None


def _aux_label(
    loss_kind: Optional[str],
    control_norm: Optional[str],
    reach_alpha: Optional[float] = None,
    reach_beta: Optional[float] = None,
    progress_mode: Optional[str] = None,
) -> str:
    base = _LOSS_KIND_LABEL.get(loss_kind or "aux", "Auxiliary")
    if loss_kind == "lyapunov" and control_norm:
        return f"{base} (V_t norm = {control_norm})"
    if loss_kind == "lyapunov_reach" and (reach_alpha is not None or reach_beta is not None):
        a = "?" if reach_alpha is None else f"{float(reach_alpha):.2g}"
        b = "?" if reach_beta is None else f"{float(reach_beta):.2g}"
        pm_tag = f", mode={progress_mode}" if progress_mode else ""
        return f"{base} (alpha={a}, beta={b}{pm_tag})"
    return base


def plot_training_curves(
    run_dirs: Union[PathLike, Sequence[PathLike]],
    save_path: Optional[PathLike] = None,
    title: Optional[str] = None,
) -> Path:
    """Render NTP vs auxiliary loss from trainer_state.json log_history.

    Parameters
    ----------
    run_dirs : path or list of paths
        HuggingFace Trainer output directory, or list of such directories to
        overlay on a single figure.
    save_path : path, optional
        Output PNG path. If None, writes to ``<first_run>/loss_curves.png``
        (single run) or ``<cwd>/loss_curves.png`` (overlay).
    title : str, optional
        Figure title. Defaults to the run-dir basename for a single run, or
        "Training curves (N runs)" for an overlay.

    Returns
    -------
    Path
        Path of the written PNG.
    """
    # Normalize inputs.
    if isinstance(run_dirs, (str, os.PathLike)):
        run_paths: List[Path] = [Path(run_dirs)]
    else:
        run_paths = [Path(p) for p in run_dirs]
    if not run_paths:
        raise ValueError("run_dirs is empty")

    # Collect per-run series.
    per_run: List[Dict[str, Any]] = []
    for rp in run_paths:
        history = _load_log_history(rp)
        lm_x, lm_y = _series(history, "lm_loss")
        jepa_x, jepa_y = _series(history, "jepa_loss")
        total_x, total_y = _series(history, "loss")
        loss_kind = _infer_tag(history, "loss_kind")
        control_norm = _infer_tag(history, "control_norm")
        reach_alpha_s = _infer_tag(history, "reach_alpha")
        reach_beta_s = _infer_tag(history, "reach_beta")
        progress_mode = _infer_tag(history, "progress_mode")
        try:
            reach_alpha = float(reach_alpha_s) if reach_alpha_s is not None else None
        except (TypeError, ValueError):
            reach_alpha = None
        try:
            reach_beta = float(reach_beta_s) if reach_beta_s is not None else None
        except (TypeError, ValueError):
            reach_beta = None
        lbd_used = _infer_lbd_used(history)
        # Geometric diagnostics (only present for control_JEPA / reach_JEPA
        # runs trained with the upgraded logger). Any missing key yields an
        # empty series, which the plotter silently skips.
        mv_x, mv_y = _series(history, "mean_V")
        rv_x, rv_y = _series(history, "raw_viol")
        vr_x, vr_y = _series(history, "viol_rate")
        e2_x, e2_y = _series(history, "mean_e2_over_L2")
        pL_x, pL_y = _series(history, "mean_p_over_L")
        # Schedule-tracking diagnostics (reach_JEPA schedule_asym only).
        tau_x, tau_y = _series(history, "mean_tau")
        gap_x, gap_y = _series(history, "mean_progress_gap")
        lag_x, lag_y = _series(history, "lag_rate")
        has_geom = bool(
            mv_x or rv_x or vr_x or e2_x or pL_x or tau_x or gap_x or lag_x
        )
        per_run.append(
            dict(
                path=rp,
                name=rp.name,
                lm=(lm_x, lm_y),
                jepa=(jepa_x, jepa_y),
                total=(total_x, total_y),
                loss_kind=loss_kind,
                control_norm=control_norm,
                reach_alpha=reach_alpha,
                reach_beta=reach_beta,
                progress_mode=progress_mode,
                lbd_used=lbd_used,
                mean_V=(mv_x, mv_y),
                raw_viol=(rv_x, rv_y),
                viol_rate=(vr_x, vr_y),
                e2_over_L2=(e2_x, e2_y),
                p_over_L=(pL_x, pL_y),
                tau=(tau_x, tau_y),
                prog_gap=(gap_x, gap_y),
                lag_rate=(lag_x, lag_y),
                has_geom=has_geom,
            )
        )

    # Figure. Add a third panel for geometric diagnostics if any run has them.
    any_geom = any(r["has_geom"] for r in per_run)
    if any_geom:
        fig, (ax1, ax2, ax3) = plt.subplots(
            3, 1, figsize=(10, 9.0), sharex=True,
            gridspec_kw={"height_ratios": [1.0, 1.0, 1.2]},
        )
    else:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6.4), sharex=True)
        ax3 = None

    cmap = plt.get_cmap("tab10")
    for i, run in enumerate(per_run):
        color = cmap(i % 10)
        label_run = run["name"] if len(per_run) > 1 else None

        tx, ty = run["total"]
        if tx:
            ax1.plot(
                tx, ty,
                linestyle="--", color=color, alpha=0.35, linewidth=1.1,
                label=(f"{label_run} — HF total (x grad_accum)" if label_run else "HF total (x grad_accum)"),
            )
        lx, ly = run["lm"]
        if lx:
            ax1.plot(
                lx, ly,
                color=color, linewidth=1.5,
                label=(f"{label_run} — NTP" if label_run else "NTP (lm_loss)"),
            )

        jx, jy = run["jepa"]
        if jx:
            aux_lbl = _aux_label(
                run["loss_kind"],
                run["control_norm"],
                run.get("reach_alpha"),
                run.get("reach_beta"),
                run.get("progress_mode"),
            )
            if run["lbd_used"] is not None:
                aux_lbl = f"{aux_lbl} [λ={run['lbd_used']:.4g}]"
            ax2.plot(
                jx, jy,
                color=color, linewidth=1.5,
                label=(f"{label_run} — {aux_lbl}" if label_run else aux_lbl),
            )

        if ax3 is not None and run["has_geom"]:
            prefix = f"{label_run} — " if label_run else ""
            # Primary axis: mean_V and e2/L2 (dimensionless, both ~ O(1)).
            mv_xr, mv_yr = run["mean_V"]
            if mv_xr:
                ax3.plot(
                    mv_xr, mv_yr,
                    color=color, linewidth=1.6,
                    label=f"{prefix}mean V",
                )
            e2x, e2y = run["e2_over_L2"]
            if e2x:
                ax3.plot(
                    e2x, e2y,
                    color=color, linestyle="--", linewidth=1.2,
                    label=f"{prefix}e²/L² (transverse)",
                )
            pLx, pLy = run["p_over_L"]
            if pLx:
                ax3.plot(
                    pLx, pLy,
                    color=color, linestyle=":", linewidth=1.2,
                    label=f"{prefix}p/L (progress)",
                )
            # Schedule (target) curve, plotted in a lighter shade on same axis.
            tau_xr, tau_yr = run.get("tau", ([], []))
            if tau_xr:
                ax3.plot(
                    tau_xr, tau_yr,
                    color=color, linestyle=(0, (1, 2)), linewidth=1.0, alpha=0.55,
                    label=f"{prefix}tau_t (schedule target)",
                )
            # Secondary axis: violation rate [0,1], raw violation (signed),
            # and lag_rate / signed progress gap (schedule_asym only).
            vrx, vry = run["viol_rate"]
            rvx, rvy = run["raw_viol"]
            lagx, lagy = run.get("lag_rate", ([], []))
            gapx, gapy = run.get("prog_gap", ([], []))
            if vrx or rvx or lagx or gapx:
                if not hasattr(ax3, "_twin_axis"):
                    ax3._twin_axis = ax3.twinx()
                ax3b = ax3._twin_axis
                if vrx:
                    ax3b.plot(
                        vrx, vry,
                        color=color, linestyle="-.", linewidth=1.0, alpha=0.7,
                        label=f"{prefix}viol_rate (right)",
                    )
                if rvx:
                    ax3b.plot(
                        rvx, rvy,
                        color=color, marker=".", linestyle="none",
                        markersize=2.5, alpha=0.55,
                        label=f"{prefix}raw_viol (right)",
                    )
                if lagx:
                    ax3b.plot(
                        lagx, lagy,
                        color=color, linestyle=(0, (3, 1, 1, 1)),
                        linewidth=1.1, alpha=0.8,
                        label=f"{prefix}lag_rate (right)",
                    )
                if gapx:
                    ax3b.plot(
                        gapx, gapy,
                        color=color, marker="x", linestyle="none",
                        markersize=3.0, alpha=0.6,
                        label=f"{prefix}signed gap (right)",
                    )

    ax1.set_ylabel("NTP loss")
    ax1.grid(True, linestyle=":", alpha=0.5)
    ax1.legend(loc="upper right", fontsize=8)

    ax2.set_ylabel("Auxiliary loss\n(surrogate, plotted)")
    ax2.grid(True, linestyle=":", alpha=0.5)
    ax2.legend(loc="upper right", fontsize=8)

    if ax3 is not None:
        ax3.set_ylabel("Geometric diagnostics\n(what actually moves)")
        ax3.set_xlabel("Optimizer step")
        ax3.grid(True, linestyle=":", alpha=0.5)
        ax3.legend(loc="upper right", fontsize=7)
        ax3b = getattr(ax3, "_twin_axis", None)
        if ax3b is not None:
            ax3b.set_ylabel("Violation (0-1) / raw gap")
            ax3b.legend(loc="lower right", fontsize=7)
    else:
        ax2.set_xlabel("Optimizer step")

    if title is None:
        if len(per_run) == 1:
            title = f"Training curves — {per_run[0]['name']}"
        else:
            title = f"Training curves (overlay of {len(per_run)} runs)"
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    if save_path is None:
        if len(per_run) == 1:
            save_path = run_paths[0] / "loss_curves.png"
        else:
            save_path = Path.cwd() / "loss_curves.png"
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    return save_path


def main() -> None:
    p = argparse.ArgumentParser(
        description="Plot NTP / auxiliary-loss training curves from Trainer runs."
    )
    p.add_argument(
        "run_dirs",
        nargs="+",
        type=str,
        help="One or more Trainer output directories (each containing trainer_state.json).",
    )
    p.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output PNG path. Default: <run>/loss_curves.png (single) or ./loss_curves.png (overlay).",
    )
    p.add_argument("--title", type=str, default=None, help="Figure title.")
    args = p.parse_args()

    out = plot_training_curves(args.run_dirs, args.out, args.title)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
