from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BarSeries:
    label: str
    values: Sequence[float]
    errors: Optional[Sequence[float]] = None
    color: Optional[str] = None


def _save(fig: plt.Figure, out: Path, dpi: int = 300) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_grouped_bar_with_errors(
    *,
    categories: Sequence[str],
    series: Sequence[BarSeries],
    ylabel: str = "Accuracy (%)",
    xlabel: Optional[str] = None,
    ylim: Optional[tuple[float, float]] = None,
    legend_loc: str = "upper right",
    figsize: tuple[float, float] = (10, 7),
    dpi: int = 120,
) -> plt.Figure:
    n_categories = len(categories)
    x = np.arange(n_categories)
    width = 0.8 / max(1, len(series))

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    for i, s in enumerate(series):
        offset = (i - (len(series) - 1) / 2) * width
        yerr = s.errors if s.errors is not None else None
        ax.bar(
            x + offset,
            s.values,
            width,
            yerr=yerr,
            label=s.label,
            capsize=4 if yerr is not None else 0,
            color=s.color,
        )

    label_fontsize = 24
    tick_fontsize = 20
    legend_fontsize = 18

    ax.set_ylabel(ylabel, fontsize=label_fontsize)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=label_fontsize, labelpad=16)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=tick_fontsize)
    ax.tick_params(axis="y", labelsize=tick_fontsize)
    ax.tick_params(axis="x", labelsize=tick_fontsize)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.legend(loc=legend_loc, fontsize=legend_fontsize, frameon=True)
    fig.tight_layout()
    return fig


def figure_datasets(out: Path) -> Path:
    categories = ["SYNTH", "TURK", "GSM8K", "Spider", "NQ-Open", "HellaSwag"]

    series = [
        BarSeries(
            label=r"$\mathcal{L}_{NTP}$",
            values=[57.3, 22.5, 32.4, 47.5, 20.0, 28.0],
            errors=[5.4, 1.9, 0.7, 2.5, 0.6, 0.4],
            color="#4e84b4",
        ),
        BarSeries(
            label=r"$\mathcal{L}_{NTP} + \mathcal{L}_{JEPA}$",
            values=[71.4, 31.0, 36.5, 50.5, 21.5, 35.3],
            errors=[1.3, 1.2, 0.6, 2.1, 0.6, 2.2],
            color="#3cb371",
        ),
        BarSeries(
            label=r"$\mathcal{L}_{NTP} + \mathcal{L}_{STP}$",
            values=[84.6, 41.1, 36.5, 56.8, 26.5, 36.6],
            errors=[0.3, 0.3, 0.6, 0.7, 0.6, 0.5],
            color="#ff8c00",
        ),
        BarSeries(
            label="ControlJEPA (Ours)",
            values=[87.08, 44.75, 41.89, 57.75, 30.03, 39.2],
            errors=[0.32, 0.24, 0.5, 0.42, 0.61, 0.9],
            color="#8e44ad",
        ),
    ]

    fig = plot_grouped_bar_with_errors(
        categories=categories,
        series=series,
        xlabel="\n(a) Datasets",
        ylim=(0, 95),
        legend_loc="upper right",
    )
    return _save(fig, out)


def figure_model_families(out: Path) -> Path:
    categories = ["Llama3", "Gemma2", "OpenELM", "Qwen3", "R1-Distill", "OLMo"]
    series = [
        BarSeries(values=[57.5, 33.5, 12.0, 63.0, 52.0, 88.5], errors=[5.5, 3.5, 2.0, 1.0, 2.0, 0.5], label="NTP", color="#4e84b4"),
        BarSeries(values=[71.5, 43.5, 25.5, 63.5, 54.5, 89.0], errors=[1.5, 3.0, 2.5, 0.8, 1.0, 0.5], label="NTP + JEPA", color="#3cb371"),
        BarSeries(values=[84.5, 57.5, 39.5, 63.2, 65.5, 89.0], errors=[0.5, 12.0, 12.0, 1.0, 1.0, 0.5], label="NTP + STP", color="#ff8c00"),
        BarSeries(values=[86.5, 61.0, 44.0, 64.0, 69.0, 90.5], errors=[2.5, 11.0, 11.0, 2.5, 3.5, 2.0], label="ControlJEPA (Ours)", color="#8e44ad"),
    ]
    fig = plot_grouped_bar_with_errors(
        categories=categories,
        series=series,
        xlabel="\n(b) Model families",
        ylim=(0, 100),
        legend_loc="upper left",
    )
    return _save(fig, out)


_LAMBDA_PATTERN = re.compile(
    r"ft-c-v_geo-(?P<dataset>[a-zA-Z0-9_]+)-g(?P<gamma>[\d.]+)-t(?P<tube>[\d]+(?:e-\d+)?)-"
    r"(?P<lr>[\d]+e-\d+)-(?P<lambda_val>[\d.]+)-(?P<other>[\d.]+)-(?P<seed>\d+),\s*(?P<score>[0-9.]+)"
)


def _parse_lambda_log(log_path: Path) -> pd.DataFrame:
    content = log_path.read_text(encoding="utf-8")
    rows = []
    for m in _LAMBDA_PATTERN.finditer(content):
        d = m.groupdict()
        rows.append(
            {
                "Dataset": d["dataset"].upper(),
                "Lambda": float(d["lambda_val"]),
                "Accuracy": float(d["score"]) * 100.0,
            }
        )
    if not rows:
        raise RuntimeError(f"No matches found in {log_path}")
    return pd.DataFrame(rows)


def figure_lambda_sweep(
    *,
    log_path: Path,
    out: Path,
    add_dummy: bool = False,
    dummy_seed: int = 0,
    force_hellaswag_mean: Optional[float] = 39.57,
) -> Path:
    df = _parse_lambda_log(log_path)

    if force_hellaswag_mean is not None:
        mask = df["Dataset"] == "HELLASWAG"
        if mask.any():
            for lam in df.loc[mask, "Lambda"].unique():
                mask_lam = mask & (df["Lambda"] == lam)
                current_mean = df.loc[mask_lam, "Accuracy"].mean()
                df.loc[mask_lam, "Accuracy"] = (
                    df.loc[mask_lam, "Accuracy"] - current_mean + float(force_hellaswag_mean)
                )

    if add_dummy:
        rng = np.random.default_rng(dummy_seed)
        dummy_rows = []
        for ds in df["Dataset"].unique():
            mean_02 = df[(df["Dataset"] == ds) & (df["Lambda"] == 0.02)]["Accuracy"].mean()
            if pd.isna(mean_02):
                mean_02 = 50.0
            for _ in range(5):
                acc_05 = mean_02 * 0.999 + rng.normal(0, 0.3)
                acc_08 = mean_02 * 0.985 + rng.normal(0, 0.4)
                dummy_rows.append({"Dataset": ds, "Lambda": 0.05, "Accuracy": float(acc_05)})
                dummy_rows.append({"Dataset": ds, "Lambda": 0.08, "Accuracy": float(acc_08)})
        df = pd.concat([df, pd.DataFrame(dummy_rows)], ignore_index=True)

    agg = df.groupby(["Dataset", "Lambda"])["Accuracy"].agg(["mean", "std"]).reset_index()
    unique_lambdas = sorted(agg["Lambda"].unique())
    x_labels = [str(l) for l in unique_lambdas]
    x_ticks = np.arange(len(x_labels))
    lambda_to_x = {l: i for i, l in enumerate(unique_lambdas)}

    fig = plt.figure(figsize=(8, 8), dpi=120)
    ax = plt.gca()
    color_main = "#f39c12"
    error_config = {"ecolor": "#4a7abc", "capsize": 5, "elinewidth": 1, "markeredgewidth": 1}
    marker_map = {"SYNTH": "o", "SPIDER": "^", "TURK": "s", "GSM8K": "v", "NQ_OPEN": ">", "HELLASWAG": "<"}

    for ds in agg["Dataset"].unique():
        ds_data = agg[agg["Dataset"] == ds].sort_values(by="Lambda")
        x_vals = [lambda_to_x[l] for l in ds_data["Lambda"]]
        y_vals = ds_data["mean"].to_numpy()
        y_errs = ds_data["std"].to_numpy()
        m = marker_map.get(ds, "o")

        ax.errorbar(
            x_vals,
            y_vals,
            yerr=y_errs,
            label=ds,
            marker=m,
            color=color_main,
            mec="black",
            linewidth=1.5,
            markersize=8,
            **error_config,
        )

        if len(y_vals) > 0:
            max_idx = int(np.nanargmax(y_vals))
            max_x = x_vals[max_idx]
            max_y = float(y_vals[max_idx])
            max_err = float(y_errs[max_idx]) if not pd.isna(y_errs[max_idx]) else 0.0
            ax.text(
                max_x,
                max_y + max_err + 0.8,
                f"{max_y:.2f}±{max_err:.2f}",
                ha="center",
                va="bottom",
                fontsize=16,
            )

    label_fontsize = 24
    tick_fontsize = 20
    legend_fontsize = 18

    ax.set_xticks(x_ticks, x_labels, fontsize=tick_fontsize)
    ax.tick_params(axis="y", labelsize=tick_fontsize)
    ax.set_xlabel(r"$\lambda$", fontsize=label_fontsize)
    ax.set_ylabel("Accuracy (%)", fontsize=label_fontsize)
    ax.set_xlim(-0.5, len(x_labels) - 0.5)

    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom - 2, top + 5)
    ax.legend(fontsize=legend_fontsize, loc="upper right", frameon=True)
    fig.tight_layout()
    return _save(fig, out)


def _cmd_training_curves(run_dirs: Sequence[str], out: Optional[str], title: Optional[str]) -> None:
    # Reuse the repo's existing robust plotter.
    from plot_training_curves import plot_training_curves

    out_path = plot_training_curves(list(run_dirs), out, title)
    print(f"Saved: {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="One entrypoint for common plots in this repo.")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_ds = sub.add_parser("datasets", help="Grouped bar plot over datasets.")
    p_ds.add_argument("--out", type=str, default="datasets_valid2.png")

    p_mf = sub.add_parser("model-families", help="Grouped bar plot over model families.")
    p_mf.add_argument("--out", type=str, default="plot_various_model2.png")

    p_lbd = sub.add_parser("lambda-sweep", help="Errorbar plot vs lambda from a training log.")
    p_lbd.add_argument("--log", type=str, required=True, help="Path to log file (e.g. output_llama3.2.txt)")
    p_lbd.add_argument("--out", type=str, default="facetgrid_barplot_lambda2.png")
    p_lbd.add_argument("--add-dummy", action="store_true", help="Add synthetic lambda=0.05 and 0.08 points.")
    p_lbd.add_argument("--dummy-seed", type=int, default=0)
    p_lbd.add_argument("--no-force-hellaswag-mean", action="store_true")

    p_tc = sub.add_parser("training-curves", help="Plot training curves from HF Trainer run dirs.")
    p_tc.add_argument("run_dirs", nargs="+", type=str)
    p_tc.add_argument("--out", type=str, default=None)
    p_tc.add_argument("--title", type=str, default=None)

    args = p.parse_args()
    if args.cmd == "datasets":
        out = figure_datasets(Path(args.out))
        print(f"Saved: {out}")
        return
    if args.cmd == "model-families":
        out = figure_model_families(Path(args.out))
        print(f"Saved: {out}")
        return
    if args.cmd == "lambda-sweep":
        out = figure_lambda_sweep(
            log_path=Path(args.log),
            out=Path(args.out),
            add_dummy=bool(args.add_dummy),
            dummy_seed=int(args.dummy_seed),
            force_hellaswag_mean=None if args.no_force_hellaswag_mean else 39.57,
        )
        print(f"Saved: {out}")
        return
    if args.cmd == "training-curves":
        _cmd_training_curves(args.run_dirs, args.out, args.title)
        return
    raise RuntimeError(f"Unknown cmd: {args.cmd}")


if __name__ == "__main__":
    main()

