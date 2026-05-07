#!/usr/bin/env python3
"""
Train one base LLM on one dataset under three regimes and compare validation perplexity (PPL).

Methods (aligned with run_stp.py naming intent):
  1. regular   — `stp.py --regular` (standard causal LM fine-tuning). Training
                 is skipped; included in PPL evaluation only if a checkpoint
                 directory already exists on disk.
  2. stp       — `stp.py --linear random_span --random_span_e2e` (same JEPA path as `run_stp.py` / paper-style STP aux).
  3. dynamics  — `stp.py --dynamics_tube` (Lyapunov tube + optional local TS term).

At the end of training, a figure `loss_curves.png` is saved under `output_root/`
showing L_NTP and the method-specific auxiliary loss (L_STP for method=stp;
L_Lyapunov for method=dynamics) vs. training steps, parsed from each run's
`trainer_state.json`.

PPL is exp(mean NLL) over all non-masked label positions (labels != -100), using the same
chat template / masking as each training path.

STP-style extras: --eval_accuracy, --eval_token_accuracy, --eval_snr_proxy;
--data_fraction for seeded train subsampling (data efficiency).

Examples
--------
  # Train all three (each run uses torchrun internally). Do NOT wrap this script in torchrun.
  python compare_three_method.py \\
      --model_name meta-llama/Llama-3.2-1B-Instruct \\
      --dataset_name synth --data_prefix datasets/ \\
      --num_epochs 1 --batch_size 2 --grad_accum 4 --nproc 2

  # Evaluate existing checkpoints only
  python compare_three_method.py --skip_train \\
      --model_name meta-llama/Llama-3.2-1B-Instruct \\
      --dataset_name synth --data_prefix datasets/ \\
      --ckpt_regular ./out-regular --ckpt_stp ./out-stp --ckpt_dynamics ./out-dynamics
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import socket
import subprocess
import sys
from pathlib import Path

import eval_testset as stp_eval_metrics
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling



# Project root (directory containing this file)
ROOT = Path(__file__).resolve().parent


def _pick_free_port() -> int:
    """Bind to port 0 to get an ephemeral free port (avoids EADDRINUSE on default 29500)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _import_stp_dataset():
    """Lazy import so --help works without full training deps in some envs."""
    import stp  # noqa: WPS433 — intentional side-effect import of project module

    return stp.load_and_prepare_dataset


def build_eval_dataset(
    eval_jsonl: Path,
    tokenizer,
    model_name: str,
    max_length: int,
    method: str,
    predictors: int = 0,
):
    """
    method in {"regular", "stp", "dynamics"} — must match how the checkpoint was trained.
    """
    load_and_prepare_dataset = _import_stp_dataset()
    if method == "regular":
        return load_and_prepare_dataset(
            str(eval_jsonl),
            tokenizer,
            model_name,
            max_length=max_length,
            predictors=predictors,
            regular=True,
            linear=None,
        )
    if method == "dynamics":
        linear = "dynamics"
    elif method == "stp":
        linear = "random_span"
    else:
        linear = None
    return load_and_prepare_dataset(
        str(eval_jsonl),
        tokenizer,
        model_name,
        max_length=max_length,
        predictors=predictors,
        regular=False,
        linear=linear,
    )


@torch.no_grad()
def compute_perplexity(
    checkpoint_dir: Path,
    eval_jsonl: Path,
    base_model_name: str,
    method: str,
    max_length: int = 512,
    batch_size: int = 4,
    predictors: int = 0,
) -> float:
    """Mean NLL over label tokens (labels != -100); returns PPL = exp(nll)."""
    checkpoint_dir = Path(checkpoint_dir)
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_dir), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(checkpoint_dir),
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    model.eval()

    ds = build_eval_dataset(
        eval_jsonl, tokenizer, base_model_name, max_length, method, predictors=predictors
    )
    cols = ["input_ids", "labels", "attention_mask"]
    ds = ds.remove_columns([c for c in ds.column_names if c not in cols])

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, pad_to_multiple_of=None)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collator)

    device = next(model.parameters()).device
    total_nll = 0.0
    total_tokens = 0

    for batch in tqdm(loader, desc=f"PPL[{method}]"):
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch["labels"]
        out = model(**batch)
        # HF returns mean loss over non-ignored labels in the batch
        n_tokens = (labels != -100).sum().item()
        if n_tokens == 0:
            continue
        # Recover sum of NLL: loss is mean over valid positions in the batch
        total_nll += out.loss.item() * n_tokens
        total_tokens += n_tokens

    if total_tokens == 0:
        return float("inf")
    mean_nll = total_nll / total_tokens
    return math.exp(mean_nll)


def run_torchrun_stp(
    *,
    stp_py: Path,
    output_dir: Path,
    train_file: Path,
    model_name: str,
    num_epochs: int,
    learning_rate: float,
    finetune_seed: int,
    batch_size: int,
    grad_accum: int,
    max_length: int,
    nproc: int,
    method: str,
    last_token: int,
    lbd: float,
    predictors: int,
    tube_gamma: float = 0.95,
    tube_tau: float = 1e-3,
    lbd_ts: float = 0.0,
    tube_log_interval: int = 50,
    single_gpu: bool = False,
    gpu_id: int = 0,
):
    """Invoke training for one mode (torchrun or single-GPU python)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stp_args = [
        str(stp_py),
        "--train_file",
        str(train_file),
        "--output_dir",
        str(output_dir),
        "--num_epochs",
        str(num_epochs),
        "--finetune_seed",
        str(finetune_seed),
        "--model_name",
        model_name,
        "--learning_rate",
        str(learning_rate),
        "--batch_size",
        str(batch_size),
        "--grad_accum",
        str(grad_accum),
        "--max_length",
        str(max_length),
        "--last_token",
        str(last_token),
        "--lbd",
        str(lbd),
        "--predictors",
        str(predictors),
    ]

    if method == "regular":
        stp_args.append("--regular")
    elif method == "stp":
        # Match `run_stp.py` run_jepa and paper-style random-span cosine auxiliary (Fig. 1b regime).
        stp_args.extend(
            [
                "--linear",
                "random_span",
                # "--random_span_e2e",
            ]
        )
    elif method == "dynamics":
        stp_args.extend(
            [
                "--dynamics_tube",
                "--tube_gamma",
                str(tube_gamma),
                "--tube_tau",
                str(tube_tau),
                "--lbd_ts",
                str(lbd_ts),
                "--tube_log_interval",
                str(tube_log_interval),
            ]
        )
    else:
        raise ValueError(method)

    env = os.environ.copy()
    if single_gpu:
        env.pop("WORLD_SIZE", None)
        env.pop("RANK", None)
        env.pop("LOCAL_RANK", None)
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        cmd = [sys.executable] + stp_args
        print(f"\n>>> single-gpu mode (CUDA_VISIBLE_DEVICES={gpu_id})", flush=True)
        print(">>>", " ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=str(ROOT), env=env, check=True)
        return

    # Fresh port each subprocess: a shell-exported MASTER_PORT would otherwise be reused
    # across all three torchrun calls and can cause EADDRINUSE.
    master_addr = env.get("MASTER_ADDR", "127.0.0.1")
    master_port = str(_pick_free_port())
    env["MASTER_ADDR"] = master_addr
    env["MASTER_PORT"] = master_port

    rdzv_flags = [
        f"--master-addr={master_addr}",
        f"--master-port={master_port}",
    ]
    if shutil.which("torchrun"):
        cmd = ["torchrun", f"--nproc_per_node={nproc}"] + rdzv_flags + stp_args
    else:
        cmd = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            f"--nproc_per_node={nproc}",
        ] + rdzv_flags + stp_args

    print(
        f"\n>>> rendezvous {master_addr}:{master_port} (set MASTER_ADDR/MASTER_PORT to override)",
        flush=True,
    )
    print(">>>", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(ROOT), env=env, check=True)


def parse_args():
    p = argparse.ArgumentParser(description="Compare regular vs STP vs dynamics training via PPL.")
    p.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    p.add_argument("--dataset_name", type=str, default="synth", help="Stem; uses {data_prefix}{name}_train.jsonl")
    p.add_argument("--data_prefix", type=str, default="datasets/", help="Prefix for train/test JSONL paths.")
    p.add_argument("--output_root", type=str, default="compare_three_runs_vgeo2", help="Directory for three checkpoints.")
    p.add_argument("--num_epochs", type=int, default=6)
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--finetune_seed", type=int, default=42)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--nproc", type=int, default=2, help="torchrun --nproc_per_node")
    p.add_argument("--single_gpu", action="store_true", help="Train with plain python on one GPU (no torchrun).")
    p.add_argument("--gpu_id", type=int, default=2, help="GPU id used when --single_gpu is set.")
    p.add_argument("--last_token", type=int, default=-2)
    p.add_argument("--lbd", type=float, default=0.02, help="Aux loss weight for regular (unused) / stp.")
    p.add_argument(
        "--lbd_dynamics",
        type=float,
        default=0.05,
        help="Aux weight for dynamics tube only; ablation suggests sweeping ~0.01–0.05.",
    )
    p.add_argument("--tube_gamma", type=float, default=0.9, help="Dynamics: Lyapunov decay factor (passed to stp.py).")
    p.add_argument(
        "--tube_tau",
        type=float,
        default=1e-4,
        help="Dynamics: tube slack tau (passed to stp.py). Larger tau relaxes V_{t+1} <= γ V_t + τ.",
    )
    p.add_argument(
        "--lbd_ts",
        type=float,
        default=0.0,
        help="Dynamics: temporal-straightening curvature weight; default 0 for pure Lyapunov-tube ablation.",
    )
    p.add_argument(
        "--tube_log_interval",
        type=int,
        default=50,
        help="Dynamics: print [tube_diag] every N steps (0 = off). Passed to stp.py.",
    )
    p.add_argument(
        "--only_dynamics",
        action="store_true",
        help="Train and compute PPL only for dynamics (skip regular and STP).",
    )
    p.add_argument("--predictors", type=int, default=0)
    p.add_argument("--eval_batch_size", type=int, default=4, help="Batch size for PPL only.")
    p.add_argument("--skip_train", action="store_true", help="Only compute PPL on existing dirs.")
    p.add_argument(
        "--ckpt_regular",
        type=str,
        default="",
        help="Optional checkpoint path for a regular baseline.",
    )
    p.add_argument("--ckpt_stp", type=str, default=None)
    p.add_argument("--ckpt_dynamics", type=str, default=None)
    p.add_argument(
        "--data_fraction",
        type=float,
        default=1.0,
        help="Random subsample of train JSONL (seeded) before training; 1.0 = full train.",
    )
    p.add_argument("--train_subset_seed", type=int, default=42, help="Seed for --data_fraction shuffle.")
    p.add_argument(
        "--eval_accuracy",
        action="store_true",
        default=True,
        help="Greedy exact-match accuracy vs gold assistant (eval_testset).",
    )
    p.add_argument(
        "--eval_token_accuracy",
        action="store_true",
        help="Teacher-forced next-token accuracy on supervised labels.",
    )
    p.add_argument(
        "--eval_snr_proxy",
        action="store_true",
        default=True,
        help="Tube geometry SNR: mean parallel^2 / perp^2 (STP signal/noise picture).",
    )
    p.add_argument("--max_gen_eval", type=int, default=500, help="Max examples for --eval_accuracy.")
    p.add_argument(
        "--eval_profile",
        type=str,
        default="fork",
        choices=("fork", "llm_jepa_official"),
        help=(
            "Greedy exact-match generation: fork uses min(256,max_length) new tokens; "
            "llm_jepa_official uses reference defaults (128 new, 512 total)."
        ),
    )
    p.add_argument("--snr_max_batches", type=int, default=80, help="Max batches for --eval_snr_proxy.")
    p.add_argument(
        "--token_acc_max_batches",
        type=int,
        default=0,
        help="Max batches for token accuracy; 0 = full eval set.",
    )
    return p.parse_args()


def _read_log_history(state_path: Path) -> list[dict]:
    if not state_path.is_file():
        return []
    try:
        with open(state_path) as f:
            state = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"Warning: could not read {state_path}: {e}")
        return []
    return list(state.get("log_history", []))


def _series(history: list[dict], key: str) -> tuple[list[int], list[float]]:
    steps: list[int] = []
    values: list[float] = []
    for rec in history:
        if key in rec and rec.get("step") is not None:
            try:
                steps.append(int(rec["step"]))
                values.append(float(rec[key]))
            except (TypeError, ValueError):
                continue
    return steps, values


def plot_training_curves(method_dirs: dict, out_path: Path) -> None:
    """Save a PNG comparing L_NTP and each method's auxiliary loss over training steps.

    Reads each method's `trainer_state.json` (produced by HF Trainer.save_state)
    and plots lm_loss (NTP) and jepa_loss (STP cosine / Lyapunov tube) per
    logging step. Requires matplotlib; silently skips if not installed.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping loss-curve plot.")
        return

    aux_label = {"stp": r"$\mathcal{L}_{\mathrm{STP}}$", "dynamics": r"$\mathcal{L}_{\mathrm{Lyap}}$"}
    method_label = {"stp": "STP", "dynamics": "Dynamics"}
    lm_color = {"stp": "tab:blue", "dynamics": "tab:green"}
    aux_color = {"stp": "tab:orange", "dynamics": "tab:red"}

    fig, ax = plt.subplots(figsize=(9, 5.5))
    any_plotted = False
    for method in ("stp", "dynamics"):
        ckpt = method_dirs.get(method)
        if ckpt is None:
            continue
        history = _read_log_history(Path(ckpt) / "trainer_state.json")
        if not history:
            print(f"No trainer_state.json/log_history for method={method} at {ckpt}.")
            continue
        steps_lm, lm_vals = _series(history, "lm_loss")
        steps_aux, aux_vals = _series(history, "jepa_loss")
        if steps_lm:
            ax.plot(
                steps_lm,
                lm_vals,
                color=lm_color[method],
                linewidth=2,
                label=rf"$\mathcal{{L}}_{{\mathrm{{NTP}}}}$ ({method_label[method]})",
            )
            any_plotted = True
        if steps_aux:
            ax.plot(
                steps_aux,
                aux_vals,
                color=aux_color[method],
                linewidth=2,
                linestyle="--",
                label=f"{aux_label[method]} ({method_label[method]})",
            )
            any_plotted = True

    if not any_plotted:
        print("No loss curves available to plot; skipping figure.")
        plt.close(fig)
        return

    ax.set_xlabel("Steps")
    ax.set_ylabel("Loss")
    ax.set_title("Training loss curves: STP vs Dynamics (Lyapunov tube)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def main():
    args = parse_args()
    prefix = args.data_prefix
    train_file = ROOT / f"{prefix}{args.dataset_name}_train.jsonl".replace("//", "/")
    eval_file = ROOT / f"{prefix}{args.dataset_name}_test.jsonl".replace("//", "/")
    if not train_file.is_file():
        sys.exit(f"Missing train file: {train_file}")
    if not eval_file.is_file():
        sys.exit(f"Missing eval file: {eval_file}")

    out_root = ROOT / args.output_root
    out_root.mkdir(parents=True, exist_ok=True)
    dirs = {
        "regular": Path(args.ckpt_regular) if args.ckpt_regular else out_root / "regular",
        "stp": Path(args.ckpt_stp) if args.ckpt_stp else out_root / "stp",
        "dynamics": Path(args.ckpt_dynamics) if args.ckpt_dynamics else out_root / "dynamics",
    }

    stp_py = ROOT / "stp.py"
    if not stp_py.is_file():
        sys.exit(f"Missing {stp_py}")

    # Skip training the `regular` (NTP-only) method; keep it in PPL evaluation
    # only when a checkpoint directory is already present on disk.
    if args.only_dynamics:
        train_methods: tuple[str, ...] = ("dynamics",)
    else:
        train_methods = ("dynamics", "stp")
    if args.only_dynamics:
        ppl_methods: tuple[str, ...] = ("dynamics",)
    else:
        ppl_methods = ("dynamics", "stp")
        if dirs["regular"].is_dir():
            ppl_methods = ppl_methods + ("regular",)

    effective_train = train_file
    if args.data_fraction < 1.0 - 1e-9:
        effective_train = out_root / (
            f"_train_subset_{args.dataset_name}_f{args.data_fraction}_s{args.train_subset_seed}.jsonl"
        )
        effective_train = stp_eval_metrics.materialize_train_fraction(
            train_file, effective_train, args.data_fraction, seed=args.train_subset_seed
        )
        print(f"Using training subset ({args.data_fraction:.4f} of lines): {effective_train}", flush=True)

    if not args.skip_train:
        for method in train_methods:
            lbd = args.lbd_dynamics if method == "dynamics" else args.lbd
            run_torchrun_stp(
                stp_py=stp_py,
                output_dir=dirs[method],
                train_file=effective_train,
                model_name=args.model_name,
                num_epochs=args.num_epochs,
                learning_rate=args.learning_rate,
                finetune_seed=args.finetune_seed,
                batch_size=args.batch_size,
                grad_accum=args.grad_accum,
                max_length=args.max_length,
                nproc=args.nproc,
                method=method,
                last_token=args.last_token,
                lbd=lbd,
                predictors=args.predictors,
                tube_gamma=args.tube_gamma,
                tube_tau=args.tube_tau,
                lbd_ts=args.lbd_ts,
                tube_log_interval=args.tube_log_interval,
                single_gpu=args.single_gpu,
                gpu_id=args.gpu_id,
            )

    results = {}
    for method in ppl_methods:
        ckpt = dirs[method]
        if not ckpt.is_dir():
            print(f"Skip PPL: missing checkpoint dir {ckpt}")
            results[method] = None
            continue
        ppl = compute_perplexity(
            ckpt,
            eval_file,
            args.model_name,
            method,
            max_length=args.max_length,
            batch_size=args.eval_batch_size,
            predictors=args.predictors,
        )
        results[method] = ppl

    print("\n========== Perplexity comparison ==========")
    for m in ppl_methods:
        v = results.get(m)
        print(f"  {m:12s}  PPL = {v if v is not None else 'N/A'}")

    acc_results: dict = {}
    tok_results: dict = {}
    snr_results: dict = {}
    tok_max = None if args.token_acc_max_batches <= 0 else args.token_acc_max_batches

    if args.eval_accuracy:
        print("\n========== Exact-match accuracy (greedy) ==========")
        for m in ppl_methods:
            ckpt = dirs[m]
            if not ckpt.is_dir():
                acc_results[m] = None
                continue
            acc = stp_eval_metrics.compute_exact_match_accuracy(
                ckpt,
                eval_file,
                args.model_name,
                max_new_tokens=min(256, args.max_length),
                max_length=args.max_length,
                max_examples=args.max_gen_eval,
                eval_profile=args.eval_profile,
            )
            acc_results[m] = acc
            print(f"  {m:12s}  acc = {acc:.4f}")

    if args.eval_token_accuracy:
        print("\n========== Teacher-forced token accuracy ==========")
        for m in ppl_methods:
            ckpt = dirs[m]
            if not ckpt.is_dir():
                tok_results[m] = None
                continue
            tacc = stp_eval_metrics.compute_teacher_forced_token_accuracy(
                ckpt,
                eval_file,
                args.model_name,
                m,
                max_length=args.max_length,
                batch_size=args.eval_batch_size,
                predictors=args.predictors,
                max_batches=tok_max,
            )
            tok_results[m] = tacc
            if tacc == tacc:
                print(f"  {m:12s}  token_acc = {tacc:.4f}")
            else:
                print(f"  {m:12s}  token_acc = nan")

    if args.eval_snr_proxy:
        print("\n========== Tube SNR proxy (eval set, mean over batches) ==========")
        for m in ppl_methods:
            ckpt = dirs[m]
            if not ckpt.is_dir():
                snr_results[m] = None
                continue
            snr = stp_eval_metrics.compute_tube_snr_proxy(
                ckpt,
                eval_file,
                args.model_name,
                max_length=args.max_length,
                batch_size=max(1, args.eval_batch_size // 2),
                predictors=args.predictors,
                max_batches=args.snr_max_batches,
            )
            snr_results[m] = snr
            print(f"  {m:12s}  SNR_db ≈ {snr.get('snr_db', float('nan')):.3f}")

    summary_path = out_root / "ppl_comparison.json"
    summary = {
        "model_name": args.model_name,
        "dataset": args.dataset_name,
        "eval_file": str(eval_file),
        "train_file_effective": str(effective_train),
        "data_fraction": args.data_fraction,
        "train_subset_seed": args.train_subset_seed,
        "num_epochs": args.num_epochs,
        "tube_gamma": args.tube_gamma,
        "lbd_dynamics": args.lbd_dynamics,
        "tube_tau": args.tube_tau,
        "ppl": {k: (float(v) if v is not None and not math.isinf(v) else None) for k, v in results.items()},
    }
    if acc_results:
        summary["exact_match_accuracy"] = {k: (float(v) if v is not None and v == v else None) for k, v in acc_results.items()}
    if tok_results:
        summary["teacher_forced_token_accuracy"] = {
            k: (float(v) if v is not None and v == v else None) for k, v in tok_results.items()
        }
    if snr_results:
        summary["snr_proxy"] = snr_results

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {summary_path}")

    plot_training_curves(
        {m: dirs[m] for m in ("stp", "dynamics") if m in dirs},
        out_root / "loss_curves.png",
    )


if __name__ == "__main__":
    main()
