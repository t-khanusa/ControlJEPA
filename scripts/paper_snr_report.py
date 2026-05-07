#!/usr/bin/env python3
"""Pretty-print tube geometric SNR + Gaussian surrogates.

This is a thin wrapper around ``diagnose_signal_noise.measure_one_model``: it runs a
forward-only pass over JSONL chats, aggregates chord‑projection energies, and echoes
information‑theoretic quantities **conditionally** interpreting geometric SNR as the
Gaussian‑channel SNR in Appendix H (**explicit proxy**, not latent identifiability).

Example:
  conda run -n controlJEPA python scripts/paper_snr_report.py \\
    --ckpt Llama3.2-1B-Instruct/ft-c-v_geo-synth-g0.80-t1e-3-2e-5-0.01-0-82 \\
    --base_model meta-llama/Llama-3.2-1B-Instruct \\
    --data datasets/synth_test.jsonl --n 128 --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Repo root (parent of scripts/) — ``diagnose_signal_noise`` lives alongside ``scripts/``, not installed as a package
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from diagnose_signal_noise import load_examples, measure_one_model  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--base_model", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    ap.add_argument("--data", type=str, default="datasets/synth_test.jsonl")
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--layer", type=int, default=-1)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--save_json", type=str, default="")
    args = ap.parse_args()

    examples = load_examples(Path(args.data), args.n)
    stats = measure_one_model(
        Path(args.ckpt),
        args.base_model,
        examples,
        max_length=args.max_length,
        device=args.device,
        layer=args.layer,
    )

    geom_keys = (
        "snr_p_over_e",
        "mean_p2_signal",
        "mean_e2_noise",
        "mean_sin2_theta",
        "mean_vgeo2",
        "mean_h2",
        "n_tokens",
        "n_examples",
    )
    apx_keys = sorted(k for k in stats if k.startswith("appx_"))

    print("\n=== Geometric Semantic Tube SNR proxy (Fig. 1 narration) ===")
    for k in geom_keys:
        if k in stats:
            print(f"  {k}: {stats[k]}")

    print("\n=== Appendix H Gaussian surrogates (plug-in from SNR_geom; NOT latent ID) ===")
    for k in apx_keys:
        print(f"  {k}: {stats[k]}")

    print(
        "\nNote: Appendix H stresses latent SNR is intractable; interpret "
        "`appx_*` only qualitatively as ‘if SNR_geom acted like 𝔼‖Z‖²/𝔼‖N‖²’."
    )

    if args.save_json:
        p = Path(args.save_json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(stats, indent=2))
        print(f"\nSaved: {p}")


if __name__ == "__main__":
    main()
