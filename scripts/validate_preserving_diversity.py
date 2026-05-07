#!/usr/bin/env python3
"""
Validate Polymorphism / Preserving Diversity helpers in ``stp.py``.

Use the project's Conda environment (activate first)::

    conda activate controlJEPA

Or::

    bash scripts/run_under_control_jepa.sh python scripts/validate_preserving_diversity.py

Checks:

  * ``chord_difference_text_minus_code``: shape / finite / non-zero grads.
  * ``polymorphism_decorrelation_loss``: finite; gradient flows through ``hidden_states``.
  * Optional SVD diagnostics on synthetic normalized ``delta`` rows (purely illustrative).
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device for autograd smoke tests (e.g. cuda:0 when available).",
    )
    parser.add_argument(
        "--toy-svd",
        action="store_true",
        help="Singular-value snapshot for collapsed vs varied normalized delta rows.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help=(
            "Optional HF checkpoint dir: NL-RX trailing .* vs .*.* split accuracy on "
            "datasets/synth_test.jsonl via eval_testset (slow; GPU recommended)."
        ),
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="Chat-template baseline name when using --checkpoint.",
    )
    args = parser.parse_args()

    import torch

    from stp import chord_difference_text_minus_code, polymorphism_decorrelation_loss

    device = torch.device(args.device)
    dtype = torch.float64

    torch.manual_seed(7)

    B, T, D = 8, 30, 32
    user_se = torch.tensor([[5, 9]] * B, dtype=torch.long, device=device)
    asst_se = torch.tensor([[14, 19]] * B, dtype=torch.long, device=device)

    hidden = torch.randn(B, T, D, device=device, dtype=dtype, requires_grad=True)
    delta = chord_difference_text_minus_code(hidden, user_se, asst_se)

    assert delta.shape == (B, D), f"unexpected delta shape {delta.shape}"

    proj = torch.randn(D, 8, device=device, dtype=dtype)
    proj = torch.nn.functional.normalize(proj, dim=0, eps=1e-12)

    diversity = polymorphism_decorrelation_loss(delta, proj)
    diversity.backward()

    assert hidden.grad is not None, "expected grads on hidden_states"
    gst = float(hidden.grad.detach().float().abs().sum())
    assert math.isfinite(gst) and gst > 0.0, "non-finite or zero aggregate grad"

    print(f"polymorphism_decorrelation_loss={diversity.detach().float().item():.6f}")
    print(f"sum|grad(hidden)|={gst:.6e}")
    print("chord_difference + diversity backward: OK")

    # Second forward/backward sanity (reuse graph-free tensors)
    h2 = torch.randn(B, T, D, device=device, dtype=dtype, requires_grad=True)
    delta2 = chord_difference_text_minus_code(h2, user_se, asst_se)
    L2 = polymorphism_decorrelation_loss(delta2, proj)
    assert L2.shape == ()

    if args.toy_svd:
        import numpy as np

        np.random.seed(0)
        v = torch.randn(D, dtype=dtype)
        collapsed = torch.nn.functional.normalize(v, dim=-1).unsqueeze(0).expand(B, -1)
        zn_c = collapsed.float().detach().cpu().numpy()
        diverse = torch.randn(B, D, dtype=dtype)
        zn_d = torch.nn.functional.normalize(diverse, dim=1).float().detach().cpu().numpy()
        _, sv_c, _ = np.linalg.svd(zn_c, full_matrices=False)
        _, sv_d, _ = np.linalg.svd(zn_d, full_matrices=False)
        print("toy-SVD singular values top-8 (illus.)")
        print("  collapsed batch rows (~rank-1):", sv_c[:8])
        print("  varied batch rows (~full rank):", sv_d[:8])

    ck = args.checkpoint.strip()
    if ck:
        test_json = _PROJECT_ROOT / "datasets" / "synth_test.jsonl"
        if not test_json.is_file():
            print(f"SKIP --checkpoint nlrx drill: missing {test_json}")
        else:
            from eval_testset import compute_nlrx_star_suffix_exact_match

            metrics = compute_nlrx_star_suffix_exact_match(
                Path(ck).resolve(),
                test_json,
                args.model_name,
                max_new_tokens=64,
                max_length=256,
                max_examples=32,
                eval_profile="fork",
            )
            print("--checkpoint nlrx stratified greedy EM:", metrics)

    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
