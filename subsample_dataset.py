"""Create subsampled training datasets for data efficiency experiments.

Given a JSONL training file, creates subsets at fractions 1/2, 1/4, 1/8,
1/16, 1/32. Sampling is seeded for reproducibility; larger fractions are
strict supersets of smaller ones (nested sampling).
"""

import argparse
import json
import os
import random


def subsample(input_file: str, output_dir: str, fractions: list[float],
              seed: int = 42) -> list[str]:
    with open(input_file) as f:
        lines = f.readlines()
    n = len(lines)

    rng = random.Random(seed)
    indices = list(range(n))
    rng.shuffle(indices)

    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(input_file))[0]
    created = []

    for frac in sorted(fractions, reverse=True):
        k = max(1, int(round(n * frac)))
        subset_idx = sorted(indices[:k])
        tag = _frac_tag(frac)
        out_path = os.path.join(output_dir, f"{base}_frac{tag}.jsonl")
        with open(out_path, "w") as f:
            for i in subset_idx:
                f.write(lines[i])
        created.append(out_path)
        print(f"  {out_path}: {k} examples (fraction={frac})")

    return created


def _frac_tag(frac: float) -> str:
    inv = round(1.0 / frac)
    if abs(frac * inv - 1.0) < 1e-9:
        return f"1_{inv}"
    return f"{frac:.4f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Defaults to same directory as input_file")
    parser.add_argument("--fractions", type=float, nargs="+",
                        default=[0.5, 0.25, 0.125, 0.0625, 0.03125])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = args.output_dir or os.path.dirname(args.input_file) or "."
    print(f"Subsampling {args.input_file} -> {out_dir}")
    subsample(args.input_file, out_dir, args.fractions, seed=args.seed)
    print("Done.")


if __name__ == "__main__":
    main()
