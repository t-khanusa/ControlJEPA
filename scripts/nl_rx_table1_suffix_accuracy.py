#!/usr/bin/env python3
"""
Batch test all ft-c-v_geo-synth-* checkpoints in /root/khanhnt/Llama3.2-1B-Instruct.

Runs the NL-RX Table 1 style greedy exact-match accuracy by gold suffix for each checkpoint found.

Example usage (Conda environment):
    conda activate controlJEPA
    python scripts/nl_rx_table1_suffix_accuracy.py

Or (from any env):
    conda run -n controlJEPA -- python scripts/nl_rx_table1_suffix_accuracy.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
import argparse

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

def _count_bins_jsonl(train_jsonl: Path) -> dict[str, int]:
    import json as _json
    from eval_testset import nlrx_star_suffix_bin

    c = {"suffix_one_star": 0, "suffix_multi_star": 0, "other": 0}
    for ln in train_jsonl.read_text().splitlines():
        ln = ln.strip()
        if not ln:
            continue
        gold = _json.loads(ln)["messages"][2]["content"]
        c[nlrx_star_suffix_bin(gold)] += 1
    return c

def _pct(x: float) -> str:
    if x != x:
        return "nan"
    return f"{100.0 * x:.1f}%"

def run_eval_for_checkpoint(
    ck: Path,
    eval_path: Path,
    train_path: Path,
    base_model_name: str,
    max_new_tokens: int,
    max_length: int,
    max_examples: int | None,
    eval_profile: str = "fork",
    paper_two_rows: bool = False,
    column_label: str = "",
    json_out_dir: Path | None = None,
):
    from eval_testset import compute_nlrx_star_suffix_exact_match

    metrics = compute_nlrx_star_suffix_exact_match(
        ck,
        eval_path,
        base_model_name,
        max_new_tokens=max_new_tokens,
        max_length=max_length,
        max_examples=max_examples,
        eval_profile=eval_profile,
    )

    col = column_label.strip() or ck.name
    dot = metrics["suffix_one_star"]["accuracy"]
    doubled = metrics["suffix_multi_star"]["accuracy"]
    train_ratio_note = ""
    if train_path and train_path.is_file():
        tr = _count_bins_jsonl(train_path)
        n1, nm = tr["suffix_one_star"], tr["suffix_multi_star"]
        ratio = (n1 / nm) if nm > 0 else float("nan")
        train_ratio_note = (
            f"(train gold: .* n={n1}, .*.*(+): n={nm}; .* is ~{ratio:.1f}x more frequent than .*.*(+))"
        )

    overall = metrics["overall"]["accuracy"]
    n_dot = int(metrics["suffix_one_star"]["count"])
    n_dbl = int(metrics["suffix_multi_star"]["count"])
    n_other = int(metrics["other"]["count"])

    print()
    print("Table 1 style — greedy strip-exact-match on NL-RX synth test")
    print(f"checkpoint: {ck}")
    print(f"base_model_name (prompt template): {base_model_name}")
    print(f"eval: {eval_path}")
    print(f"eval_profile: {eval_profile}")
    if train_ratio_note:
        print("Suffix prevalence " + train_ratio_note)
    print()
    hdr = "| Suffix | " + col + " |"
    sep = "| :--- | :--- |"
    row1 = f"| .* | {_pct(dot)} |  (test n={n_dot})"
    row2 = f"| .*.* | {_pct(doubled)} |  (test n={n_dbl}; ≥2 peeled trailing .*)"
    rowo = (
        "| other gold (no peeled .* suffix) | "
        f"{_pct(metrics['other']['accuracy'])} |  (test n={n_other})"
    )
    ov = "| **overall** | " + _pct(overall) + " |"
    print(hdr)
    print(sep)
    print(row1)
    print(row2)
    if not paper_two_rows:
        print(rowo)
        print(ov)

    blob = {
        "checkpoint": str(ck),
        "eval_jsonl": str(eval_path),
        "eval_profile": eval_profile,
        "column_label": col,
        "train_ratio_note_suffix_counts": {},
        "metrics": metrics,
    }
    if train_path and train_path.is_file():
        blob["train_ratio_note_suffix_counts"] = _count_bins_jsonl(train_path)

    if json_out_dir:
        json_out_dir.mkdir(parents=True, exist_ok=True)
        outp = json_out_dir / f"{col}.json"
        outp.write_text(json.dumps(blob, indent=2))
        print(f"\nWrote {outp}")

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--checkpoints_dir",
        type=str,
        default="/root/khanhnt/Llama3.2-1B-Instruct",
        help="Directory containing candidate checkpoints.",
    )
    p.add_argument(
        "--glob",
        type=str,
        default="ft-c-v_geo-synth-*",
        help="Glob pattern matching checkpoints to evaluate.",
    )
    p.add_argument(
        "--base_model_name",
        type=str,
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="Upstream model id used for chat template / prompting (ORIGINAL); see evaluate.py.",
    )
    p.add_argument(
        "--eval_jsonl",
        type=str,
        default=str(_REPO / "datasets" / "synth_test.jsonl"),
        help="SYNTH-format JSONL (messages...) for NL-RX eval.",
    )
    p.add_argument(
        "--train_jsonl",
        type=str,
        default=str(_REPO / "datasets" / "synth_train.jsonl"),
        help="Optional: count suffix imbalance versus training gold (caption text).",
    )
    p.add_argument("--max_examples", type=int, default=None, help="Cap eval lines for smoke tests.")
    p.add_argument(
        "--eval_profile",
        choices=("fork", "llm_jepa_official"),
        default="fork",
        help="fork=min(256,max_length) new tokens typical of this repo; llm_jepa_official=gallery defaults.",
    )
    p.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Used when eval_profile=fork.",
    )
    p.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Total context hint passed to GenerationConfig-equivalent fork path.",
    )
    p.add_argument(
        "--paper_two_rows",
        action="store_true",
        help="Print only the two paper rows (suffix .* and .*.*[+]); omit other/overall.",
    )
    p.add_argument("--json_out_dir", type=str, default="", help="Optional output directory for metrics JSON for each checkpoint.")
    args = p.parse_args()

    checkpoints_dir = Path(args.checkpoints_dir).resolve()
    if not checkpoints_dir.is_dir():
        sys.exit(f"Missing checkpoints directory: {checkpoints_dir}")

    eval_path = Path(args.eval_jsonl).resolve()
    if not eval_path.is_file():
        sys.exit(f"Missing eval JSONL: {eval_path}")
    train_path = Path(args.train_jsonl).resolve()

    ck_glob = args.glob
    matched = sorted(checkpoints_dir.glob(ck_glob))
    if not matched:
        print(f"No checkpoints matching '{ck_glob}' found in {checkpoints_dir}")
        return 1

    out_dir = Path(args.json_out_dir).resolve() if args.json_out_dir else None

    print(f"Found {len(matched)} checkpoint(s):")
    for ck in matched:
        print(f" - {ck}")
    print("\nStarting evaluation...\n")
    for ck in matched:
        run_eval_for_checkpoint(
            ck=ck,
            eval_path=eval_path,
            train_path=train_path,
            base_model_name=args.base_model_name,
            max_new_tokens=args.max_new_tokens,
            max_length=args.max_length,
            max_examples=args.max_examples,
            eval_profile=args.eval_profile,
            paper_two_rows=args.paper_two_rows,
            column_label=ck.name,
            json_out_dir=out_dir,
        )
        print("\n" + "=" * 80 + "\n")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
